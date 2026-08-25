"""Trajectory optimization with Poll-Schumann Aircraft Performance and climate term.

See :class:`Optimizer` for the main entrypoint.
"""

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pycontrails import Flight, JetA, MetDataArray, MetDataset
from pycontrails.core import airports
from pycontrails.models.ps_model import ps_aircraft_params
from pycontrails.physics import geo, jet, units

from contrailopt import ps, slerp
from contrailopt.dag import (
    AirportCoords,
    EdgeMetLookup,
    HorizontalDAG,
    Track,
    validate_flight_profile,
    _neighborhood_edges,
)
from contrailopt.grid_utils import flight_profile_from_met

if TYPE_CHECKING:
    from cartopy.mpl.geoaxes import GeoAxes
    from matplotlib.animation import FuncAnimation

FLOAT_DTYPE = np.float32

_SEGMENT_DTYPE = np.dtype(
    [
        ("longitude", FLOAT_DTYPE),
        ("latitude", FLOAT_DTYPE),
        ("altitude_ft", FLOAT_DTYPE),
        ("elapsed_s", FLOAT_DTYPE),
        ("cost", FLOAT_DTYPE),
        ("fuel", FLOAT_DTYPE),
        ("eef", FLOAT_DTYPE),
    ]
)

_WAYPOINT_DTYPE = np.dtype(
    [
        ("longitude", FLOAT_DTYPE),
        ("latitude", FLOAT_DTYPE),
        ("altitude_ft", FLOAT_DTYPE),
        ("elapsed_s", FLOAT_DTYPE),
        ("mach_number", FLOAT_DTYPE),
        ("eef_per_m", FLOAT_DTYPE),
        ("air_temperature", FLOAT_DTYPE),
        ("tailwind", FLOAT_DTYPE),
        ("node_index", np.int64),
        ("sample_index", np.int64),
    ]
)

# Takeoff mass tolerance for the convergence loop in Optimizer.solve.
MASS_CONVERGENCE_KG = 20.0

# IPCC AR5 reference AGWP100: https://www.ipcc.ch/site/assets/uploads/2018/07/WGI_AR5.Chap_.8_SM.pdf
_AGWP_CO2 = 91.7e-15  # W m-2 yr kg-1
_SECONDS_PER_YEAR = 60 * 60 * 24 * 365  # s yr-1
_SURFACE_AREA_EARTH = 5.101e14  # m2
_J_PER_KG_CO2 = _AGWP_CO2 * _SURFACE_AREA_EARTH * _SECONDS_PER_YEAR  # J kg-1
J_PER_TONNE_CO2 = _J_PER_KG_CO2 * 1000.0  # J tonne-1


def _expand_edge_samples(
    met_lookup: EdgeMetLookup,
    flat_edge_idx: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Gather sample point indices for a batch of edges."""
    n_edge = len(flat_edge_idx)
    edge_ptr = met_lookup.edge_ptr
    starts = edge_ptr[flat_edge_idx]
    n_samp = edge_ptr[flat_edge_idx + 1] - starts
    cumsum = np.cumsum(n_samp)

    offset = np.arange(n_samp.sum()) - np.repeat(cumsum - n_samp, n_samp)
    sample_idxs = np.repeat(starts, n_samp) + offset

    sample_to_edge = np.repeat(np.arange(n_edge, dtype=np.int64), n_samp)
    edge_bounds = cumsum - n_samp
    return sample_idxs, sample_to_edge, edge_bounds


def _estimate_sample_times(
    met_lookup: EdgeMetLookup,
    sample_idxs: npt.NDArray[np.int64],
    sample_to_edge: npt.NDArray[np.int64],
    src_idx: npt.NDArray[np.int64],
    takeoff_time: pd.Timestamp,
    src_elapsed: npt.NDArray[FLOAT_DTYPE],
    climb_time: npt.NDArray[FLOAT_DTYPE],
    fl_choices: npt.NDArray[FLOAT_DTYPE],
    mach_choices: npt.NDArray[FLOAT_DTYPE],
) -> npt.NDArray[np.datetime64]:
    """Estimate arrival datetime64 at each sample point per FL.

    Returns shape ``(n_sample, n_fl)``.
    """
    # Per-FL climb time for each sample's edge: (n_sample, n_fl)
    sample_climb_s = climb_time[sample_to_edge]
    edge_start_s = src_elapsed[src_idx[sample_to_edge], np.newaxis] + sample_climb_s

    # Per-FL TAS estimate using mean Mach and ISA temperature at each FL
    T_isa = units.m_to_T_isa(units.ft_to_m(fl_choices))  # (n_fl,)
    approx_tas = units.mach_number_to_tas(mach_choices.mean(), T_isa)  # (n_fl,)
    offset_s = met_lookup.cum_dist[sample_idxs, np.newaxis] / approx_tas[np.newaxis, :]

    total_s = edge_start_s + offset_s
    return np.datetime64(takeoff_time) + (total_s * 1e9).astype("timedelta64[ns]")


def _cruise_zone_weights(
    met_lookup: EdgeMetLookup,
    sample_idxs: npt.NDArray[np.int64],
    sample_to_edge: npt.NDArray[np.int64],
    climb_dist: npt.NDArray[FLOAT_DTYPE],
    descent_dd: npt.NDArray[FLOAT_DTYPE],
    flat_dist: npt.NDArray[FLOAT_DTYPE],
) -> npt.NDArray[FLOAT_DTYPE]:
    """Fraction of each sample segment that lies within the cruise zone.

    Each edge is partitioned into climb, cruise, and descent zones by
    distance from the edge source. The cruise zone spans from
    ``climb_dist`` to ``flat_dist - descent_dd``, and these boundaries
    vary by FL. Each sample covers the segment ``[cum_dist, cum_dist +
    delta_dist]``. The returned weight is the fraction of that segment
    overlapping the cruise zone:

    - 0.0 for samples entirely in climb or descent
    - 1.0 for samples entirely in cruise
    - a value in (0, 1) for samples straddling a boundary

    Returns shape ``(n_sample, n_fl)``.
    """
    seg_dist = met_lookup.delta_dist[sample_idxs][:, np.newaxis]  # (n_sample, 1)
    cum_dist = met_lookup.cum_dist[sample_idxs][:, np.newaxis]  # (n_sample, 1)
    cruise_lo = climb_dist[sample_to_edge]  # (n_sample, n_fl)
    cruise_hi = flat_dist[sample_to_edge, np.newaxis] - descent_dd[sample_to_edge]
    seg_end = cum_dist + seg_dist
    overlap = np.clip(np.minimum(seg_end, cruise_hi) - np.maximum(cum_dist, cruise_lo), 0.0, None)
    return np.divide(overlap, seg_dist, out=np.zeros_like(overlap), where=seg_dist > 0.0)


def _climbing_fl_idxs(
    dist: npt.NDArray[FLOAT_DTYPE],
    climb_dist: npt.NDArray[FLOAT_DTYPE],
    src_alt_ft: npt.NDArray[FLOAT_DTYPE],
    tgt_alt_ft: npt.NDArray[FLOAT_DTYPE],
    fl_choices: npt.NDArray[FLOAT_DTYPE],
) -> npt.NDArray[np.int64]:
    """Find the flight level closest to the aircraft's altitude partway through a climb.

    This function assumes the altitude increases linearly over the climb.

    Returns
    -------
    npt.NDArray[np.int64]
        Index into ``fl_choices``, shaped like the broadcast of the other arguments.
    """
    climbed = np.divide(
        dist,
        climb_dist,
        out=np.ones_like(climb_dist),
        where=climb_dist > 0.0,
    )
    np.minimum(climbed, 1.0, out=climbed)  # hold the target level beyond the climb
    altitude_ft = src_alt_ft + (tgt_alt_ft - src_alt_ft) * climbed

    upper = np.clip(np.searchsorted(fl_choices, altitude_ft), 1, len(fl_choices) - 1)
    lower = upper - 1
    below = altitude_ft - fl_choices[lower] <= fl_choices[upper] - altitude_ft

    return np.where(below, lower, upper)


def _calculate_cruise_at_samples(
    met_lookup: EdgeMetLookup,
    flat_edge_idx: npt.NDArray[np.int64],
    src_idx: npt.NDArray[np.int64],
    fl_choices: npt.NDArray[FLOAT_DTYPE],
    mach_choices: npt.NDArray[FLOAT_DTYPE],
    post_climb_mass: npt.NDArray[FLOAT_DTYPE],
    atyp: ps_aircraft_params.PSAircraftEngineParams,
    takeoff_time: pd.Timestamp,
    src_elapsed: npt.NDArray[FLOAT_DTYPE],
    climb_time: npt.NDArray[FLOAT_DTYPE],
    climb_dist: npt.NDArray[FLOAT_DTYPE],
    descent_dd: npt.NDArray[FLOAT_DTYPE],
    flat_dist: npt.NDArray[FLOAT_DTYPE],
    src_alt_ft: npt.NDArray[FLOAT_DTYPE],
) -> tuple[
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[np.bool_],
    npt.NDArray[FLOAT_DTYPE],
]:
    """Compute per-edge cruise fuel, time, feasibility, and EEF from met samples.

    Met data is interpolated at each sample point along the edge for every
    candidate FL in ``fl_choices``. Cruise performance (fuel flow, TAS, wind)
    is evaluated at each candidate FL (the destination node FL, not the source node FL),
    searching over ``mach_choices``. For step-climbs, the cruise-zone weighting zeros out the
    climb portion so that only the candidate level flight segment contributes to fuel and time. For
    step-descents, climb_dist is zero, so the full edge is treated as cruise at the candidate FL.

    The EEF term (``eef_per_m * delta_dist``) is accumulated over the full
    edge without cruise-zone weighting because we want contrail forcing to apply
    regardless of whether the aircraft is cruising or climbing/descending along the edge.

    Parameters
    ----------
    met_lookup : EdgeMetLookup
        Pre-built met interpolator.
    flat_edge_idx : npt.NDArray[np.int64]
        Global edge index for each edge. Shape ``(n_edge,)``.
    src_idx : npt.NDArray[np.int64]
        Maps each edge to its source in the active-source arrays. Shape ``(n_edge,)``.
    fl_choices : npt.NDArray[FLOAT_DTYPE]
        Candidate FLs in feet. Shape ``(n_fl,)``.
    mach_choices : npt.NDArray[FLOAT_DTYPE]
        Candidate Mach numbers. Shape ``(n_mach,)``.
    post_climb_mass : npt.NDArray[FLOAT_DTYPE]
        Mass after climb for each edge and FL. Shape ``(n_edge, n_fl)``.
    atyp : ps_aircraft_params.PSAircraftEngineParams
        Aircraft/engine parameters.
    takeoff_time : pd.Timestamp
        Flight departure time.
    src_elapsed : npt.NDArray[FLOAT_DTYPE]
        Elapsed time at each active source. Shape ``(n_src,)``.
    climb_time : npt.NDArray[FLOAT_DTYPE]
        Time spent climbing for each edge and FL. Shape ``(n_edge, n_fl)``.
    climb_dist : npt.NDArray[FLOAT_DTYPE]
        Ground distance consumed by climb. Shape ``(n_edge, n_fl)``.
    descent_dd : npt.NDArray[FLOAT_DTYPE]
        Descent distance deducted from edge length. Shape ``(n_edge, n_fl)``.
    flat_dist : npt.NDArray[FLOAT_DTYPE]
        Total edge distance. Shape ``(n_edge,)``.
    src_alt_ft : npt.NDArray[FLOAT_DTYPE]
        Altitude at which each edge starts (the source FL, or the airport elevation
        for edges leaving the origin). Shape ``(n_edge,)``.

    Returns
    -------
    cruise_fuel : npt.NDArray[FLOAT_DTYPE]
        Shape ``(n_edge, n_fl, n_mach)``.
    cruise_time : npt.NDArray[FLOAT_DTYPE]
        Shape ``(n_edge, n_fl, n_mach)``.
    cruise_feasible : npt.NDArray[np.bool_]
        Shape ``(n_edge, n_fl, n_mach)``.
    cruise_eef : npt.NDArray[FLOAT_DTYPE]
        Shape ``(n_edge, n_fl)``. Integrated EEF in joules along the edge.
        Zero if ``eef_per_m`` is not in the met data.
    """
    sample_idxs, sample_to_edge, edge_bounds = _expand_edge_samples(met_lookup, flat_edge_idx)

    # Estimate arrival time at each sample point, then call met lookup
    sample_dt = _estimate_sample_times(
        met_lookup,
        sample_idxs,
        sample_to_edge,
        src_idx,
        takeoff_time,
        src_elapsed,
        climb_time,
        fl_choices,
        mach_choices,
    )
    sample_met = met_lookup(sample_idxs, sample_dt)

    air_temp_3d = sample_met.air_temperature[:, :, np.newaxis]
    mass_3d = post_climb_mass[sample_to_edge][:, :, np.newaxis]

    # Along-track wind component
    tailwind = -sample_met.headwind

    # Weight by cruise-zone overlap, then reduce to per-edge totals
    weight = _cruise_zone_weights(
        met_lookup,
        sample_idxs,
        sample_to_edge,
        climb_dist,
        descent_dd,
        flat_dist,
    )
    seg_dist = met_lookup.delta_dist[sample_idxs]

    # Cruise performance at each (sample, FL, Mach)
    with np.errstate(over="ignore", invalid="ignore"):
        # Ignore numpy errors at infeasible points
        ff, feas = ps.cruise_performance(
            fl_choices[np.newaxis, :, np.newaxis],
            mach_choices[np.newaxis, np.newaxis, :],
            mass_3d,
            air_temp_3d,
            atyp,
        )

    tas = units.mach_number_to_tas(mach_choices[np.newaxis, np.newaxis, :], air_temp_3d)
    ground_speed = tas + tailwind[:, :, np.newaxis]  # (n_sample, n_fl, n_mach)

    # Negative ground_speed could be handled gracefully with the feasible mask, but it's not
    # realistic and probably indicates an actual problem with the input
    if np.any(ground_speed < 0.0):
        raise RuntimeError("Negative ground speed: headwind exceeds TAS at some sample point")

    seg_time = seg_dist[:, np.newaxis, np.newaxis] / ground_speed * weight[:, :, np.newaxis]
    seg_fuel = ff * seg_time

    cruise_time = np.add.reduceat(seg_time, edge_bounds, axis=0)
    cruise_fuel = np.add.reduceat(seg_fuel, edge_bounds, axis=0)
    cruise_feasible = np.add.reduceat(~feas, edge_bounds, axis=0) == 0

    # Accumulate eef_per_m over the full edge (no cruise zone weighting)
    if sample_met.eef_per_m is not None:
        sample_fi = _climbing_fl_idxs(
            met_lookup.cum_dist[sample_idxs][:, np.newaxis],
            climb_dist[sample_to_edge],
            src_alt_ft[sample_to_edge][:, np.newaxis],
            fl_choices[np.newaxis, :],
            fl_choices,
        )
        eef_per_m = np.take_along_axis(sample_met.eef_per_m, sample_fi, axis=1)
        seg_eef = eef_per_m * seg_dist[:, np.newaxis]
        cruise_eef = np.add.reduceat(seg_eef, edge_bounds, axis=0)
    else:
        cruise_eef = np.zeros((1, 1), dtype=FLOAT_DTYPE)

    return cruise_fuel, cruise_time, cruise_feasible, cruise_eef


@dataclass(kw_only=True, slots=True, frozen=True)
class DAGState:
    """Working arrays for the wavefront dynamic program."""

    best_cost: npt.NDArray[FLOAT_DTYPE]  # (n_h, n_fl)
    best_mass: npt.NDArray[FLOAT_DTYPE]  # (n_h, n_fl) arrival mass at each state
    best_time: npt.NDArray[FLOAT_DTYPE]  # (n_h, n_fl) arrival time in seconds from takeoff
    best_mach: npt.NDArray[FLOAT_DTYPE]  # (n_h, n_fl) incoming cruise mach
    best_eef : npt.NDArray[FLOAT_DTYPE]  # (n_h, n_fl) cruise EEF on incoming edge
    best_climb_dist: npt.NDArray[FLOAT_DTYPE]  # (n_h, n_fl) climb distance on incoming edge
    best_climb_time: npt.NDArray[FLOAT_DTYPE]  # (n_h, n_fl) climb time on incoming edge
    best_prev_h: npt.NDArray[np.int64]  # (n_h, n_fl) previous horizontal node, -1 = no predecessor
    best_prev_fi: npt.NDArray[np.int64]  # (n_h, n_fl) previous fl index, -1 = came from origin

    @classmethod
    def initialize(cls, n_h: int, n_cols: int) -> Self:
        """Initialize arrays to default values for a DAG search."""
        return cls(
            best_cost=np.full((n_h, n_cols), np.inf, dtype=FLOAT_DTYPE),
            best_mass=np.full((n_h, n_cols), np.nan, dtype=FLOAT_DTYPE),
            best_time=np.full((n_h, n_cols), np.nan, dtype=FLOAT_DTYPE),
            best_mach=np.full((n_h, n_cols), np.nan, dtype=FLOAT_DTYPE),
            best_eef=np.full((n_h, n_cols), np.nan, dtype=FLOAT_DTYPE),
            best_climb_dist=np.full((n_h, n_cols), np.nan, dtype=FLOAT_DTYPE),
            best_climb_time=np.full((n_h, n_cols), np.nan, dtype=FLOAT_DTYPE),
            best_prev_h=np.full((n_h, n_cols), -1, dtype=np.int64),
            best_prev_fi=np.full((n_h, n_cols), -1, dtype=np.int64),
        )


@dataclass(kw_only=True, slots=True, frozen=True)
class DAGResult:
    """The output of :meth:`Optimizer.solve`."""

    state: DAGState
    amass_init: float
    trip_fuel: float
    payload: float
    reserve_fuel: float
    landing_mass: float


@dataclass(kw_only=True, slots=True, frozen=True)
class _SolverCtx:
    """Shared state passed to wavefront relaxation."""

    dag: HorizontalDAG
    fl_choices: npt.NDArray[FLOAT_DTYPE]
    mach_choices: npt.NDArray[FLOAT_DTYPE]
    atyp: ps_aircraft_params.PSAircraftEngineParams
    cost_index: float  # kg fuel / minute of flight time
    eef_cost_factor: float  # kg fuel / EEF Joule
    step_penalty_kg: float  # artificial kg fuel penalty for step climbs, on top of the maneuver
    allow_cooling_credit: bool
    origin_elev_ft: float
    dest_elev_ft: float
    takeoff_time: pd.Timestamp
    final_descent_dist: npt.NDArray[FLOAT_DTYPE]
    final_descent_fuel: npt.NDArray[FLOAT_DTYPE]
    final_descent_time: npt.NDArray[FLOAT_DTYPE]
    met_lookup: EdgeMetLookup | None


def _compute_ground_climbs(
    fl_choices: npt.NDArray[FLOAT_DTYPE],
    src_mass: float,
    atyp: ps_aircraft_params.PSAircraftEngineParams,
    origin_elev_ft: float,
    delta_isa: npt.NDArray[FLOAT_DTYPE],
    tailwind: npt.NDArray[FLOAT_DTYPE],
) -> tuple[
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[np.bool_],
]:
    """Compute climb from the ground to each candidate FL.

    Only used for the origin wavefront. The low-altitude phase (ground to
    ``fl_choices[0]``) always uses ISA temperature + zero wind.
    The upper phase (``fl_choices[0]`` to each FL) uses the
    provided ``delta_isa`` and ``tailwind``.

    Parameters
    ----------
    delta_isa : npt.NDArray[FLOAT_DTYPE]
        Temperature offset from ISA at base_alt. Shape ``(n_edge, 1)``.
    tailwind : npt.NDArray[FLOAT_DTYPE]
        Along-track tailwind component in m/s. Shape ``(n_edge, 1)``.

    Returns
    -------
    Five arrays each of shape ``(n_edge, n_fl)``.
    """
    base_alt = fl_choices[0]
    init_dist, init_fuel, init_time, base_mass = ps.climb_to_target(
        src_mass,
        origin_elev_ft,
        base_alt,
        atyp,
    )
    next_dist, next_fuel, next_time, post_climb_mass, feasible = ps.compute_climb_segment(
        base_alt,
        fl_choices,
        np.full_like(fl_choices, base_mass),
        atyp,
        delta_isa=delta_isa,
        tailwind=tailwind,
    )

    climb_dist = init_dist + next_dist
    climb_fuel = init_fuel + next_fuel
    climb_time = init_time + next_time

    return climb_dist, climb_fuel, climb_time, post_climb_mass, feasible


def _compute_edge_climbs(
    fl_idxs: npt.NDArray[np.int64],
    fl_choices: npt.NDArray[FLOAT_DTYPE],
    src_idx: npt.NDArray[np.int64],
    src_masses: npt.NDArray[FLOAT_DTYPE],
    src_elapsed: npt.NDArray[FLOAT_DTYPE],
    flat_edge_idx: npt.NDArray[np.int64],
    atyp: ps_aircraft_params.PSAircraftEngineParams,
    takeoff_time: pd.Timestamp,
    met_lookup: EdgeMetLookup | None,
) -> tuple[
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[np.bool_],
]:
    """Compute start-of-edge FL-to-FL step climbs with met-based corrections.

    The delta between met and ISA temperature at the source node is computed
    once and added as a constant offset at every altitude step during the climb.
    Tailwind is computed from wind at the source node in the direction of the edge
    azimuth and converts true air distance to ground distance inside
    ``compute_climb_segment``.

    Parameters
    ----------
    fl_idxs : npt.NDArray[np.int64]
        Source FL index for each active source. Shape ``(n_src,)``.
    fl_choices : npt.NDArray[FLOAT_DTYPE]
        Candidate destination FLs. Shape ``(n_fl,)``.
    src_idx : npt.NDArray[np.int64]
        Maps each edge to its source in the active-source arrays. Shape ``(n_edge,)``.
    src_masses : npt.NDArray[FLOAT_DTYPE]
        Mass at each active source. Shape ``(n_src,)``.
    src_elapsed : npt.NDArray[FLOAT_DTYPE]
        Elapsed time at each active source. Shape ``(n_src,)``.
    flat_edge_idx : npt.NDArray[np.int64]
        Global edge index for each edge. Shape ``(n_edge,)``.
    atyp : ps_aircraft_params.PSAircraftEngineParams
        Aircraft/engine parameters.
    takeoff_time : pd.Timestamp
        Flight departure time.
    met_lookup : EdgeMetLookup or None
        Met interpolator. If None, ISA + zero wind is used.

    Returns
    -------
    climb_dist : npt.NDArray[FLOAT_DTYPE]
        Ground distance consumed by climb. Shape ``(n_edge, n_fl)``.
    climb_fuel : npt.NDArray[FLOAT_DTYPE]
        Fuel burned during climb. Shape ``(n_edge, n_fl)``.
    climb_time : npt.NDArray[FLOAT_DTYPE]
        Time spent climbing. Shape ``(n_edge, n_fl)``.
    post_climb_mass : npt.NDArray[FLOAT_DTYPE]
        Aircraft mass after climb. Shape ``(n_edge, n_fl)``.
    feasible : npt.NDArray[np.bool_]
        True where the climb is feasible or dst <= src (step-down). Shape ``(n_edge, n_fl)``.
    """
    n_fl = len(fl_choices)
    edge_src_fi = fl_idxs[src_idx]
    edge_src_fl = fl_choices[edge_src_fi]
    _neighborhood_edges,
    edge_src_mass = src_masses[src_idx]

    if met_lookup is not None:
        edge_start = met_lookup.edge_ptr[flat_edge_idx]
        src_time_s = src_elapsed[src_idx]
        src_dt = np.datetime64(takeoff_time) + (src_time_s * 1e9).astype("timedelta64[ns]")
        climb_dt = np.broadcast_to(src_dt[:, np.newaxis], (len(src_idx), n_fl))
        climb_met = met_lookup(edge_start, climb_dt)

        edge_arange = np.arange(len(src_idx))
        met_T = climb_met.air_temperature[edge_arange, edge_src_fi]
        isa_T = units.m_to_T_isa(units.ft_to_m(edge_src_fl))
        delta_isa = (met_T - isa_T)[:, np.newaxis]

        tailwind = -climb_met.headwind
    else:
        delta_isa = 0.0
        tailwind = 0.0

    climb_dist, climb_fuel, climb_time, post_climb_mass, feasible = ps.compute_climb_segment(
        edge_src_fl[:, np.newaxis],
        fl_choices[np.newaxis, :],
        edge_src_mass[:, np.newaxis],
        atyp,
        delta_isa=delta_isa,
        tailwind=tailwind,
    )

    # Step-downs (dst <= src) bypass the climb model and are always feasible
    feasible = feasible | (fl_choices[np.newaxis, :] <= edge_src_fl[:, np.newaxis])
    return climb_dist, climb_fuel, climb_time, post_climb_mass, feasible


def _isa_cruise(
    fl_choices: npt.NDArray[FLOAT_DTYPE],
    mach_choices: npt.NDArray[FLOAT_DTYPE],
    edge_mass: npt.NDArray[FLOAT_DTYPE],
    atyp: ps_aircraft_params.PSAircraftEngineParams,
    cruise_dist: npt.NDArray[FLOAT_DTYPE],
) -> tuple[npt.NDArray[FLOAT_DTYPE], npt.NDArray[FLOAT_DTYPE], npt.NDArray[np.bool_]]:
    """Compute cruise fuel, time, and feasibility using ISA temperatures (no met)."""
    fl_3d = fl_choices[np.newaxis, :, np.newaxis]
    air_temperature_3d = units.m_to_T_isa(units.ft_to_m(fl_3d))
    mass_3d = edge_mass[:, :, np.newaxis]

    mach_3d = mach_choices[np.newaxis, np.newaxis, :]
    with np.errstate(over="ignore", invalid="ignore"):
        # Ignore numpy errors at infeasible points
        ff, cruise_feasible = ps.cruise_performance(
            fl_3d,
            mach_3d,
            mass_3d,
            air_temperature_3d,
            atyp,
        )

    tas = units.mach_number_to_tas(mach_3d, air_temperature_3d)
    cruise_time = cruise_dist[:, :, np.newaxis] / tas
    cruise_fuel = ff * cruise_time
    return cruise_fuel, cruise_time, cruise_feasible


def _relax_wavefront(wave: npt.NDArray[np.int64], ctx: _SolverCtx, state: DAGState) -> None:
    """Relax all edges leaving one wavefront of source nodes."""
    fl_choices = ctx.fl_choices
    mach_choices = ctx.mach_choices
    n_fl = len(fl_choices)

    wave_costs = state.best_cost[wave]
    active_mask = np.isfinite(wave_costs)
    if not np.any(active_mask):
        return

    w_idxs, fl_idxs = np.nonzero(active_mask)
    h_idxs = wave[w_idxs]

    src_costs = state.best_cost[h_idxs, fl_idxs]
    src_masses = state.best_mass[h_idxs, fl_idxs]
    src_elapsed = state.best_time[h_idxs, fl_idxs]

    # Expand CSR adjacency for active sources into flat edge arrays
    flat_nbr, flat_dist, src_idx, flat_edge_idx = ctx.dag.expand_neighbors(h_idxs)

    # Climb from src_fl to each dst_fl: (n_edge, n_fl)
    ground_fi = len(fl_choices)
    if (fl_idxs == ground_fi).any():
        # Origin wavefront: climb from ground, with met-based corrections if available
        n_edge = len(src_idx)

        if ctx.met_lookup is not None:
            edge_start = ctx.met_lookup.edge_ptr[flat_edge_idx]
            src_time = np.broadcast_to(np.datetime64(ctx.takeoff_time), (n_edge, len(fl_choices)))
            climb_met = ctx.met_lookup(edge_start, src_time)

            # Use lowest FL index for temperature/wind (closest to base_alt)
            met_T = climb_met.air_temperature[:, 0]
            base_alt = fl_choices[0]
            isa_T = units.m_to_T_isa(units.ft_to_m(base_alt))
            delta_isa = (met_T - isa_T)[:, np.newaxis]

            tailwind = -climb_met.headwind
        else:
            delta_isa = np.zeros((n_edge, 1), dtype=FLOAT_DTYPE)
            tailwind = np.zeros((n_edge, 1), dtype=FLOAT_DTYPE)

        climb_dist, climb_fuel, climb_time, post_climb_mass, feasible = _compute_ground_climbs(
            fl_choices,
            src_masses[0],
            ctx.atyp,
            ctx.origin_elev_ft,
            delta_isa,
            tailwind,
        )
    else:
        climb_dist, climb_fuel, climb_time, post_climb_mass, feasible = _compute_edge_climbs(
            fl_idxs,
            fl_choices,
            src_idx,
            src_masses,
            src_elapsed,
            flat_edge_idx,
            ctx.atyp,
            ctx.takeoff_time,
            ctx.met_lookup,
        )

    climb_cost = ctx.cost_index / 60.0 * climb_time + climb_fuel

    # The ground slot sits one past the FLs, so append the airport elevation to index it
    alt_by_fi = np.append(fl_choices, ctx.origin_elev_ft).astype(FLOAT_DTYPE)
    src_alt_ft = alt_by_fi[fl_idxs[src_idx]]

    # Descent: ground descent if neighbor is destination.
    # FL-to-FL step-downs are treated as part of cruise (mild thrust reduction at cruise Mach,
    # negligible extra distance or time vs. level flight).
    is_dest = (flat_nbr == ctx.dag.h_dest)[:, np.newaxis]
    descent_dd = is_dest * ctx.final_descent_dist
    descent_dt = is_dest * ctx.final_descent_time
    descent_df = is_dest * ctx.final_descent_fuel

    # Cruise fuel/time: (n_edge, n_fl, n_mach)
    cruise_dist = flat_dist[:, np.newaxis] - climb_dist - descent_dd
    if ctx.met_lookup is not None:
        cruise_fuel, cruise_time, cruise_feasible, cruise_eef = _calculate_cruise_at_samples(
            ctx.met_lookup,
            flat_edge_idx,
            src_idx,
            fl_choices,
            mach_choices,
            post_climb_mass,
            ctx.atyp,
            ctx.takeoff_time,
            src_elapsed,
            climb_time,
            climb_dist,
            descent_dd,
            flat_dist,
            src_alt_ft,
        )
    else:
        cruise_fuel, cruise_time, cruise_feasible = _isa_cruise(
            fl_choices,
            mach_choices,
            post_climb_mass,
            ctx.atyp,
            cruise_dist,
        )
        cruise_eef = np.zeros((1, 1), dtype=FLOAT_DTYPE)

    valid = feasible[:, :, np.newaxis] & cruise_feasible & (cruise_dist[:, :, np.newaxis] > 0.0)
    cruise_cost = ctx.cost_index / 60.0 * cruise_time + cruise_fuel
    descent_cost = ctx.cost_index / 60.0 * descent_dt + descent_df

    # CO2 climate cost of fuel burn (kg-fuel-equivalent)
    fuel_co2_cost = (
        JetA.ei_co2
        * _J_PER_KG_CO2
        * ctx.eef_cost_factor
        * (climb_fuel[:, :, np.newaxis] + cruise_fuel + descent_df[:, :, np.newaxis])
    )
    # CO2e climate cost of contrail EEF (kg-fuel-equivalent)
    contrail_co2e_cost = (
        ctx.eef_cost_factor
        * (cruise_eef if ctx.allow_cooling_credit else np.maximum(cruise_eef, 0.0))
    )[:, :, np.newaxis]

    # Add an artificial penalty for any step climb / descent.
    # Exclude both initial climb and final descent from the penalty.
    edge_src_fi = fl_idxs[src_idx][:, np.newaxis]
    is_step = (np.arange(n_fl)[np.newaxis, :] != edge_src_fi) & (edge_src_fi < n_fl)
    step_penalty = np.where(is_step & ~is_dest, ctx.step_penalty_kg, 0.0).astype(FLOAT_DTYPE)

    # The main cost function
    total_cost = (
        src_costs[src_idx, np.newaxis, np.newaxis]
        + climb_cost[:, :, np.newaxis]
        + step_penalty[:, :, np.newaxis]
        + descent_cost[:, :, np.newaxis]
        + cruise_cost
        + contrail_co2e_cost
        + fuel_co2_cost
    )
    total_cost = np.where(valid, total_cost, np.inf)

    # Best mach per (edge, FL)
    best_mach_idx = np.argmin(total_cost, axis=2, keepdims=True)
    best_total = np.take_along_axis(total_cost, best_mach_idx, axis=2).squeeze(2)
    best_mach = mach_choices[best_mach_idx.squeeze(2)]
    best_cruise_fuel = np.take_along_axis(cruise_fuel, best_mach_idx, axis=2).squeeze(2)
    best_cruise_time = np.take_along_axis(cruise_time, best_mach_idx, axis=2).squeeze(2)
    arrival_mass = post_climb_mass - best_cruise_fuel - descent_df
    arrival_time = src_elapsed[src_idx, None] + climb_time + best_cruise_time + descent_dt

    # Scatter-min into real FL slots only (not the ground slot)
    # We'd really want to call something like np.argminimum.at to get the winners first,
    # but it doesn't exist. Note that np.minimum.at(a, indices, b) is equivalent to:
    # for i, idx in enumerate(indices):
    #     a[idx] = min(a[idx], b[i])
    np.minimum.at(state.best_cost[:, :n_fl], flat_nbr, best_total)

    # Determining the winners can fail if the dtypes are different, so idiot check first
    if state.best_cost.dtype != best_total.dtype:
        raise RuntimeError(
            f"Dtype mismatch: state.best_cost is {state.best_cost.dtype}, "
            f"but best_total is {best_total.dtype}"
        )
    winners = np.isfinite(best_total) & (best_total == state.best_cost[flat_nbr, :n_fl])

    wi, wj = np.nonzero(winners)  # theoretically ties could happen, so the last gets chosen
    state.best_mass[flat_nbr[wi], wj] = arrival_mass[wi, wj]
    state.best_time[flat_nbr[wi], wj] = arrival_time[wi, wj]
    state.best_mach[flat_nbr[wi], wj] = best_mach[wi, wj]
    state.best_eef[flat_nbr[wi], wj] = cruise_eef[wi, wj]
    state.best_climb_dist[flat_nbr[wi], wj] = climb_dist[wi, wj]
    state.best_climb_time[flat_nbr[wi], wj] = climb_time[wi, wj]
    state.best_prev_h[flat_nbr[wi], wj] = h_idxs[src_idx[wi]]
    state.best_prev_fi[flat_nbr[wi], wj] = fl_idxs[src_idx[wi]]


def solve_dag(
    dag: HorizontalDAG,
    amass_init: float,
    fl_choices: npt.NDArray[FLOAT_DTYPE],
    mach_choices: npt.NDArray[FLOAT_DTYPE],
    atyp: ps_aircraft_params.PSAircraftEngineParams,
    cost_index: float,
    eef_cost_factor: float,
    origin_elev_ft: float,
    dest_elev_ft: float,
    takeoff_time: pd.Timestamp,
    met_lookup: EdgeMetLookup | None,
    allow_cooling_credit: bool,
    step_penalty_kg: float,
    on_wavefront: Callable[[npt.NDArray[np.int64], DAGState], None] | None = None,
) -> DAGState:
    """Solve shortest-path DP on the :class:`HorizontalDAG`, tracking mass exactly."""
    fl_choices = fl_choices.astype(FLOAT_DTYPE, copy=False)
    mach_choices = mach_choices.astype(FLOAT_DTYPE, copy=False)

    # Compute descent dist/fuel/time for each FL once up front.
    fd_dist, fd_fuel, fd_time = ps.final_descent(fl_choices, dest_elev_ft, atyp)
    ctx = _SolverCtx(
        dag=dag,
        fl_choices=fl_choices,
        mach_choices=mach_choices,
        atyp=atyp,
        cost_index=cost_index,
        eef_cost_factor=eef_cost_factor,
        step_penalty_kg=step_penalty_kg,
        allow_cooling_credit=allow_cooling_credit,
        origin_elev_ft=origin_elev_ft,
        dest_elev_ft=dest_elev_ft,
        takeoff_time=takeoff_time,
        final_descent_dist=fd_dist,
        final_descent_fuel=fd_fuel,
        final_descent_time=fd_time,
        met_lookup=met_lookup,
    )

    n_h = dag.n_nodes  # number of horizontal nodes
    n_fl = len(fl_choices)  # number of cruise FL choices
    ground_fi = n_fl  # extra column for origin/destination ground states
    n_cols = n_fl + 1
    state = DAGState.initialize(n_h, n_cols)

    # Seed origin at the ground slot
    state.best_cost[dag.h_origin, ground_fi] = 0.0
    state.best_mass[dag.h_origin, ground_fi] = amass_init
    state.best_time[dag.h_origin, ground_fi] = 0.0

    for wave in dag.topo_wavefronts():
        _relax_wavefront(wave, ctx, state)
        if on_wavefront is not None:
            on_wavefront(wave, state)

    # Populate destination ground slot from best FL
    h_dest = dag.h_dest
    best_dest_fi = np.argmin(state.best_cost[h_dest, :n_fl]).item()
    if np.isfinite(state.best_cost[h_dest, best_dest_fi]):
        state.best_cost[h_dest, ground_fi] = state.best_cost[h_dest, best_dest_fi]
        state.best_mass[h_dest, ground_fi] = state.best_mass[h_dest, best_dest_fi]
        state.best_time[h_dest, ground_fi] = state.best_time[h_dest, best_dest_fi]
        state.best_mach[h_dest, ground_fi] = state.best_mach[h_dest, best_dest_fi]
        state.best_climb_dist[h_dest, ground_fi] = state.best_climb_dist[h_dest, best_dest_fi]
        state.best_climb_time[h_dest, ground_fi] = state.best_climb_time[h_dest, best_dest_fi]
        state.best_prev_h[h_dest, ground_fi] = state.best_prev_h[h_dest, best_dest_fi]
        state.best_prev_fi[h_dest, ground_fi] = state.best_prev_fi[h_dest, best_dest_fi]

    return state


@dataclass(kw_only=True, slots=True, frozen=True)
class _ProfileArrays:
    """Numpy arrays backing the track solver: met per (waypoint, FL) plus track geometry."""

    air_temperature: npt.NDArray[FLOAT_DTYPE]  # (n_wp, n_fl)
    u_wind: npt.NDArray[FLOAT_DTYPE]  # (n_wp, n_fl)
    v_wind: npt.NDArray[FLOAT_DTYPE]  # (n_wp, n_fl)
    eef_per_m: npt.NDArray[FLOAT_DTYPE] | None  # (n_wp, n_fl)
    cum_dist: npt.NDArray[FLOAT_DTYPE]  # (n_wp,) along-track distance at each waypoint
    seg_dist: npt.NDArray[FLOAT_DTYPE]  # (n_wp - 1,) waypoint-to-waypoint distance
    seg_azimuth: npt.NDArray[FLOAT_DTYPE]  # (n_wp - 1,) radians

    @classmethod
    def build(cls, dag: HorizontalDAG | Track, profile: xr.Dataset) -> Self:
        lon, lat = dag.lon, dag.lat
        return cls(
            air_temperature=profile["air_temperature"].values,
            u_wind=profile["u_wind"].values,
            v_wind=profile["v_wind"].values,
            eef_per_m=profile["eef_per_m"].values if "eef_per_m" in profile else None,
            cum_dist=dag.cum_dist,
            seg_dist=dag.segment_dist,
            seg_azimuth=np.deg2rad(geo.azimuth(lon[:-1], lat[:-1], lon[1:], lat[1:])),
        )


def _expand_transitions(
    start: npt.NDArray[np.int64],
    end: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Flatten each transition into the run of waypoints it covers, from ``start`` to ``end``.

    Returns ``(node, transition, bounds)``: the waypoint index of each sample, the index
    of the transition it belongs to, and per-transition offsets for ``np.add.reduceat``.
    """
    n_samp = (end - start) + 1
    bounds = np.zeros(len(start) + 1, dtype=np.int64)
    np.cumsum(n_samp, out=bounds[1:])
    transition = np.repeat(np.arange(len(start), dtype=np.int64), n_samp)
    offset = np.arange(bounds[-1], dtype=np.int64) - np.repeat(bounds[:-1], n_samp)
    node = np.repeat(start, n_samp) + offset
    return node, transition, bounds[:-1]


def _track_cruise(
    pa: _ProfileArrays,
    start: npt.NDArray[np.int64],
    arrival: npt.NDArray[np.int64],
    fl_idx: npt.NDArray[np.int64],
    fl_choices: npt.NDArray[FLOAT_DTYPE],
    mach_choices: npt.NDArray[FLOAT_DTYPE],
    mass: npt.NDArray[FLOAT_DTYPE],
    lead_dist: npt.NDArray[FLOAT_DTYPE],
    trail_dist: npt.NDArray[FLOAT_DTYPE],
    src_alt_ft: npt.NDArray[FLOAT_DTYPE],
    atyp: ps_aircraft_params.PSAircraftEngineParams,
) -> tuple[
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[np.bool_],
]:
    """Integrate level cruise for a batch of transitions at every candidate Mach."""
    node, tr, bounds = _expand_transitions(start, arrival)
    fli = fl_idx[tr]

    # Each sample integrates the segment leaving its node
    # The arrival node has no next segment, so length 0
    safe = np.minimum(node, len(pa.seg_dist) - 1)
    is_last = node == arrival[tr]
    delta = np.where(is_last, 0.0, pa.seg_dist[safe])
    azimuth = pa.seg_azimuth[safe].copy()
    prev = np.maximum(np.arange(len(node), dtype=np.int64) - 1, 0)
    azimuth[is_last] = azimuth[prev][is_last]

    temperature = pa.air_temperature[node, fli]
    tailwind = pa.u_wind[node, fli] * np.sin(azimuth) + pa.v_wind[node, fli] * np.cos(azimuth)

    # Fraction of each sample's segment lying in the cruise zone
    span = pa.cum_dist[arrival] - pa.cum_dist[start]
    x_lo = pa.cum_dist[node] - pa.cum_dist[start[tr]]
    overlap = np.clip(
        np.minimum(x_lo + delta, (span - trail_dist)[tr]) - np.maximum(x_lo, lead_dist[tr]),
        0.0,
        None,
    )
    weight = np.divide(overlap, delta, out=np.zeros_like(delta), where=delta > 0.0)

    # Cruise performance at each (sample, Mach)
    tas = units.mach_number_to_tas(mach_choices[np.newaxis, :], temperature[:, np.newaxis])
    ground_speed = tas + tailwind[:, np.newaxis]
    with np.errstate(over="ignore", invalid="ignore"):
        ff, feas = ps.cruise_performance(
            fl_choices[fli][:, np.newaxis],
            mach_choices[np.newaxis, :],
            mass[tr][:, np.newaxis],
            temperature[:, np.newaxis],
            atyp,
        )
    feas = feas & (ground_speed > 0.0)

    seg_time = np.divide(
        (delta * weight)[:, np.newaxis],
        ground_speed,
        out=np.zeros_like(ground_speed),
        where=ground_speed > 0.0,
    )
    cruise_time = np.add.reduceat(seg_time, bounds, axis=0)
    cruise_fuel = np.add.reduceat(ff * seg_time, bounds, axis=0)
    feasible = np.add.reduceat(~feas, bounds, axis=0) == 0

    # The climb must fit within the available span, or the transition can't complete (1 m buffer)
    feasible &= (lead_dist + trail_dist <= span + 1.0)[:, np.newaxis]

    if pa.eef_per_m is None:
        eef = np.zeros(len(start), dtype=FLOAT_DTYPE)
    else:
        sample_fli = _climbing_fl_idxs(
            x_lo, lead_dist[tr], src_alt_ft[tr], fl_choices[fli], fl_choices
        )
        eef = np.add.reduceat(pa.eef_per_m[node, sample_fli] * delta, bounds)

    return cruise_fuel, cruise_time, eef, feasible


def _relax_transitions(
    pa: _ProfileArrays,
    state: DAGState,
    h: int,
    arrival: npt.NDArray[np.int64],
    fl_choices: npt.NDArray[FLOAT_DTYPE],
    mach_choices: npt.NDArray[FLOAT_DTYPE],
    target_fi: npt.NDArray[np.int64],
    arrival_fi: npt.NDArray[np.int64],
    src_fi: npt.NDArray[np.int64],
    src_cost: npt.NDArray[FLOAT_DTYPE],
    src_elapsed: npt.NDArray[FLOAT_DTYPE],
    lead_dist: npt.NDArray[FLOAT_DTYPE],
    trail_dist: npt.NDArray[FLOAT_DTYPE],
    fixed_fuel: npt.NDArray[FLOAT_DTYPE],
    fixed_time: npt.NDArray[FLOAT_DTYPE],
    cruise_mass: npt.NDArray[FLOAT_DTYPE],
    post_mass: npt.NDArray[FLOAT_DTYPE],
    feasible: npt.NDArray[np.bool_],
    atyp: ps_aircraft_params.PSAircraftEngineParams,
    cost_index: float,
    eef_cost_factor: float,
    allow_cooling_credit: bool,
) -> None:
    """Cruise a batch of transitions at every candidate Mach number, pick the best, and relax.

    Each transition carries a maneuver with known fuel and time (``fixed_fuel`` and ``fixed_time``:
    a step climb or step descent at its start over ``lead_dist``, or a final descent at its
    end over ``trail_dist``) plus a level cruise over the rest, whose fuel and time depend on
    the Mach number choice.
    """
    n = len(target_fi)

    # The ground slot sits one past the FLs
    src_alt_ft = np.append(fl_choices, fl_choices[0])[src_fi]

    cruise_fuel, cruise_time, eef, feasible_m = _track_cruise(
        pa,
        np.full(n, h, dtype=np.int64),
        arrival,
        target_fi,
        fl_choices,
        mach_choices,
        cruise_mass,
        lead_dist,
        trail_dist,
        src_alt_ft,
        atyp,
    )

    mach_cost = cost_index / 60.0 * cruise_time + cruise_fuel
    mach_cost = np.where(feasible_m, mach_cost, np.inf)
    best_m = np.argmin(mach_cost, axis=1)

    def _take(a: npt.NDArray) -> npt.NDArray:
        return np.take_along_axis(a, best_m[:, np.newaxis], axis=1)[:, 0]

    best_cruise_fuel = _take(cruise_fuel)
    best_cruise_time = _take(cruise_time)

    _relax_track_batch(
        state,
        arrival,
        arrival_fi,
        src_cost,
        h,
        src_fi,
        fixed_fuel + best_cruise_fuel,
        fixed_time + best_cruise_time,
        src_elapsed,
        post_mass - best_cruise_fuel,
        eef,
        feasible & np.isfinite(_take(mach_cost)),
        mach_choices[best_m],
        lead_dist,
        fixed_time,
        cost_index,
        eef_cost_factor,
        allow_cooling_credit,
    )


def _arrival_node(
    cum: npt.NDArray[FLOAT_DTYPE],
    src: int,
    dist: npt.NDArray[FLOAT_DTYPE],
    n_h: int,
) -> npt.NDArray[np.int64]:
    """Return the first waypoint at or beyond ``dist`` along the track from ``src``."""
    return np.clip(np.searchsorted(cum, cum[src] + dist, side="left"), src + 1, n_h - 1)


def _step_change(
    pa: _ProfileArrays,
    src_h: int,
    src_fl: npt.NDArray[FLOAT_DTYPE],
    tgt_fl: npt.NDArray[FLOAT_DTYPE],
    src_mass: npt.NDArray[FLOAT_DTYPE],
    fl_choices: npt.NDArray[FLOAT_DTYPE],
    atyp: ps_aircraft_params.PSAircraftEngineParams,
) -> tuple[
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[np.int64],
    npt.NDArray[np.bool_],
]:
    """Make a step climb or a step descent waypoint by waypoint, reading the weather as it flies.

    A maneuver at cruise can take minutes and cover a hundred kilometers, and over that stretch
    the heading and the wind both change. For this reason, it's important to use the entire
    step geometry instead of just the geometry at the source node.

    This function steps along the waypoints. At each one, it reads the temperature and wind at the
    flight level nearest the current altitude, works out how long the segment takes at the
    resulting ground speed, and gains or loses whatever altitude comes out. It stops on reaching the
    target level.

    A climb takes its rate from the thrust available; a descent follows a fixed 3 degree path.

    Return a tuple of:
    - ground distance covered in m
    - fuel burned in kg
    - time taken in s
    - mass on levelling off in kg
    - the first waypoint at or after levelling off
    - a boolean feasibility mask
    """
    n_h = len(pa.cum_dist)

    alt = np.asarray(src_fl, dtype=FLOAT_DTYPE).copy()
    mass = np.asarray(src_mass, dtype=FLOAT_DTYPE).copy()
    dist = np.zeros_like(alt)
    fuel = np.zeros_like(alt)
    time = np.zeros_like(alt)
    arrival = np.full(len(alt), min(src_h + 1, n_h - 1), dtype=np.int64)
    feasible = np.ones(len(alt), dtype=bool)

    climbing = tgt_fl > src_fl
    moving = climbing | (tgt_fl < src_fl)

    for j in range(src_h, n_h - 1):
        active = moving & feasible & (np.abs(tgt_fl - alt) > 1.0)
        if not active.any():
            break

        # Weather at this waypoint, at the flight level nearest where the aircraft now is
        fi = np.argmin(np.abs(fl_choices[np.newaxis, :] - alt[:, np.newaxis]), axis=1)
        air_temperature = pa.air_temperature[j, fi]
        az = pa.seg_azimuth[j].item()
        tailwind = pa.u_wind[j, fi] * np.sin(az) + pa.v_wind[j, fi] * np.cos(az)

        rocd = np.zeros_like(alt)
        tas = np.ones_like(alt)
        ff = np.zeros_like(alt)

        up = active & climbing
        if up.any():
            with np.errstate(over="ignore", invalid="ignore"):
                ff[up], rocd[up], tas[up], can_climb = ps.climb_performance(
                    alt[up], mass[up], air_temperature[up], atyp
                )
            feasible[up] &= can_climb

        down = active & ~climbing
        if down.any():
            with np.errstate(over="ignore", invalid="ignore"):
                ff[down], rocd[down], tas[down] = ps.descent_performance(
                    alt[down], mass[down], air_temperature[down], atyp
                )

        # Time to fly this segment and the altitude gained
        ground_speed = np.maximum(tas + tailwind, 1.0)  # clip at 1 to avoid div by 0 (unlikely)
        seg_time = pa.seg_dist[j] / ground_speed
        gain = rocd / 60.0 * seg_time
        remaining = tgt_fl - alt

        # A maneuver that finishes inside this segment only takes the partial time it needs
        done = np.abs(gain) >= np.abs(remaining)
        part = np.divide(
            np.abs(remaining) * 60.0,
            np.abs(rocd),
            out=np.zeros_like(alt),
            where=rocd != 0.0,
        )
        step_time = np.where(done, part, seg_time)

        step = active & feasible
        dist[step] += (ground_speed * step_time)[step]
        fuel[step] += (ff * step_time)[step]
        time[step] += step_time[step]
        mass[step] -= (ff * step_time)[step]
        alt[step] = np.where(done, tgt_fl, alt + gain)[step]
        arrival[step] = j + 1

    # Anything still moving ran out of track before it levelled off, so mark it infeasible
    feasible &= ~(moving & (np.abs(tgt_fl - alt) > 1.0))
    return dist, fuel, time, mass, arrival, feasible


def _determine_final_descent(
    pa: _ProfileArrays,
    h_dest: int,
    dest_elev_ft: float,
    fl_choices: npt.NDArray[FLOAT_DTYPE],
    atyp: ps_aircraft_params.PSAircraftEngineParams,
) -> tuple[
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[np.int64],
]:
    """Reverse the final descent to the destination to determine where it begins.

    The final descent is the longest maneuver of the flight (around 200 km and 15-20 minutes),
    so it needs the weather along its length. It is computed backwards because the solver needs
    to know where the descent starts, given that it has to finish at the runway. One walk up
    from the destination crosses every candidate flight level in turn, so each level's top of
    descent, distance, fuel and time all come out of the same calculation.

    Mass is taken as the maximum landing weight, as :func:`ps.final_descent` does. Distance and
    time do not depend on it, and the fuel depends only weakly.

    Return a tuple of, one entry per candidate flight level:
    - descent ground distance in m
    - descent fuel in kg
    - descent time in s
    - the waypoint at or before which the descent must begin, or -1 if the track is too short
    """
    n_fl = len(fl_choices)
    mass = np.array([atyp.amass_mlw], dtype=FLOAT_DTYPE)

    dist = np.zeros(n_fl, dtype=FLOAT_DTYPE)
    fuel = np.zeros(n_fl, dtype=FLOAT_DTYPE)
    time = np.zeros(n_fl, dtype=FLOAT_DTYPE)
    tod_node = np.full(n_fl, -1, dtype=np.int64)

    alt = dest_elev_ft

    # Altitude only increases below, so a level at or below the starting altitude never gets
    # crossed and would keep the loop from ever finishing. That happens when the flown descent
    # is reused and dest_elev_ft is the lowest candidate FL. Such a level needs no descent
    tod_node[fl_choices <= alt] = h_dest - 1

    run_dist = run_fuel = run_time = 0.0

    for j in range(h_dest - 1, -1, -1):
        if (tod_node >= 0).all():
            break

        # Weather over the segment the aircraft flies from j to j+1, at the level nearest
        # the altitude it is passing through there
        fi = np.argmin(np.abs(fl_choices - alt)).item()
        air_temperature = pa.air_temperature[j, fi : fi + 1]
        az = pa.seg_azimuth[j].item()
        tailwind = pa.u_wind[j, fi] * np.sin(az) + pa.v_wind[j, fi] * np.cos(az)

        ff, rocd, tas = ps.descent_performance(
            np.array([alt], dtype=FLOAT_DTYPE), mass, air_temperature, atyp
        )
        ground_speed = max((tas[0] + tailwind).item(), 1.0)
        seg_time = pa.seg_dist[j] / ground_speed
        rise = abs(rocd[0].item()) / 60.0 * seg_time  # altitude gained walking backwards

        # Record every candidate level this segment passes through, at the fraction of the
        # segment where the crossing happens rather than at the waypoint
        crossed = (tod_node < 0) & (fl_choices > alt) & (fl_choices <= alt + rise)
        if crossed.any() and rise > 0.0:
            frac = (fl_choices[crossed] - alt) / rise
            dist[crossed] = run_dist + frac * pa.seg_dist[j]
            time[crossed] = run_time + frac * seg_time
            fuel[crossed] = run_fuel + frac * ff[0] * seg_time
            tod_node[crossed] = j

        run_dist += pa.seg_dist[j]
        run_time += seg_time
        run_fuel += ff[0] * seg_time
        alt += rise

    return dist, fuel, time, tod_node


def solve_track(
    dag: HorizontalDAG | Track,
    profile: xr.Dataset,
    amass_init: float,
    fl_choices: npt.NDArray[FLOAT_DTYPE],
    mach_choices: npt.NDArray[FLOAT_DTYPE],
    atyp: ps_aircraft_params.PSAircraftEngineParams,
    cost_index: float,
    eef_cost_factor: float,
    origin_elev_ft: float,
    dest_elev_ft: float,
    allow_cooling_credit: bool,
    step_penalty_kg: float,
) -> DAGState:
    """Solve the vertical profile along a fixed track, choosing the flight level and Mach number.

    Unlike :func:`solve_dag`, there is no precomputed edge set. From a given state, each action
    determines how far the aircraft advances: a level cruise (target FL equal to the current
    one) moves one waypoint; a step climb or step descent moves as far as the maneuver model's
    own distance requires, then cruises the remainder of that segment to the next waypoint.
    """
    fl_choices = fl_choices.astype(FLOAT_DTYPE, copy=False)
    mach_choices = mach_choices.astype(FLOAT_DTYPE, copy=False)
    pa = _ProfileArrays.build(dag, profile)

    n_h = dag.n_nodes
    n_fl = len(fl_choices)
    ground_fi = n_fl
    state = DAGState.initialize(n_h, n_fl + 1)

    cum = pa.cum_dist
    h_dest = dag.h_dest

    fd_dist, fd_fuel, fd_time, tod_node = _determine_final_descent(
        pa, h_dest, dest_elev_ft, fl_choices, atyp
    )

    state.best_cost[dag.h_origin, ground_fi] = 0.0
    state.best_mass[dag.h_origin, ground_fi] = amass_init
    state.best_time[dag.h_origin, ground_fi] = 0.0

    all_fl = np.arange(n_fl, dtype=np.int64)

    for h in range(h_dest):
        active = np.flatnonzero(np.isfinite(state.best_cost[h]))
        if not active.size:
            continue

        if h == dag.h_origin:
            # Nothing can transition backward into the ground slot, so it is always the
            # only active state at the origin
            src_mass = state.best_mass[h, ground_fi].item()

            # Below the lowest candidate FL the climb is flown at ISA with no wind, so it is the
            # same for every target and its distance is settled before any weather is read. Fly
            # that first, then step climb from the waypoint it reaches, at the lowest FL.
            base_dist, base_fuel, base_time, base_mass = ps.climb_to_target(
                src_mass, origin_elev_ft, fl_choices[0], atyp
            )
            base_h = _arrival_node(cum, h, np.array([base_dist]), n_h).item()

            up_dist, up_fuel, up_time, post_mass, arrival, feasible = _step_change(
                pa,
                base_h,
                np.full(n_fl, fl_choices[0], dtype=FLOAT_DTYPE),
                fl_choices,
                np.full(n_fl, base_mass, dtype=FLOAT_DTYPE),
                fl_choices,
                atyp,
            )
            climb_dist = base_dist + up_dist
            climb_fuel = base_fuel + up_fuel
            climb_time = base_time + up_time
            feasible = feasible | (fl_choices == fl_choices[0])
            feasible &= arrival != h_dest
            _relax_transitions(
                pa,
                state,
                h,
                arrival,
                fl_choices,
                mach_choices,
                target_fi=all_fl,
                arrival_fi=all_fl,
                src_fi=np.full(n_fl, ground_fi, dtype=np.int64),
                src_cost=np.full(n_fl, state.best_cost[h, ground_fi], dtype=FLOAT_DTYPE),
                src_elapsed=np.full(n_fl, state.best_time[h, ground_fi], dtype=FLOAT_DTYPE),
                lead_dist=climb_dist,
                trail_dist=np.zeros(n_fl, dtype=FLOAT_DTYPE),
                fixed_fuel=climb_fuel,
                fixed_time=climb_time,
                cruise_mass=post_mass,
                post_mass=post_mass,
                feasible=feasible,
                atyp=atyp,
                cost_index=cost_index,
                eef_cost_factor=eef_cost_factor,
                allow_cooling_credit=allow_cooling_credit,
            )
            continue

        n_active = len(active)
        src_fl_active = fl_choices[active]
        src_mass_active = state.best_mass[h, active]
        src_cost_active = state.best_cost[h, active]
        src_elapsed_active = state.best_time[h, active]

        # Every active source against every candidate target FL, flattened row-major
        # (source varies slowest). Same-FL targets are the plain cruise action: the maneuver
        # model returns zero distance/time/fuel for them, so they advance exactly one
        # waypoint. Step-downs are always feasible in the sense that the aircraft can always
        # give up altitude; without them it could climb to a level it cannot sustain as fuel
        # burns off and have no way back down. They can still run out of track, which
        # ``_step_change`` reports.
        src_fl_2d = np.repeat(src_fl_active, n_fl)
        tgt_fl_2d = np.tile(fl_choices, n_active)
        mass_2d = np.repeat(src_mass_active, n_fl)

        step_dist, step_fuel, step_time, step_mass, arrival, feasible = _step_change(
            pa, h, src_fl_2d, tgt_fl_2d, mass_2d, fl_choices, atyp
        )

        # A step change of either sign is a lead maneuver over ``step_dist``, then level
        # cruise at the target FL for the rest of the segment.
        penalty = np.where(tgt_fl_2d != src_fl_2d, step_penalty_kg, 0.0).astype(FLOAT_DTYPE)
        feasible |= tgt_fl_2d == src_fl_2d

        # Only the final descent below can finish the flight.
        feasible &= arrival != h_dest

        _relax_transitions(
            pa,
            state,
            h,
            arrival,
            fl_choices,
            mach_choices,
            target_fi=np.tile(all_fl, n_active),
            arrival_fi=np.tile(all_fl, n_active),
            src_fi=np.repeat(active, n_fl),
            src_cost=np.repeat(src_cost_active, n_fl),
            src_elapsed=np.repeat(src_elapsed_active, n_fl),
            lead_dist=step_dist,
            trail_dist=np.zeros(n_active * n_fl, dtype=FLOAT_DTYPE),
            fixed_fuel=step_fuel + penalty,
            fixed_time=step_time,
            cruise_mass=step_mass,
            post_mass=step_mass,
            feasible=feasible,
            atyp=atyp,
            cost_index=cost_index,
            eef_cost_factor=eef_cost_factor,
            allow_cooling_credit=allow_cooling_credit,
        )

        # Final descent: the subset of active sources at their FL's top-of-descent node
        at_tod = tod_node[active] == h
        if at_tod.any():
            act_d = active[at_tod]
            n_d = len(act_d)
            _relax_transitions(
                pa,
                state,
                h,
                np.full(n_d, h_dest, dtype=np.int64),
                fl_choices,
                mach_choices,
                target_fi=act_d,
                arrival_fi=np.full(n_d, ground_fi, dtype=np.int64),
                src_fi=act_d,
                src_cost=src_cost_active[at_tod],
                src_elapsed=src_elapsed_active[at_tod],
                lead_dist=np.zeros(n_d, dtype=FLOAT_DTYPE),
                trail_dist=fd_dist[act_d],
                fixed_fuel=fd_fuel[act_d],
                fixed_time=fd_time[act_d],
                cruise_mass=src_mass_active[at_tod],
                post_mass=src_mass_active[at_tod] - fd_fuel[act_d],
                feasible=np.ones(n_d, dtype=bool),
                atyp=atyp,
                cost_index=cost_index,
                eef_cost_factor=eef_cost_factor,
                allow_cooling_credit=allow_cooling_credit,
            )

    # The descent lands in the destination's ground slot. Mirror it into the cruise column
    # it descended from, which is where reconstruct_path starts its walk back.
    prev_fi = state.best_prev_fi[h_dest, ground_fi].item()
    if prev_fi >= 0:
        for arr in (
            state.best_cost,
            state.best_mass,
            state.best_time,
            state.best_mach,
            state.best_climb_dist,
            state.best_climb_time,
            state.best_prev_h,
            state.best_prev_fi,
        ):
            arr[h_dest, prev_fi] = arr[h_dest, ground_fi]

    return state


def _relax_track_batch(
    state: DAGState,
    arrival: npt.NDArray[np.int64],
    arrival_fi: npt.NDArray[np.int64],
    src_cost: npt.NDArray[FLOAT_DTYPE],
    src_h: int,
    src_fi: npt.NDArray[np.int64],
    fuel: npt.NDArray[FLOAT_DTYPE],
    duration: npt.NDArray[FLOAT_DTYPE],
    src_elapsed: npt.NDArray[FLOAT_DTYPE],
    arrival_mass: npt.NDArray[FLOAT_DTYPE],
    eef: npt.NDArray[FLOAT_DTYPE],
    feasible: npt.NDArray[np.bool_],
    mach: npt.NDArray[FLOAT_DTYPE],
    climb_dist: npt.NDArray[FLOAT_DTYPE],
    climb_time: npt.NDArray[FLOAT_DTYPE],
    cost_index: float,
    eef_cost_factor: float,
    allow_cooling_credit: bool,
) -> None:
    """Write a batch of candidate transitions into ``state`` via a scatter-min.

    Unlike a plain "compare current best, assign if better", this function is safe when multiple
    candidates in the SAME call converge on the same ``(arrival, arrival_fi)`` slot (e.g.,
    two different source flight levels whose climbs happen to land on the same waypoint and
    target level). This mirrors ``np.minimum.at`` within ``_relax_wavefront``.
    """
    priced_eef = eef if allow_cooling_credit else np.maximum(eef, 0.0)
    total = (
        src_cost
        + cost_index / 60.0 * duration
        + fuel
        + eef_cost_factor * priced_eef
        + JetA.ei_co2 * _J_PER_KG_CO2 * eef_cost_factor * fuel
    )
    total = np.where(feasible, total, np.inf)

    # Determining the winners can fail if the dtypes are different, so idiot check first
    if state.best_cost.dtype != total.dtype:
        raise RuntimeError(
            f"Dtype mismatch: state.best_cost is {state.best_cost.dtype}, "
            f"but total is {total.dtype}"
        )
    np.minimum.at(state.best_cost, (arrival, arrival_fi), total)

    winners = np.isfinite(total) & (total == state.best_cost[arrival, arrival_fi])
    if not winners.any():
        return

    idx = np.flatnonzero(winners)  # theoretically ties could happen, so the last gets chosen
    a, g = arrival[idx], arrival_fi[idx]
    state.best_mass[a, g] = arrival_mass[idx]
    state.best_time[a, g] = src_elapsed[idx] + duration[idx]
    state.best_mach[a, g] = mach[idx]
    state.best_climb_dist[a, g] = climb_dist[idx]
    state.best_climb_time[a, g] = climb_time[idx]
    state.best_prev_h[a, g] = src_h
    state.best_prev_fi[a, g] = src_fi[idx]


def estimate_flight_hours(
    origin: AirportCoords,
    dest: AirportCoords,
    mach_number: float = 0.75,
    max_headwind: float = 40.0,
) -> int:
    """Estimate an upper-bound for flight duration in hours.

    This function computes the worst-case flight time assuming the aircraft flies at
    ``mach_number`` at FL400 with a sustained headwind of ``max_headwind``.

    Parameters
    ----------
    origin : AirportCoords
        Origin airport.
    dest : AirportCoords
        Destination airport.
    mach_number : float, default 0.75
        Cruise Mach number. Suggest using the aircraft's design Mach number minus a margin.
    max_headwind : float, default 40.0
        Assumed maximum sustained headwind in m/s.

    Returns
    -------
    int
        Ceiling of estimated flight time in hours.
    """
    dist = geo.haversine(*origin.coords, *dest.coords)
    T_cold = units.m_to_T_isa(units.ft_to_m(40_000.0))
    min_tas = units.mach_number_to_tas(mach_number, T_cold)
    slow_gs = min_tas - max_headwind
    return int(np.ceil(dist / slow_gs / 3600.0))


def _estimate_trip_fuel(
    origin: AirportCoords,
    dest: AirportCoords,
    atyp: ps_aircraft_params.PSAircraftEngineParams,
    landing_mass: float,
) -> float:
    """Estimate trip fuel from great-circle distance and PS cruise performance.

    Uses a single cruise performance evaluation at FL350 and design Mach with
    an estimated mid-flight mass to compute fuel per meter, then scales by
    the great-circle distance. Assumes zero wind.
    """
    dist_m = geo.haversine(*origin.coords, *dest.coords).item()
    mid_fl = 35_000.0
    T_isa = units.m_to_T_isa(units.ft_to_m(mid_fl))
    est_mass = 0.5 * (landing_mass + atyp.amass_mtow)
    ff, _ = ps.cruise_performance(mid_fl, atyp.m_des, est_mass, T_isa, atyp)
    tas = units.mach_number_to_tas(atyp.m_des, T_isa)
    return (ff / tas * dist_m).item()


def _estimate_mass(
    payload: float | None,
    origin: AirportCoords,
    dest: AirportCoords,
    takeoff_time: pd.Timestamp,
    aircraft_type: str,
    atyp: ps_aircraft_params.PSAircraftEngineParams,
) -> tuple[float, float, float]:
    """Estimate payload, reserve fuel and trip fuel.

    Reserve fuel follows :func:`pycontrails.physics.jet.reserve_fuel_requirements` (we
    can't call that function directly because it needs a flown trajectory).
    Specifically, this function computes reserve fuel as the larger of:
        - 90 minutes of cruise fuel burn at the end-of-cruise mass, or
        - 15% of the total trip fuel
    """
    if payload is None:
        pax_lf = jet.passenger_load_factor(origin.icao_code, takeoff_time)
        n_seats = jet.number_of_seats(aircraft_type)
        dist_km = geo.haversine(*origin.coords, *dest.coords).item() / 1000.0
        cargo_lf = jet.cargo_load_factor(
            origin.icao_code, dest.icao_code, total_flight_dist=dist_km
        )
        payload = jet.aircraft_payload(
            max_payload=atyp.amass_mpl,
            n_seats=n_seats,
            pax_lf=pax_lf,
            cargo_lf=cargo_lf,
        )

    # 90 min of cruise ff at mid-range FL and mass
    mid_fl = 35_000.0
    T_isa = units.m_to_T_isa(units.ft_to_m(mid_fl))
    reserve_fuel = 0.0
    for _ in range(3):
        landing_mass = jet.initial_aircraft_mass(
            amass_oew=atyp.amass_oew,
            amass_mtow=atyp.amass_mtow,
            payload=payload,
            total_fuel_burn=0.0,
            total_reserve_fuel=reserve_fuel,
        )
        ff, feasible = ps.cruise_performance(mid_fl, atyp.m_des, landing_mass, T_isa, atyp)
        if not feasible:
            raise RuntimeError("Mid-range cruise should be feasible for mass estimation")
        trip_fuel = _estimate_trip_fuel(origin, dest, atyp, landing_mass)

        holding_fuel = ff.item() * 90.0 * 60.0  # kg/s -> kg for 90 minutes
        contingency_fuel = 0.15 * trip_fuel
        reserve_fuel = max(holding_fuel, contingency_fuel)

    return payload, reserve_fuel, trip_fuel


def _fl_choices(origin: AirportCoords, dest: AirportCoords) -> npt.NDArray[FLOAT_DTYPE]:
    """Return candidate cruising FLs based on eastbound/westbound rules."""
    az = geo.azimuth(*origin.coords, *dest.coords)
    eastbound = az % 360.0 < 180.0
    start = 29_000.0 if eastbound else 28_000.0
    return np.arange(start, 42_000.0, 2000.0, dtype=FLOAT_DTYPE)


def _mach_choices(atyp: ps_aircraft_params.PSAircraftEngineParams) -> npt.NDArray[FLOAT_DTYPE]:
    return np.arange(
        atyp.m_des // 0.01 * 0.01,  # floor to 2 decimal places
        atyp.max_mach_num,  # not inclusive
        0.01,
        dtype=FLOAT_DTYPE,
    )


def cruise_flight_levels(
    origin_icao: str | AirportCoords,
    dest_icao: str | AirportCoords,
) -> npt.NDArray[FLOAT_DTYPE]:
    """Determine the candidate cruise flight levels for a given origin-destination pair.

    This function applies the common eastbound/westbound flight levels rules of even flight
    levels for westbound flights and odd flight levels for eastbound flights.

    There is not per-aircraft-type ceiling applied.

    Parameters
    ----------
    origin_icao : str or AirportCoords
        ICAO code for the origin airport (e.g. ``"KLAX"``) or pre-fetched coordinates.
    dest_icao : str or AirportCoords
        ICAO code for the destination airport or pre-fetched coordinates.

    Returns
    -------
    npt.NDArray[FLOAT_DTYPE]
        Array of candidate cruise flight levels in feet (e.g. ``[29000., 31000., ..., 41000.]``).
    """
    origin = AirportCoords.from_icao(origin_icao) if isinstance(origin_icao, str) else origin_icao
    dest = AirportCoords.from_icao(dest_icao) if isinstance(dest_icao, str) else dest_icao
    return _fl_choices(origin, dest)


def _nearest_airport(lon: float, lat: float, which: str) -> str:
    """Find the ICAO code of the airport nearest a waypoint, or raise if none is found."""
    airports_df = airports.global_airport_database()
    airport_icao = airports.find_nearest_airport(airports_df, lon, lat, altitude=0.0)
    if airport_icao is None:
        raise ValueError(f"No airport found near flight {which}")
    return airport_icao


def _prepare_dag(
    origin: AirportCoords,
    dest: AirportCoords,
    dag: HorizontalDAG | Track | None,
    avoidance_regions: list[list[tuple[float, float]]] | None,
    **kwargs: Any,
) -> HorizontalDAG | Track:
    """Return the solver's DAG, normalized to ``FLOAT_DTYPE``.

    The parameter ``dag`` can be one of:

    - ``None``: build an airport-anchored ``HorizontalDAG`` from a Poisson-disc sampling.
    - ``HorizontalDAG``: cast to float32, check its endpoints agree with the airports, and
      apply avoidance regions.
    - ``Track``: a fixed cruise sub-track with mid-air endpoints; cast to float32 only. The
      airport-agreement check and avoidance regions do not apply to it.
    """
    if isinstance(dag, Track):
        if avoidance_regions:
            raise ValueError("avoidance_regions are not supported for a Track")
        return Track(
            lon=dag.lon.astype(FLOAT_DTYPE, copy=False),
            lat=dag.lat.astype(FLOAT_DTYPE, copy=False),
            node_time=dag.node_time,
        )

    if dag is None:
        dag = (
            HorizontalDAG.from_poisson(
                *origin.coords,
                *dest.coords,
                dtype=FLOAT_DTYPE,
                **kwargs,
            )
            .prune_edges(degree=8)  # could expose, but user can also pass a custom dag directly
            .prune_unreachable()
        )
    else:
        # A custom DAG can have different dtypes; normalize here
        dag = HorizontalDAG(
            lon=dag.lon.astype(FLOAT_DTYPE, copy=False),
            lat=dag.lat.astype(FLOAT_DTYPE, copy=False),
            edge_dist=dag.edge_dist.astype(FLOAT_DTYPE, copy=False),
            h_origin=dag.h_origin,
            h_dest=dag.h_dest,
            adj_ptr=dag.adj_ptr,
            adj=dag.adj,
        )
        _check_airport_agreement(dag, origin, dest)

    if avoidance_regions:
        dag = dag.exclude_polygons(avoidance_regions)

    return dag


def _prepare_static_graph(
    origin: AirportCoords,
    dest: AirportCoords,
    ds: xr.Dataset,
    avoidance_regions: list[list[tuple[float, float]]] | None,
    altitude_ft: npt.NDArray[np.floating],
    takeoff_time: pd.Timestamp,
    flight_hours: int,
    **kwargs
) -> tuple[HorizontalDAG, EdgeMetLookup]:
    """Return the solver's DAG and static met lookup based on a global graph."""
    dag = HorizontalDAG.from_static_graph(
        ds,
        *origin.coords,
        *dest.coords,
        dtype=FLOAT_DTYPE,
        **kwargs
    )

    if avoidance_regions:
        dag = dag.exclude_polygons(avoidance_regions)
    dag = dag.prune_unreachable()

    met_lookup = EdgeMetLookup.from_static_graph(
        ds=ds,
        dag=dag,
        altitude_ft=altitude_ft,
        takeoff_time=takeoff_time,
        flight_hours=flight_hours
    )

    return dag, met_lookup


def _check_airport_agreement(
    dag: HorizontalDAG, origin: AirportCoords, dest: AirportCoords
) -> None:
    """Raise if the DAG's origin/destination nodes don't sit at the airports (within 10 km)."""
    lon0 = dag.lon[dag.h_origin]
    lat0 = dag.lat[dag.h_origin]
    if geo.haversine(lon0, lat0, origin.longitude, origin.latitude) > 10_000.0:  # 10 km
        raise ValueError(
            f"DAG origin ({lon0}, {lat0}) does not agree with "
            f"airport {origin.icao_code} ({origin.longitude}, {origin.latitude})"
        )
    lon1 = dag.lon[dag.h_dest]
    lat1 = dag.lat[dag.h_dest]
    if geo.haversine(lon1, lat1, dest.longitude, dest.latitude) > 10_000.0:  # 10 km
        raise ValueError(
            f"DAG dest ({lon1}, {lat1}) does not agree with "
            f"airport {dest.icao_code} ({dest.longitude}, {dest.latitude})"
        )


class Optimizer:
    """Trajectory optimizer based on the PS model with optional climate cost function.

    The optimizer builds a horizontal directed acyclic graph between two airports,
    optionally interpolates met data onto edge sample points, and solves for the minimum-cost
    path across candidate flight levels and Mach numbers with :func:`solve_dag`.

    An instance built with :meth:`from_flight` instead holds the lateral path fixed to a flown
    trajectory and optimizes only the vertical profile and Mach number with :func:`solve_track`.
    The :attr:`kind` property reports which of the two variants an instance is.

    Parameters
    ----------
    origin_icao : str or AirportCoords
        ICAO code for the origin airport (e.g. ``"KLAX"``), or an ``AirportCoords`` instance.
    dest_icao : str or AirportCoords
        ICAO code for the destination airport, or an ``AirportCoords`` instance.
    aircraft_type : str
        Aircraft type key in the PS model parameter table (e.g. ``"A320"``).
    takeoff_time : pd.Timestamp
        Departure time, used for met interpolation.
    met : MetDataset or xr.Dataset or None, default None
        Gridded met data with ``air_temperature``, ``eastward_wind``, and ``northward_wind``.
        If *None*, cruise performance uses ISA temperatures and zero wind.
    eef : xr.DataArray or MetDataArray or None, default None
        Optional ``eef_per_m`` DataArray on its own lon/lat grid. If provided,
        EEF is interpolated onto sample points independently from the weather grid, avoiding
        the need to pre-merge onto a common grid. Takes precedence over ``eef_per_m`` in
        ``met`` if both are present. Assumed to adhere to pycontrails ``MetDataArray`` conventions.
    dag : HorizontalDAG or Track or None, default None
        Pre-built graph. If *None*, a :class:`HorizontalDAG` is generated via Poisson-disk sampling
        along the great circle. A supplied ``HorizontalDAG`` must have origin and destination
        nodes agreeing with the airport coordinates. A :class:`Track` is a fixed sequence of timed
        waypoints with mid-air endpoints, normally supplied by :meth:`from_flight` rather
        than directly.
    cost_index : float, default 60.0
        Fuel-vs-time tradeoff in kg per minute. Higher values penalize time more,
        favoring faster (and more fuel-intensive) routes.
    dollar_tonne_co2e : float, default 0.0
        Carbon price in US dollars per tonne (1000kg) of CO2-equivalent. A value of
        0.0 disables the carbon cost term. If positive, either ``met`` must contain a
        ``eef_per_m`` variable or the ``eef`` parameter must be provided.
    dollar_kg_fuel : float, default 1.0
        Fuel price in US dollars per kg. Only used to convert the carbon cost into the
        fuel-equivalent units of the objective function. Ignored if ``dollar_tonne_co2e`` is 0.0.
    step_penalty_kg : float, default 0.0
        Cost in kg of fuel charged for changing flight level, on top of the maneuver's own fuel
        and time, to discourage marginally beneficial steps. The initial climb and final descent
        are exempt. The penalty enters the objective only, not the aircraft mass, so reported fuel
        burn stays physical.
    met_spacing_m : float, default 25_000.0
        Spacing in meters between met sample points along each edge.
    aggregate_met : bool, default False
        Aggregate meteorology to hold a single value per edge after sampling based on 
        ``met_spacing_m``
    flight_hours : int or None, default None
        Upper-bound flight duration in hours for met time window. If None, estimated from
        the aircraft type. Providing an explicit value decouples the met lookup from the
        aircraft, allowing the user to call the :meth:`solve` method with a different aircraft
        type without re-initializing the optimizer.
    allow_cooling_credit : bool, default False
        If True, negative EEF (cooling contrails) reduces cost when ``dollar_tonne_co2e`` is set.
        If False, negative EEF is clipped to zero in the cost function but still reported
        in the output flight. Only used if ``dollar_tonne_co2e`` is set.
    avoidance_regions : list[list[tuple[float, float]]] or None, default None
        Polygons to exclude from the search, defined as lists of ``(lon, lat)`` vertices.
        Edges intersecting any polygon are removed and the :class:`HorizontalDAG` is re-pruned.
        Not supported when ``dag`` is a :class:`Track`.
    fl_choices : npt.NDArray[FLOAT_DTYPE] or None, default None
        Candidate cruise flight levels in feet. If *None*, the eastbound/westbound defaults
        from :func:`cruise_flight_levels` are used.
    **kwargs
        Additional parameters for :class:`HorizontalDAG` generation if ``dag`` is None. Passed into
        :meth:`HorizontalDAG.from_poisson`.
    """

    def __init__(
        self,
        origin_icao: str | AirportCoords,
        dest_icao: str | AirportCoords,
        aircraft_type: str,
        takeoff_time: pd.Timestamp,
        *,
        static_graph: xr.Dataset | None = None,
        met: MetDataset | xr.Dataset | None = None,
        eef: xr.DataArray | MetDataArray | None = None,
        dag: HorizontalDAG | Track | None = None,
        cost_index: float = 60.0,
        dollar_tonne_co2e: float = 0.0,
        dollar_kg_fuel: float = 1.0,
        step_penalty_kg: float = 0.0,
        met_spacing_m: float = 25_000.0,
        aggregate_met: bool = False,
        flight_hours: int | None = None,
        allow_cooling_credit: bool = False,
        avoidance_regions: list[list[tuple[float, float]]] | None = None,
        fl_choices: npt.NDArray[FLOAT_DTYPE] | None = None,
        **kwargs: Any,
    ) -> None:
        self.origin = (
            AirportCoords.from_icao(origin_icao) if isinstance(origin_icao, str) else origin_icao
        )
        self.dest = AirportCoords.from_icao(dest_icao) if isinstance(dest_icao, str) else dest_icao

        if takeoff_time.tzinfo:
            takeoff_time = takeoff_time.tz_convert("UTC").tz_localize(None)
        self.takeoff_time = takeoff_time
        self.cost_index = float(cost_index)
        self.dollar_tonne_co2e = float(dollar_tonne_co2e)
        self.dollar_kg_fuel = float(dollar_kg_fuel)
        self.step_penalty_kg = float(step_penalty_kg)
        self.allow_cooling_credit = allow_cooling_credit
        self.aircraft_type = aircraft_type
        self.atyp = ps_aircraft_params.load_aircraft_engine_params()[aircraft_type]

        if dollar_tonne_co2e:
            if met is None:
                raise ValueError("met must be provided when dollar_tonne_co2e is set")
            if "eef_per_m" not in met and eef is None:
                raise ValueError(
                    "met must contain 'eef_per_m' or eef must be provided"
                    " when dollar_tonne_co2e is set"
                )

        self.fl_choices = (
            fl_choices if fl_choices is not None else cruise_flight_levels(origin_icao, dest_icao)
        )
        self.mach_choices = _mach_choices(self.atyp)
        self.avoidance_regions = avoidance_regions

        # Set by from_flight: a validated (waypoint, altitude_ft) profile. When present,
        # solve() dispatches to solve_track instead of solve_dag.
        self.profile_ds: xr.Dataset | None = None

        # Set by from_flight(use_flown_climb_descent=True): the flown climb/descent below the
        # hand-off (lowest candidate FL) are taken as-is and only the cruise is optimized. When
        # set, solve_track starts/ends at ``cruise_elev_ft`` (the hand-off), the DP's initial
        # mass is reduced by the flown climb fuel, and to_flight splices the flown segments back.
        self.cruise_elev_ft: float | None = None
        self.prepend_flown: dict[str, npt.NDArray] | None = None
        self.append_flown: dict[str, npt.NDArray] | None = None

        self.result: DAGResult | None = None

        # Prepare DAG and meteorology from static graph if provided (most performance)
        if static_graph is not None:
            flight_hours = flight_hours or estimate_flight_hours(
                self.origin, self.dest, self.atyp.m_des
            )
            dag, met_lookup = _prepare_static_graph(
                self.origin,
                self.dest,
                static_graph,
                avoidance_regions=self.avoidance_regions,
                altitude_ft=self.fl_choices,
                takeoff_time=self.takeoff_time,
                flight_hours=flight_hours,
                **kwargs
            )
            self.dag = dag
            self.met_lookup = met_lookup
            return

        # Otherwise, fall back to dynamic methods (less efficient)
        self.dag = _prepare_dag(
            self.origin,
            self.dest,
            dag,
            self.avoidance_regions,
            **kwargs,
        )

        if met is not None:
            flight_hours = flight_hours or estimate_flight_hours(
                self.origin, self.dest, self.atyp.m_des
            )
            met_lookup = EdgeMetLookup.from_met(
                met=met,
                dag=self.dag,
                altitude_ft=self.fl_choices,
                takeoff_time=self.takeoff_time,
                flight_hours=flight_hours,
                spacing_m=met_spacing_m,
                eef=eef,
            )
            if aggregate_met:
                met_lookup = met_lookup.aggregate()
            self.met_lookup = met_lookup
        else:
            self.met_lookup = None



    @classmethod
    def from_flight(
        cls,
        flight: Flight,
        *,
        met: MetDataset | xr.Dataset | None = None,
        fl_profile: xr.Dataset | None = None,
        aircraft_type: str | None = None,
        origin_icao: str | None = None,
        dest_icao: str | None = None,
        eef: xr.DataArray | MetDataArray | None = None,
        altitude_ft: npt.NDArray[np.floating] | None = None,
        cost_index: float = 60.0,
        dollar_tonne_co2e: float = 0.0,
        dollar_kg_fuel: float = 1.0,
        allow_cooling_credit: bool = False,
        use_flown_climb_descent: bool = False,
    ) -> Self:
        """Build a vertical-profile optimizer from a :class:`pycontrails.Flight` trajectory.

        The optimized flight follows the flight's lateral path exactly, choosing flight level
        and Mach number along it via :func:`solve_track`. Weather comes from one of two
        sources:

        - ``met``: raw gridded 4D met, used to interpolate the flight's waypoints at each
          candidate flight level.
        - ``fl_profile``: an already-interpolated ``(waypoint, altitude_ft)`` dataset carrying
          ``air_temperature``, ``u_wind``, ``v_wind``, and optionally ``eef_per_m``, aligned
          waypoint-for-waypoint with ``flight``.

        Parameters
        ----------
        flight : Flight
            Trajectory supplying the lateral path, schedule, and (for ``use_flown_climb_descent``)
            the flown altitude profile.
        met : MetDataset or xr.Dataset or None
            Gridded met to interpolate. Mutually exclusive with ``fl_profile``.
        fl_profile : xr.Dataset or None
            Pre-interpolated per-waypoint met columns. Mutually exclusive with ``met``.
        aircraft_type : str or None
            PS model key. If *None*, taken from ``flight.attrs``.
        origin_icao, dest_icao : str or None
            ICAO codes. If *None*, taken from ``flight.attrs``, else the nearest airport.
        eef : xr.DataArray or MetDataArray or None
            Effective energy forcing per meter, if supplied separately from ``met``.
        altitude_ft : npt.NDArray[np.floating] or None
            Candidate flight levels in feet, used only with ``met``. If *None*, the
            eastbound/westbound defaults from :func:`cruise_flight_levels` are used. With
            ``fl_profile`` the levels come from its ``altitude_ft`` coordinate.
        cost_index : float, default 60.0
            Fuel-vs-time tradeoff in kg per minute.
        dollar_tonne_co2e : float, default 0.0
            Carbon price per tonne CO2-equivalent. If positive, the met source must carry
            ``eef_per_m``.
        dollar_kg_fuel : float, default 1.0
            Fuel price per kg, converting carbon cost into fuel-equivalent units.
        allow_cooling_credit : bool, default False
            If True, negative EEF reduces cost when ``dollar_tonne_co2e`` is set.
        use_flown_climb_descent : bool, default False
            If True, the flown initial climb and final descent below the lowest candidate
            flight level are taken from ``flight`` unchanged, and only the cruise phase above
            it is optimized.
        """
        if not flight:
            raise ValueError("Flight must be non-empty")
        if (met is None) == (fl_profile is None):
            raise ValueError("Provide exactly one of 'met' or 'fl_profile'")

        aircraft_type = aircraft_type or flight.get_constant("aircraft_type", None)
        if aircraft_type is None:
            raise ValueError("An 'aircraft_type' must be provided or present in flight.attrs")

        origin_icao = origin_icao or flight.get_constant("origin_airport", None)
        dest_icao = dest_icao or flight.get_constant("destination_airport", None)
        lon = flight["longitude"]
        lat = flight["latitude"]
        if origin_icao is None:
            origin_icao = _nearest_airport(lon[0].item(), lat[0].item(), "origin")
        if dest_icao is None:
            dest_icao = _nearest_airport(lon[-1].item(), lat[-1].item(), "destination")

        if fl_profile is not None:
            ds = fl_profile
        else:
            if altitude_ft is None:
                altitude_ft = cruise_flight_levels(origin_icao, dest_icao)
            ds = flight_profile_from_met(met, lon, lat, flight["time"], altitude_ft, eef=eef)

        return cls._from_profile(
            ds,
            flight,
            aircraft_type,
            origin_icao,
            dest_icao,
            cost_index=cost_index,
            dollar_tonne_co2e=dollar_tonne_co2e,
            dollar_kg_fuel=dollar_kg_fuel,
            allow_cooling_credit=allow_cooling_credit,
            use_flown_climb_descent=use_flown_climb_descent,
        )

    @classmethod
    def _from_profile(
        cls,
        ds: xr.Dataset,
        flight: Flight,
        aircraft_type: str,
        origin_icao: str,
        dest_icao: str,
        *,
        cost_index: float,
        dollar_tonne_co2e: float,
        dollar_kg_fuel: float,
        allow_cooling_credit: bool,
        use_flown_climb_descent: bool,
    ) -> Self:
        """Construct a track optimizer from a validated per-waypoint profile and its flight."""
        lon_f = flight["longitude"]
        lat_f = flight["latitude"]
        time_f = flight["time"]
        cruise_elev_ft: float | None

        if not use_flown_climb_descent:
            toc = 0
            tod = flight.size - 1
            cruise_elev_ft = None
        else:
            # Hand off to the optimizer at the lowest candidate flight level: reuse the flown
            # climb up to the first waypoint that reaches it, and the descent past the last.
            thres = ds["altitude_ft"].min().item()
            above = flight.altitude_ft >= thres
            if not above.any():
                raise ValueError(f"Flight never reaches the lowest candidate FL ({thres:.0f} ft)")

            reached = np.flatnonzero(above)
            toc = reached[0].item()
            tod = reached[-1].item()
            cruise_elev_ft = thres

        sl = slice(toc, tod + 1)
        lon, lat, time = lon_f[sl], lat_f[sl], time_f[sl]
        ds_cruise = ds.isel(waypoint=sl)

        # Drop really short segments that can mess with the optimizer
        keep = geo.segment_haversine(lon, lat) > 10.0  # drop < 10m segments
        keep[0] = True
        keep[-1] = True
        lon = lon[keep]
        lat = lat[keep]
        time = time[keep]
        ds_cruise = ds_cruise.isel(waypoint=keep)

        # The profile path follows a fixed track, so it needs only the ordered timed
        # waypoints, not an edge set -- a Track, not a HorizontalDAG.
        track = Track(lon=lon, lat=lat, node_time=time)

        opt = cls(
            origin_icao,
            dest_icao,
            aircraft_type,
            takeoff_time=pd.Timestamp(time[0]),
            met=None,
            dag=track,
            cost_index=cost_index,
            dollar_kg_fuel=dollar_kg_fuel,
            allow_cooling_credit=allow_cooling_credit,
            fl_choices=ds_cruise["altitude_ft"].values.astype(FLOAT_DTYPE),
        )

        # store the validated profile for solve_track to read by position.
        opt.profile_ds = validate_flight_profile(ds_cruise, opt.dag.n_nodes)

        if cruise_elev_ft is not None:
            opt.cruise_elev_ft = cruise_elev_ft
            opt.prepend_flown = {
                "longitude": lon_f[:toc],
                "latitude": lat_f[:toc],
                "altitude_ft": flight.altitude_ft[:toc],
                "time": time_f[:toc],
            }
            opt.append_flown = {
                "longitude": lon_f[tod + 1 :],
                "latitude": lat_f[tod + 1 :],
                "altitude_ft": flight.altitude_ft[tod + 1 :],
                "cum_time_offset": time_f[tod + 1 :] - time_f[tod],
            }

        # dollar_tonne_co2e is applied after the profile is attached so we can validate against
        # the profile's own variables rather than the (absent) gridded met in __init__.
        if dollar_tonne_co2e:
            if "eef_per_m" not in opt.profile_ds:
                raise ValueError("ds must contain 'eef_per_m' when dollar_tonne_co2e is set")
            opt.dollar_tonne_co2e = float(dollar_tonne_co2e)

        return opt

    @property
    def eef_cost_factor(self) -> float:
        """Compute the kg-fuel-equivalent cost per J of effective energy forcing."""
        return self.dollar_tonne_co2e / (J_PER_TONNE_CO2 * self.dollar_kg_fuel)

    @property
    def kind(self) -> str:
        """Return the optimization variant.

        Returns ``"track"`` for a fixed-path vertical-profile 2d optimizer (built via
        :meth:`from_flight`), or ``"dag"`` for the full 4d lateral-plus-vertical DAG optimizer.
        """
        return "track" if self.profile_ds is not None else "dag"

    def __repr__(self) -> str:
        status = "solved" if self.result is not None else "unsolved"
        if self.profile_ds is not None:
            met = "profile"
        elif self.met_lookup is not None:
            met = "with met"
        else:
            met = "no met"
        name = type(self).__name__
        return (
            f"{name}({self.origin.icao_code} -> {self.dest.icao_code}, "
            f"{self.aircraft_type}, {self.takeoff_time}, "
            f"{self.dag.n_nodes} nodes, {met}, {status})"
        )

    def solve(
        self,
        n_iter: int = 3,
        cost_index: float | None = None,
        dollar_tonne_co2e: float | None = None,
        aircraft_type: str | None = None,
        payload: float | None = None,
        allow_cooling_credit: bool | None = None,
        step_penalty_kg: float | None = None,
    ) -> DAGResult:
        """Solve the trajectory optimization via shortest-path dynamic programming on the DAG.

        This method iteratively re-solves the DAG to converge on takeoff mass.

        - Estimate trip fuel from great-circle distance and set initial takeoff mass
          as ``landing mass + estimated trip fuel``, capped at MTOW
        - Solve the dynamic program to find the optimal path and trip fuel
        - Update takeoff mass as ``landing mass + trip fuel``, capped at MTOW
        - Stop after the takeoff mass estimate converges or after ``n_iter`` iterations

        Parameters
        ----------
        n_iter : int, default 3
            Maximum number of mass-convergence iterations. Each iteration re-solves the full DP.
        cost_index : float or None, default None
            If provided, updates ``self.cost_index`` before solving. This parameter is safe to vary
            between calls without rebuilding intermediate artifacts.
        dollar_tonne_co2e : float or None, default None
            If provided, updates ``self.dollar_tonne_co2e`` before solving. Safe to vary between
            calls without rebuilding intermediate artifacts.
        aircraft_type : str or None, default None
            If provided, updates ``self.aircraft_type``, ``self.atyp``, and ``self.mach_choices``
            before solving. Safe to vary between calls without rebuilding the DAG or met lookup
            provided the met lookup was built with a sufficiently long ``flight_hours`` window
            to accommodate the new aircraft's speed.
        payload : float or None, default None
            Aircraft payload in kg if known. If None, this is estimated with pycontrails.
        allow_cooling_credit : bool or None, default None
            If provided, updates ``self.allow_cooling_credit`` before solving.

        Returns
        -------
        DAGResult
            DP state, takeoff mass, trip fuel, payload, reserve fuel, and landing mass.
        """
        if cost_index is not None:
            self.cost_index = float(cost_index)
        if dollar_tonne_co2e is not None:
            self.dollar_tonne_co2e = float(dollar_tonne_co2e)
            if self.dollar_tonne_co2e:
                if self.profile_ds is not None:
                    if "eef_per_m" not in self.profile_ds:
                        raise ValueError("profile must contain 'eef_per_m' when pricing carbon")
                elif self.met_lookup is None:
                    raise ValueError("met must be provided when dollar_tonne_co2e is set")
                elif "eef_per_m" not in self.met_lookup.ds:
                    raise ValueError("met must contain 'eef_per_m' when dollar_tonne_co2e is set")
        if allow_cooling_credit is not None:
            self.allow_cooling_credit = allow_cooling_credit
        if step_penalty_kg is not None:
            self.step_penalty_kg = float(step_penalty_kg)
        if aircraft_type is not None:
            self.aircraft_type = aircraft_type
            self.atyp = ps_aircraft_params.load_aircraft_engine_params()[aircraft_type]
            self.mach_choices = _mach_choices(self.atyp)

        payload, reserve_fuel, fuel_estimate = _estimate_mass(
            payload,
            self.origin,
            self.dest,
            self.takeoff_time,
            self.aircraft_type,
            self.atyp,
        )
        mass_kwargs = {
            "amass_oew": self.atyp.amass_oew,
            "amass_mtow": self.atyp.amass_mtow,
            "payload": payload,
            "total_reserve_fuel": reserve_fuel,
        }
        amass_init = jet.initial_aircraft_mass(total_fuel_burn=fuel_estimate, **mass_kwargs)

        on_track = self.profile_ds is not None

        # When the flown climb is reused, the DP starts at the hand-off altitude, not the
        # ground. Estimate the fuel burned climbing there and reduce the initial mass, and
        # have solve_track climb/descend from the hand-off elevation rather than the airport.
        climb_elev = self.origin.elevation_ft
        dest_elev = self.dest.elevation_ft
        if self.cruise_elev_ft is not None:
            # climb_to_target returns (dist, fuel, time, mass)
            flown_climb_fuel = ps.climb_to_target(
                amass_init, self.origin.elevation_ft, self.cruise_elev_ft, self.atyp
            )[1]
            amass_init -= flown_climb_fuel
            climb_elev = self.cruise_elev_ft
            dest_elev = self.cruise_elev_ft

        for _ in range(n_iter):
            if on_track:
                state = solve_track(
                    dag=self.dag,
                    profile=self.profile_ds,
                    amass_init=amass_init,
                    fl_choices=self.fl_choices,
                    mach_choices=self.mach_choices,
                    atyp=self.atyp,
                    cost_index=self.cost_index,
                    eef_cost_factor=self.eef_cost_factor,
                    origin_elev_ft=climb_elev,
                    dest_elev_ft=dest_elev,
                    allow_cooling_credit=self.allow_cooling_credit,
                    step_penalty_kg=self.step_penalty_kg,
                )
            else:
                state = solve_dag(
                    dag=self.dag,
                    amass_init=amass_init,
                    fl_choices=self.fl_choices,
                    mach_choices=self.mach_choices,
                    atyp=self.atyp,
                    cost_index=self.cost_index,
                    eef_cost_factor=self.eef_cost_factor,
                    origin_elev_ft=self.origin.elevation_ft,
                    dest_elev_ft=self.dest.elevation_ft,
                    takeoff_time=self.takeoff_time,
                    met_lookup=self.met_lookup,
                    allow_cooling_credit=self.allow_cooling_credit,
                    step_penalty_kg=self.step_penalty_kg,
                )
            ground_fi = len(self.fl_choices)
            amass_final = state.best_mass[self.dag.h_dest, ground_fi].item()
            if not np.isfinite(amass_final):
                if on_track:
                    raise ValueError(
                        "No feasible path found along the track. The profile may be too "
                        "short for the initial climb and final descent to fit, or every "
                        "candidate flight level is infeasible at the required masses."
                    )
                raise ValueError(
                    "No feasible path found. DAG edges may be too short for the "
                    "initial climb or final descent. Try adjusting the max_dist_m "
                    "if providing a custom DAG."
                )

            trip_fuel = amass_init - amass_final
            new_amass_init = jet.initial_aircraft_mass(total_fuel_burn=trip_fuel, **mass_kwargs)
            if abs(new_amass_init - amass_init) < MASS_CONVERGENCE_KG:
                break
            amass_init = new_amass_init

        self.result = DAGResult(
            state=state,
            amass_init=amass_init,
            trip_fuel=trip_fuel,
            payload=payload,
            reserve_fuel=reserve_fuel,
            landing_mass=amass_final,
        )
        return self.result

    def reconstruct_path(
        self,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[FLOAT_DTYPE],
    ]:
        """Trace backpointers from destination to origin to recover the optimal path.

        The returned arrays are ordered origin-first.

        Geographic coordinates for each waypoint are available via
        ``self.dag.lon[path_h]`` and ``self.dag.lat[path_h]``.
        Flight levels are ``self.fl_choices[path_fl_idx]`` for interior
        waypoints; the origin uses a sentinel index (``len(fl_choices)``)
        representing ground level.

        Returns
        -------
        path_h : npt.NDArray[np.int64]
            Horizontal node indices along the path.
            The first and last entries are origin and destination.
        path_fl_idx : npt.NDArray[np.int64]
            Flight level index at each node. The origin uses a sentinel
            ``ground_fl_idx = len(fl_choices)``; the destination uses the
            actual cruise FL from which the final descent begins.
        path_mach : npt.NDArray[FLOAT_DTYPE]
            Cruise Mach number on each incoming leg. The first entry ``path_mach[0]`` is NaN
            (no incoming leg at the origin).
        """
        if self.result is None:
            raise ValueError("Call solve() first")

        state = self.result.state
        dag = self.dag
        n_fl = len(self.fl_choices)

        # Start at the best real FL at the destination — this is the FL the
        # aircraft cruised at before the final descent. We don't start at the
        # ground_fi sentinel because that hides which FL was actually used;
        # the solver could step-climb on the final edge from path_fl_idx[-2]
        # to a higher FL before descending, so the winning FL at the dest
        # isn't necessarily the same as the penultimate node's FL.
        best_dest_fi = np.argmin(state.best_cost[dag.h_dest, :n_fl]).item()
        if not np.isfinite(state.best_cost[dag.h_dest, best_dest_fi]):
            raise ValueError("No feasible path to destination")

        path_h, path_fl_idx, path_mach = [], [], []
        h, fi = dag.h_dest, best_dest_fi

        while True:  # Infinite loop if dag isn't a DAG
            path_h.append(h)
            path_fl_idx.append(fi)
            path_mach.append(state.best_mach[h, fi])
            prev_h = state.best_prev_h[h, fi]
            prev_fi = state.best_prev_fi[h, fi]
            if prev_fi == -1:
                break
            h, fi = prev_h, prev_fi

        path_h.reverse()
        path_fl_idx.reverse()
        path_mach.reverse()

        if path_h[0] != dag.h_origin:  # this should never occur if solver worked correctly
            raise RuntimeError("Path reconstruction did not reach origin")

        return np.array(path_h), np.array(path_fl_idx), np.array(path_mach)

    def _resampled_segments(
        self,
        h_src: int,
        h_dst: int,
        fi_src: int,
        fi_dst: int,
        leg_mach: float,
        resample_m: float,
        skip_first: bool,
    ) -> npt.NDArray[_SEGMENT_DTYPE]:
        """Emit segments with cost data by resampling a single edge."""
        met_lookup = self.met_lookup
        state = self.result.state
        dag = self.dag

        edge_dist = geo.haversine(dag.lon[h_src], dag.lat[h_src], dag.lon[h_dst], dag.lat[h_dst])
        n_samp = np.maximum(np.ceil(edge_dist / resample_m).astype(int) + 1, 2)
        cum_dist = np.linspace(0.0, edge_dist, n_samp)
        segment_frac = 1.0 / (n_samp - 1)

        if skip_first:
            cum_dist = cum_dist[1:]
            n_samp -= 1

        lon, lat, alt, elapsed, _, _, _, _ = self._resample_edge(
            h_src,
            h_dst,
            fi_src,
            fi_dst,
            cum_dist
        )

        # Apportion costs based on segment length
        cost = state.best_cost  # cumulative
        mass = state.best_mass
        eef = state.best_eef
        edge_cost = cost[h_dst, fi_dst] - cost[h_src, fi_src]
        edge_fuel = mass[h_src, fi_src] - mass[h_dst, fi_dst]
        edge_eef = eef[h_dst, fi_dst]
        segment_cost = np.full(n_samp, edge_cost * segment_frac)
        segment_fuel = np.full(n_samp, edge_fuel * segment_frac)
        segment_eef = np.full(n_samp, edge_eef * segment_frac)
        if not skip_first:
            segment_cost[0] = np.nan
            segment_fuel[0] = np.nan
            segment_eef[0] = np.nan

        out = np.empty(n_samp, dtype=_SEGMENT_DTYPE)
        out["longitude"] = lon
        out["latitude"] = lat
        out["altitude_ft"] = alt
        out["elapsed_s"] = elapsed
        out["cost"] = segment_cost
        out["fuel"] = segment_fuel
        out["eef"] = segment_eef
        return out

    def _track_waypoints(
        self,
        h_src: int,
        h_dst: int,
        fi_src: int,
        fi_dst: int,
        leg_mach: float,
        skip_first: bool,
    ) -> npt.NDArray[_WAYPOINT_DTYPE]:
        """Emit waypoints for one leg of a track solution.

        Every waypoint the leg spans is emitted, so the flight follows the supplied path.
        Both altitude and elapsed time interpolate linearly by distance between the leg's
        DP endpoints. That is exact for a level cruise; for a climb or descent leg it is an
        approximation, since those phases run slower than cruise -- the intermediate
        waypoint times are slightly off while the endpoints are exact.
        """
        state = self.result.state
        dag = self.dag
        ds = self.profile_ds
        n_fl = len(self.fl_choices)
        ground_fi = n_fl

        nodes = np.arange(h_src + int(skip_first), h_dst + 1, dtype=np.int64)
        n_samp = len(nodes)

        cum = dag.cum_dist
        span = (cum[h_dst] - cum[h_src]).item()
        frac = (cum[nodes] - cum[h_src]) / span if span > 0.0 else np.zeros(n_samp)

        # The destination is reached at the ground slot, so take the FL descended from
        cruise_fi = fi_dst if fi_dst < n_fl else fi_src
        # When the climb/descent come from the flight, the cruise starts and ends at the
        # hand-off altitude, not the airport ground; the flown segments below splice on later.
        ground_alt = self.cruise_elev_ft if self.cruise_elev_ft is not None else None
        src_alt = (
            (ground_alt if ground_alt is not None else self.origin.elevation_ft)
            if fi_src == ground_fi
            else self.fl_choices[fi_src]
        )
        dst_alt = (
            (ground_alt if ground_alt is not None else self.dest.elevation_ft)
            if h_dst == dag.h_dest
            else self.fl_choices[cruise_fi]
        )

        t_src = state.best_time[h_src, fi_src]
        t_dst = state.best_time[h_dst, fi_dst]
        elapsed = t_src + frac * (t_dst - t_src)

        eef = ds["eef_per_m"].values[nodes, cruise_fi] if "eef_per_m" in ds else 0.0

        out = np.empty(n_samp, dtype=_WAYPOINT_DTYPE)
        out["longitude"] = dag.lon[nodes]
        out["latitude"] = dag.lat[nodes]
        out["altitude_ft"] = src_alt + frac * (dst_alt - src_alt)
        out["elapsed_s"] = elapsed
        out["mach_number"] = leg_mach
        out["eef_per_m"] = eef
        out["air_temperature"] = ds["air_temperature"].values[nodes, cruise_fi]
        out["tailwind"] = -ds["headwind"].values[nodes, cruise_fi]
        out["northward_wind"] = ds["v_wind"].values[nodes, cruise_fi]
        out["node_index"] = nodes
        out["sample_index"] = nodes
        return out

    def _edge_waypoints(
        self,
        h_src: int,
        h_dst: int,
        fi_src: int,
        fi_dst: int,
        edge_mach: float,
        skip_first: bool,
    ) -> npt.NDArray[_WAYPOINT_DTYPE]:
        """Build densified waypoints for a single path edge using met sample points.

        Returns a structured array with dtype ``_WAYPOINT_DTYPE``.
        """
        met_lookup = self.met_lookup
        dag = self.dag

        edge_idx = dag.edge_index(h_src, h_dst)
        cruise_fi = fi_dst

        # Sample range (skip source on subsequent edges to avoid duplication)
        s0 = met_lookup.edge_ptr[edge_idx]
        s1 = met_lookup.edge_ptr[edge_idx + 1]
        if skip_first:
            s0 += 1
        sample_idxs = np.arange(s0, s1)
        n_samp = len(sample_idxs)
        cum_dist = met_lookup.cum_dist[sample_idxs]

        # Resample trajectory
        lon, lat, alt, elapsed, is_first, is_last, in_climb, in_descent = self._resample_edge(
            h_src,
            h_dst,
            fi_src,
            fi_dst,
            cum_dist
        )

        # Interpolate met
        times_dt = np.datetime64(self.takeoff_time) + (elapsed * 1e9).astype("timedelta64[ns]")
        fl_arr = np.array([cruise_fi])
        interp = met_lookup(sample_idxs, times_dt[:, np.newaxis], fl_idx=fl_arr)

        eef_per_m = (
            interp.eef_per_m[:, 0]
            if interp.eef_per_m is not None
            else np.zeros(n_samp, dtype=FLOAT_DTYPE)
        )

        out = np.empty(n_samp, dtype=_WAYPOINT_DTYPE)
        out["longitude"] = lon
        out["latitude"] = lat
        out["altitude_ft"] = alt
        out["elapsed_s"] = elapsed
        out["mach_number"] = edge_mach
        if is_first and np.any(in_climb):
            out["mach_number"][in_climb] = ps.mach_schedule(alt[in_climb], self.atyp)
        elif not is_first and np.any(in_climb):
            out["mach_number"][in_climb] = self.atyp.m_des
        if is_last and np.any(in_descent):
            out["mach_number"][in_descent] = ps.mach_schedule(alt[in_descent], self.atyp)
        out["eef_per_m"] = eef_per_m
        out["air_temperature"] = interp.air_temperature[:, 0]
        out["tailwind"] = interp.headwind[:, 0]
        out["node_index"] = -1
        if not skip_first:
            out["node_index"][0] = h_src
        out["node_index"][-1] = h_dst
        out["sample_index"] = sample_idxs
        return out

    def _resample_edge(
        self,
        h_src: int,
        h_dst: int,
        fi_src: int,
        fi_dst: int,
        cum_dist: npt.NDArray[np.floating]
    ) -> tuple[
        npt.NDArray[FLOAT_DTYPE],
        npt.NDArray[FLOAT_DTYPE],
        npt.NDArray[FLOAT_DTYPE],
        npt.NDArray[FLOAT_DTYPE],
        bool,
        bool,
        npt.NDArray[np.bool],
        npt.NDArray[np.bool]
    ]:
        """Resample trajectory at specified cumulative distances along an edge."""
        state = self.result.state
        dag = self.dag

        edge_idx = dag.edge_index(h_src, h_dst)
        edge_dist = dag.edge_dist[edge_idx]
        n_samp = cum_dist.size

        t_src = state.best_time[h_src, fi_src]
        t_dst = state.best_time[h_dst, fi_dst]

        is_first = fi_src == len(self.fl_choices)
        is_last = h_dst == self.dag.h_dest
        cruise_fi = fi_dst
        cruise_fl = self.fl_choices[cruise_fi]

        # Climb dist/time from the solver state at the destination node
        climb_dist = state.best_climb_dist[h_dst, cruise_fi]
        climb_time_s = state.best_climb_time[h_dst, cruise_fi]
        src_alt = self.origin.elevation_ft if is_first else self.fl_choices[fi_src]

        # Descent (final edge only)
        if is_last:
            fd_d, _, fd_t = ps.final_descent(
                np.array([cruise_fl], dtype=FLOAT_DTYPE),
                self.dest.elevation_ft,
                self.atyp,
            )
            descent_dist = fd_d[0].item()
            descent_time = fd_t[0].item()
            dest_alt = self.dest.elevation_ft
        else:
            descent_dist = 0.0
            descent_time = 0.0
            dest_alt = cruise_fl

        # --- Altitude profile ---
        alt = np.full(n_samp, cruise_fl, dtype=FLOAT_DTYPE)
        in_climb = cum_dist < climb_dist
        if np.any(in_climb):
            frac = cum_dist[in_climb] / climb_dist  # climb_dist > 0 holds
            alt[in_climb] = src_alt + frac * (cruise_fl - src_alt)

        descent_start = edge_dist - descent_dist
        in_descent = np.zeros(n_samp, dtype=bool)
        if is_last:
            in_descent = cum_dist > descent_start
            if np.any(in_descent):
                frac = (cum_dist[in_descent] - descent_start) / descent_dist  # descent_dist > 0
                alt[in_descent] = cruise_fl + frac * (dest_alt - cruise_fl)

        # --- Time profile ---
        edge_time = t_dst - t_src
        cruise_time = edge_time - climb_time_s - descent_time  # must be non-negative
        cruise_dist_total = descent_start - climb_dist
        in_cruise = ~in_climb & ~in_descent

        elapsed = np.empty(n_samp, dtype=FLOAT_DTYPE)
        if np.any(in_climb):
            elapsed[in_climb] = t_src + climb_time_s * (cum_dist[in_climb] / climb_dist)
        if np.any(in_cruise):
            cfrac = (cum_dist[in_cruise] - climb_dist) / cruise_dist_total
            elapsed[in_cruise] = t_src + climb_time_s + cruise_time * cfrac
        if np.any(in_descent):
            dfrac = (cum_dist[in_descent] - descent_start) / descent_dist
            elapsed[in_descent] = t_src + climb_time_s + cruise_time + descent_time * dfrac

        # --- Horizontal trajectory ---
        lon, lat = slerp.gc_interp(
            dag.lon[h_src],
            dag.lat[h_src],
            dag.lon[h_dst],
            dag.lat[h_dst],
            cum_dist / edge_dist
        )

        return (
            lon.astype(FLOAT_DTYPE),
            lat.astype(FLOAT_DTYPE),
            alt,
            elapsed,
            is_first,
            is_last,
            in_climb,
            in_descent,
        )

    def to_flight(self, resample_m: float | None = None) -> Flight:
        """Return the optimal trajectory as a :class:`pycontrails.Flight`.

        When met data is available, waypoints are emitted at each edge sample
        point (~20 km spacing) with proper climb/descent altitude profiles and
        per-sample ``eef_per_m``. Without met, falls back to one waypoint per
        DAG node.

        If ``resample_m`` is provided, the flight is resampled so that no waypoints
        are separated by more then ``resample_m``. No meteorology is attached to
        resampled flights, but per-segment cost, fuel burn, and ef (if provided)
        are attached with incoming-leg semantics.

        The :meth:`solve()` method must be called first.
        """
        path_h, path_fl_idx, path_mach = self.reconstruct_path()
        state = self.result.state
        dag = self.dag
        n_fl = len(self.fl_choices)
        ground_fi = n_fl

        if resample_m is not None:
            resampled = [
                self._resampled_segments(
                    path_h[k],
                    path_h[k + 1],
                    path_fl_idx[k],
                    path_fl_idx[k + 1],
                    path_mach[k + 1],
                    resample_m,
                    skip_first=(k > 0)
                )
                for k in range(len(path_h) - 1)
            ]
            segments = np.concat(resampled)

            time = self.takeoff_time + pd.to_timedelta(segments["elapsed_s"], unit="s")
            return Flight(
                longitude=segments["longitude"],
                latitude=segments["latitude"],
                altitude_ft=segments["altitude_ft"],
                time=time,
                data={
                    "cost": segments["cost"],
                    "fuel": segments["fuel"],
                    "eef": segments["eef"],
                },
                aircraft_type=self.aircraft_type
            )

        if self.met_lookup is None and self.profile_ds is None:
            # No met at all: one waypoint per node, ISA-only solve.
            # Shift mach from incoming-leg to departing-leg semantics
            path_mach[:-1] = path_mach[1:]
            path_mach[-1] = 0.0

            altitude_ft = self.fl_choices[np.minimum(path_fl_idx, ground_fi - 1)]
            altitude_ft[0] = self.origin.elevation_ft
            altitude_ft[-1] = self.dest.elevation_ft

            elapsed_s = state.best_time[path_h, path_fl_idx]
            time = self.takeoff_time + pd.to_timedelta(elapsed_s, unit="s")
            return Flight(
                longitude=dag.lon[path_h],
                latitude=dag.lat[path_h],
                altitude_ft=altitude_ft,
                time=time,
                data={
                    "mach_number": path_mach,
                    "node_index": path_h,
                    "sample_index": np.full(len(path_h), -1, dtype=np.int64),
                },
                aircraft_type=self.aircraft_type,
            )

        if self.profile_ds is not None:
            legs = [
                self._track_waypoints(
                    path_h[k],
                    path_h[k + 1],
                    path_fl_idx[k],
                    path_fl_idx[k + 1],
                    path_mach[k + 1],
                    skip_first=(k > 0),
                )
                for k in range(len(path_h) - 1)
            ]
            has_eef = "eef_per_m" in self.profile_ds
        else:
            legs = [
                self._edge_waypoints(
                    path_h[k],
                    path_h[k + 1],
                    path_fl_idx[k],
                    path_fl_idx[k + 1],
                    path_mach[k + 1],
                    skip_first=(k > 0),
                )
                for k in range(len(path_h) - 1)
            ]
            has_eef = "eef_per_m" in self.met_lookup.ds
        wpts = np.concatenate(legs)

        lon = wpts["longitude"]
        lat = wpts["latitude"]
        alt = wpts["altitude_ft"]
        time = self.takeoff_time + pd.to_timedelta(wpts["elapsed_s"], unit="s")

        data = {
            "mach_number": wpts["mach_number"],
            "air_temperature": wpts["air_temperature"],
            "tailwind": wpts["tailwind"],
            "node_index": wpts["node_index"],
            "sample_index": wpts["sample_index"],
        }
        if has_eef:
            data["eef_per_m"] = wpts["eef_per_m"]

        # Splice the flown climb and descent back on
        if self.prepend_flown is not None:
            pre = self.prepend_flown
            app = self.append_flown
            app_time = time[-1] + app["cum_time_offset"]

            n_pre, n_app = len(pre["longitude"]), len(app["longitude"])
            lon = np.concatenate([pre["longitude"], lon, app["longitude"]])
            lat = np.concatenate([pre["latitude"], lat, app["latitude"]])
            alt = np.concatenate([pre["altitude_ft"], alt, app["altitude_ft"]])
            time = np.concatenate([pre["time"], time.to_numpy(), app_time])

            for key, fill in (
                ("mach_number", np.nan),
                ("air_temperature", np.nan),
                ("tailwind", np.nan),
                ("eef_per_m", np.nan),
                ("node_index", -1),
                ("sample_index", -1),
            ):
                if key in data:
                    dt = np.int64 if key.endswith("index") else data[key].dtype
                    data[key] = np.concatenate(
                        [np.full(n_pre, fill, dtype=dt), data[key], np.full(n_app, fill, dtype=dt)]
                    )

        return Flight(
            longitude=lon,
            latitude=lat,
            altitude_ft=alt,
            time=time,
            data=data,
            aircraft_type=self.aircraft_type,
        )

    def plot_met(
        self,
        altitude_ft: float | None = None,
        time: pd.Timestamp | None = None,
        ax: "GeoAxes | None" = None,
        show_eef: bool = True,
        **kwargs,
    ) -> "GeoAxes":
        """Plot met data on DAG nodes for a given flight level and time.

        Draws a wind quiver overlay. When ``eef_per_m`` is available in the
        :attr:`met_lookup`, also draws a scatter plot colored by EEF.

        Parameters
        ----------
        altitude_ft : float or None
            Flight level in feet (e.g. ``37000``). Snaps to the nearest available
            level. If *None*, uses the first available level.
        time : pd.Timestamp or None
            Time to select. Snaps to the nearest available time step. If *None*,
            uses the first available time step.
        ax : GeoAxes or None
            Cartopy GeoAxes to plot on. If None, calls ``self.dag.plot()`` to create one.
        **kwargs
            Passed to ``ax.quiver``.

        Returns
        -------
        GeoAxes
            The axes with the met overlay.
        """
        if self.met_lookup is None and self.profile_ds is None:
            raise ValueError("No met data available; pass met to Optimizer to use plot_met")

        if ax is None:
            ax = self.dag.plot(show_edges=False)

        # Draw avoidance regions (densify edges along geodesics)
        if self.avoidance_regions:
            for coords in self.avoidance_regions:
                poly_lons, poly_lats = [], []
                closed = [*coords, coords[0]]
                for (lon1, lat1), (lon2, lat2) in itertools.pairwise(closed):
                    poly_lons.append(lon1)
                    poly_lats.append(lat1)
                    gc_lon, gc_lat = slerp.gc_npts(lon1, lat1, lon2, lat2, 50)
                    poly_lons.extend(gc_lon)
                    poly_lats.extend(gc_lat)
                ax.fill(
                    poly_lons,
                    poly_lats,
                    transform=ax.projection,
                    alpha=0.3,
                    color="red",
                    zorder=3,
                )

        # Select per-node wind and EEF at the chosen flight level. The profile path stores met
        # directly per waypoint (no time dim); the gridded path reads it off edge samples.
        if self.profile_ds is not None:
            ds = self.profile_ds
            if altitude_ft is None:
                altitude_ft = ds["altitude_ft"][0].item()
            sel = ds.sel(altitude_ft=altitude_ft, method="nearest")
            node_lon = self.dag.lon
            node_lat = self.dag.lat
            eef = sel["eef_per_m"].values if "eef_per_m" in ds else None
            sel_time = None
        else:
            ds = self.met_lookup.ds
            if altitude_ft is None:
                altitude_ft = ds["altitude_ft"][0]
            if time is None:
                time = ds["time"][0]
            sel = ds.sel(altitude_ft=altitude_ft, time=time, method="nearest")
            # One sample index per node (first sample of each node's first outgoing edge)
            has_edges = self.dag.out_degree > 0
            node_sample_idx = self.met_lookup.edge_ptr[self.dag.adj_ptr[:-1][has_edges]]
            node_lon = self.dag.lon[has_edges]
            node_lat = self.dag.lat[has_edges]
            eef = sel.eef_per_m.values[node_sample_idx] if "eef_per_m" in ds else None
            sel_time = pd.Timestamp(sel["time"].item())

        if show_eef and eef is not None:
            finite = np.isfinite(eef)
            vmax = np.abs(eef[finite]).max()
            tcf = ax.tricontourf(
                node_lon[finite],
                node_lat[finite],
                eef[finite],
                levels=20,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                transform=ax.projection,
                zorder=0,
            )
            fig = ax.get_figure()
            pos = ax.get_position()
            cax = fig.add_axes([pos.x0 + 0.02, pos.y0 + 0.04, pos.width * 0.3, 0.015])
            fig.colorbar(tcf, cax=cax, orientation="horizontal", label="EEF (J/m)")

        fl = round(sel["altitude_ft"].item() / 100)
        title = f"FL{fl}" if sel_time is None else f"FL{fl} — {sel_time:%Y-%m-%d %H:%M UTC}"
        ax.set_title(title)
        return ax

    def animate_solve(
        self,
        display_fl_idx: int | None = None,
        ax: "GeoAxes | None" = None,
    ) -> "FuncAnimation":
        """Re-run the DP with converged mass and return a wavefront animation.

        :meth:`solve` must be called first. This re-runs a single :func:`solve_dag`
        pass with the converged ``amass_init``, capturing wavefront snapshots.

        Nodes are colored by ``best_cost[:, display_fl_idx]`` for a single FL.
        Edges whose source node has been processed are shown in blue;
        remaining edges are shown muted. The optimal path is then revealed
        dest to origin, matching the backpointer reconstruction order.

        Parameters
        ----------
        display_fl_idx : int or None
            FL index into ``fl_choices`` to display costs for. If *None*,
            uses the FL the optimal path spends the most legs at.
        ax : GeoAxes or None
            Cartopy GeoAxes to draw on. If *None*, a new figure is created.

        Returns
        -------
        FuncAnimation
            Wavefront animation with cost coloring and optimal path reveal.
        """
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.cm import ScalarMappable
        from matplotlib.collections import LineCollection
        from matplotlib.colors import BoundaryNorm

        if self.result is None:
            raise ValueError("Call solve() first")

        dag = self.dag
        n_fl = len(self.fl_choices)

        # Pick display FL from optimal path if not specified
        path_h, path_fi, _ = self.reconstruct_path()
        if display_fl_idx is None:
            ground_fi = n_fl
            cruise_fis = path_fi[path_fi != ground_fi]
            display_fl_idx = int(np.bincount(cruise_fis).argmax())
        display_fl = self.fl_choices[display_fl_idx]

        # Collect wavefront snapshots: node indices + cost at display FL
        wavefronts: list[npt.NDArray[np.int64]] = []
        cost_snapshots: list[npt.NDArray[FLOAT_DTYPE]] = []

        def capture(wave: npt.NDArray[np.int64], state: DAGState) -> None:
            wavefronts.append(wave)
            cost_snapshots.append(state.best_cost[:, display_fl_idx].copy())

        solve_dag(
            dag=dag,
            amass_init=self.result.amass_init,
            fl_choices=self.fl_choices,
            mach_choices=self.mach_choices,
            atyp=self.atyp,
            cost_index=self.cost_index,
            eef_cost_factor=self.eef_cost_factor,
            origin_elev_ft=self.origin.elevation_ft,
            dest_elev_ft=self.dest.elevation_ft,
            takeoff_time=self.takeoff_time,
            met_lookup=self.met_lookup,
            allow_cooling_credit=self.allow_cooling_credit,
            step_penalty_kg=self.step_penalty_kg,
            on_wavefront=capture,
        )

        # Cost range for colorbar (finite values from final snapshot)
        final_cost = cost_snapshots[-1]
        finite_mask = np.isfinite(final_cost)
        vmin = float(final_cost[finite_mask].min()) if finite_mask.any() else 0.0
        vmax = float(final_cost[finite_mask].max()) if finite_mask.any() else 1.0

        # Precompute all edge segments: (n_edges, 2, 2)
        edge_src = dag.edge_src
        all_segments = np.stack(
            [
                np.column_stack([dag.lon[edge_src], dag.lat[edge_src]]),
                np.column_stack([dag.lon[dag.adj], dag.lat[dag.adj]]),
            ],
            axis=1,
        )

        # Build animation on top of the DAG map
        pc = ccrs.PlateCarree()
        if ax is None:
            _, ax = plt.subplots(figsize=(24, 12), subplot_kw={"projection": pc})
        fig = ax.get_figure()
        dag.plot(ax=ax, show_edges=False)

        # Draw avoidance regions (densify edges along geodesics)
        if self.avoidance_regions:
            for coords in self.avoidance_regions:
                poly_lons, poly_lats = [], []
                closed = [*coords, coords[0]]
                for (lon1, lat1), (lon2, lat2) in itertools.pairwise(closed):
                    poly_lons.append(lon1)
                    poly_lats.append(lat1)
                    gc_lon, gc_lat = slerp.gc_npts(lon1, lat1, lon2, lat2, 50)
                    poly_lons.extend(gc_lon)
                    poly_lats.extend(gc_lat)
                ax.fill(poly_lons, poly_lats, transform=pc, alpha=0.3, color="red", zorder=3)

        # Unexplored edges (muted gray)
        unexplored_lc = LineCollection(
            all_segments,
            colors="silver",
            linewidths=0.15,
            alpha=0.6,
            transform=pc,
            zorder=4,
        )
        ax.add_collection(unexplored_lc)

        # Explored edges (blue)
        explored_lc = LineCollection(
            [],
            colors="steelblue",
            linewidths=0.15,
            alpha=0.2,
            transform=pc,
            zorder=5,
        )
        ax.add_collection(explored_lc)

        # Node scatter colored by cost at display FL
        cost_colors = np.full(dag.n_nodes, np.nan)
        cost_scatter = ax.scatter(
            dag.lon,
            dag.lat,
            c=cost_colors,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=12,
            transform=pc,
            zorder=6,
        )
        cax_cost = fig.add_axes([0.12, 0.06, 0.35, 0.02])
        fig.colorbar(
            cost_scatter, cax=cax_cost, orientation="horizontal", label="Cost (kg fuel eq.)"
        )

        # Current wavefront highlight
        wave_scatter = ax.scatter(
            [],
            [],
            s=40,
            facecolors="none",
            edgecolors="red",
            linewidths=1.5,
            transform=pc,
            zorder=7,
        )

        # Optimal path segments colored by FL (revealed dest -> origin after DP)
        path_h_rev = path_h[::-1]
        path_fi_rev = path_fi[::-1]  # fi at each node, dest -> origin
        # Build all path segments and their FL colors
        path_segments = np.stack(
            [
                np.column_stack([dag.lon[path_h_rev[:-1]], dag.lat[path_h_rev[:-1]]]),
                np.column_stack([dag.lon[path_h_rev[1:]], dag.lat[path_h_rev[1:]]]),
            ],
            axis=1,
        )
        # Each segment's FL: use the source node's fi (dest-side end)
        ground_fi = n_fl
        seg_fl = np.array(
            [
                self.fl_choices[fi] if fi != ground_fi else self.fl_choices[0]
                for fi in path_fi_rev[:-1]
            ]
        )

        fl_cmap = plt.colormaps["cool"].resampled(n_fl)
        # BoundaryNorm: one color per FL, boundaries at midpoints between FLs
        fl_step = self.fl_choices[1] - self.fl_choices[0] if n_fl > 1 else 2000.0
        fl_boundaries = np.concatenate(
            [
                [self.fl_choices[0] - fl_step / 2],
                (self.fl_choices[:-1] + self.fl_choices[1:]) / 2,
                [self.fl_choices[-1] + fl_step / 2],
            ]
        )
        fl_norm = BoundaryNorm(fl_boundaries, n_fl)
        path_lc = LineCollection(
            [],
            cmap=fl_cmap,
            norm=fl_norm,
            linewidths=2.5,
            transform=pc,
            zorder=9,
        )
        ax.add_collection(path_lc)
        cax_fl = fig.add_axes([0.55, 0.06, 0.35, 0.02])
        fl_cbar = fig.colorbar(
            ScalarMappable(norm=fl_norm, cmap=fl_cmap),
            cax=cax_fl,
            orientation="horizontal",
            label="Flight Level (ft)",
        )
        fl_cbar.set_ticks(self.fl_choices)
        fl_cbar.set_ticklabels([f"FL{fl / 100:.0f}" for fl in self.fl_choices])

        # Track explored nodes incrementally for edge classification
        explored = np.zeros(dag.n_nodes, dtype=bool)

        n_dp_frames = len(wavefronts)
        n_path_steps = len(path_h) - 1
        n_frames = n_dp_frames + n_path_steps

        def update(i: int) -> tuple:
            if i < n_dp_frames:
                wave = wavefronts[i]
                explored[wave] = True

                edge_mask = explored[edge_src]
                explored_lc.set_segments(all_segments[edge_mask])
                unexplored_lc.set_segments(all_segments[~edge_mask])

                # Update node colors from this frame's cost snapshot
                cost = cost_snapshots[i]
                cost_colors[:] = np.where(np.isfinite(cost), cost, np.nan)
                cost_scatter.set_array(cost_colors)

                wave_scatter.set_offsets(np.column_stack([dag.lon[wave], dag.lat[wave]]))
                path_lc.set_segments([])
                ax.set_title(
                    f"Wavefront {i + 1} / {n_dp_frames}  (nodes at FL{display_fl / 100:.0f})"
                )
            else:
                step = i - n_dp_frames + 1
                path_lc.set_segments(path_segments[:step])
                path_lc.set_array(seg_fl[:step])
                wave_scatter.set_offsets(np.empty((0, 2)))
                ax.set_title(f"Optimal path ({step} / {n_path_steps})")

            return unexplored_lc, explored_lc, cost_scatter, wave_scatter, path_lc

        return FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=200,
            blit=False,
        )
