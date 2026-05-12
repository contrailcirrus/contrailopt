"""Trajectory optimization with PS lookups on a horizontal DAG."""

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pycontrails import Flight, JetA, MetDataset
from pycontrails.core import airports
from pycontrails.models.ps_model import ps_aircraft_params
from pycontrails.physics import geo, jet, units

from contrailopt import ps
from contrailopt.dag import AirportCoords, EdgeMetLookup, HorizontalDAG

if TYPE_CHECKING:
    from cartopy.mpl.geoaxes import GeoAxes
    from matplotlib.animation import FuncAnimation

FLOAT_DTYPE = np.float32

_WAYPOINT_DTYPE = np.dtype(
    [
        ("longitude", FLOAT_DTYPE),
        ("latitude", FLOAT_DTYPE),
        ("altitude_ft", FLOAT_DTYPE),
        ("elapsed_s", FLOAT_DTYPE),
        ("mach_number", FLOAT_DTYPE),
        ("eef_per_m", FLOAT_DTYPE),
        ("air_temperature", FLOAT_DTYPE),
        ("eastward_wind", FLOAT_DTYPE),
        ("northward_wind", FLOAT_DTYPE),
        ("node_index", np.int64),
        ("sample_index", np.int64),
    ]
)

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

    Uses per-FL climb time and ISA-based TAS at each FL to approximate how
    long it takes the aircraft to reach each sample along the edge.

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
) -> tuple[
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[np.bool_],
    npt.NDArray[FLOAT_DTYPE],
]:
    """Compute per-edge cruise fuel, time, feasibility, and EEF from met samples.

    Met data is interpolated at each sample point along the edge for every
    candidate FL in ``fl_choices``. Cruise performance (fuel flow, TAS, wind)
    is evaluated at each candidate FL (the destination node FL, not the source node FL).
    For step-climbs, the cruise-zone weighting zeros out the climb portion so that only the
    level-flight segment contributes to fuel and time. For step-descents, climb_dist is
    zero, so the full edge is treated as cruise at the candidate FL.

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
    atyp : PSAircraftEngineParams
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

    # Cruise performance at each (sample, FL, Mach)
    air_temp_3d = sample_met.air_temperature[:, :, np.newaxis]
    mass_3d = post_climb_mass[sample_to_edge][:, :, np.newaxis]
    ff, feas = ps.cruise_performance(
        fl_choices[np.newaxis, :, np.newaxis],
        mach_choices[np.newaxis, np.newaxis, :],
        mass_3d,
        air_temp_3d,
        atyp,
    )
    tas = units.mach_number_to_tas(mach_choices[np.newaxis, np.newaxis, :], air_temp_3d)

    # Along-track ground speed
    az = met_lookup.sample_azimuth[sample_idxs][:, np.newaxis]  # (n_sample, 1)
    tailwind = sample_met.eastward_wind * np.sin(az) + sample_met.northward_wind * np.cos(az)
    ground_speed = tas + tailwind[:, :, np.newaxis]  # (n_sample, n_fl, n_mach)

    # Negative ground_speed could be handled gracefully with the feasible mask, but it's not
    # realistic and probably indicates an actual problem with the input
    if np.any(ground_speed < 0.0):
        raise RuntimeError("Negative ground speed: headwind exceeds TAS at some sample point")

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
    seg_time = seg_dist[:, np.newaxis, np.newaxis] / ground_speed * weight[:, :, np.newaxis]
    seg_fuel = ff * seg_time

    cruise_time = np.add.reduceat(seg_time, edge_bounds, axis=0)
    cruise_fuel = np.add.reduceat(seg_fuel, edge_bounds, axis=0)
    cruise_feasible = np.add.reduceat(~feas, edge_bounds, axis=0) == 0

    # Accumulate eef_per_m over the full edge (no cruise zone weighting)
    if sample_met.eef_per_m is not None:
        seg_eef = sample_met.eef_per_m * seg_dist[:, np.newaxis]
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
            best_climb_dist=np.full((n_h, n_cols), np.nan, dtype=FLOAT_DTYPE),
            best_climb_time=np.full((n_h, n_cols), np.nan, dtype=FLOAT_DTYPE),
            best_prev_h=np.full((n_h, n_cols), -1, dtype=np.int64),
            best_prev_fi=np.full((n_h, n_cols), -1, dtype=np.int64),
        )


@dataclass(kw_only=True, slots=True, frozen=True)
class DAGResult:
    """The output of Optimizer.solve()."""

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
) -> tuple[
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[np.bool_],
]:
    """Compute windless ISA climb from the ground to each candidate FL.

    Only used for the origin wavefront. Returns five arrays of shape ``(n_fl,)``.
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
        FLOAT_DTYPE(base_mass),
        atyp,
        delta_isa=0.0,
        tailwind=0.0,
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
    atyp : PSAircraftEngineParams
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

        az = met_lookup.sample_azimuth[edge_start]
        u = climb_met.eastward_wind[edge_arange, edge_src_fi]
        v = climb_met.northward_wind[edge_arange, edge_src_fi]
        tailwind = (u * np.sin(az) + v * np.cos(az))[:, np.newaxis]
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
    mach_3d = mach_choices[np.newaxis, np.newaxis, :]
    air_temperature_3d = units.m_to_T_isa(units.ft_to_m(fl_3d))
    mass_3d = edge_mass[:, :, np.newaxis]
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
        # Origin wavefront: ISA climb from ground, broadcast to (n_edge, n_fl)
        climb_dist, climb_fuel, climb_time, post_climb_mass, feasible = _compute_ground_climbs(
            fl_choices,
            src_masses[0],
            ctx.atyp,
            ctx.origin_elev_ft,
        )
        n_edge = len(src_idx)
        climb_dist = np.broadcast_to(climb_dist, (n_edge, n_fl))
        climb_fuel = np.broadcast_to(climb_fuel, (n_edge, n_fl))
        climb_time = np.broadcast_to(climb_time, (n_edge, n_fl))
        post_climb_mass = np.broadcast_to(post_climb_mass, (n_edge, n_fl))
        feasible = np.broadcast_to(feasible, (n_edge, n_fl))
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

    # The main cost function
    total_cost = (
        src_costs[src_idx, np.newaxis, np.newaxis]
        + climb_cost[:, :, np.newaxis]
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
    met_lookup: EdgeMetLookup | None = None,
    allow_cooling_credit: bool = False,
    on_wavefront: Callable[[npt.NDArray[np.int64], DAGState], None] | None = None,
) -> DAGState:
    """Solve shortest-path DP on the topo-sorted DAG, tracking mass exactly."""
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


def estimate_flight_hours(
    origin: AirportCoords,
    dest: AirportCoords,
    mach_number: float = 0.75,
    max_headwind: float = 40.0,
) -> int:
    """Estimate upper-bound flight duration in hours.

    Computes the worst-case flight time assuming the aircraft flies at
    ``mach_number`` at FL400 with a sustained headwind of ``max_headwind``.

    Parameters
    ----------
    origin : AirportCoords
        Origin airport.
    dest : AirportCoords
        Destination airport.
    mach_number : float, default 0.75
        Cruise Mach number. Use the slowest aircraft's design Mach minus a margin.
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
    origin_icao: str,
    dest_icao: str,
    takeoff_time: pd.Timestamp,
    aircraft_type: str,
    atyp: ps_aircraft_params.PSAircraftEngineParams,
) -> tuple[float, float]:
    """Estimate payload and reserve fuel.

    Reserve fuel is 90 minutes of cruise fuel flow at a mid-range FL and mass,
    following the approach in ``pycontrails.models.ps_model.ps_grid``.
    """
    if payload is None:
        pax_lf = jet.passenger_load_factor(origin_icao, takeoff_time)
        n_seats = jet.number_of_seats(aircraft_type)
        cargo_lf = jet.cargo_load_factor(origin_icao, dest_icao)
        payload = jet.aircraft_payload(
            max_payload=atyp.amass_mpl,
            n_seats=n_seats,
            pax_lf=pax_lf,
            cargo_lf=cargo_lf,
        )

    # 90 min of cruise ff at mid-range FL and mass
    mid_fl = 35_000.0
    T_isa = units.m_to_T_isa(units.ft_to_m(mid_fl))
    est_mass = atyp.amass_oew + payload + 0.5 * (atyp.amass_mtow - atyp.amass_oew)
    ff, feasible = ps.cruise_performance(mid_fl, atyp.m_des, est_mass, T_isa, atyp)
    if not feasible:
        raise RuntimeError("Mid-range cruise should be feasible for mass estimation")

    reserve_fuel = ff.item() * 90.0 * 60.0  # kg/s -> kg for 90 minutes

    return payload, reserve_fuel


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

    This function applies the common eastbound/westbound FL rules of even FLs for westbound
    flights and odd FLs for eastbound flights. There is not per-aircraft-type ceiling
    applied (this could be added if needed).

    Parameters
    ----------
    origin_icao : str | AirportCoords
        ICAO code for the origin airport (e.g. ``"KLAX"``) or pre-fetched coordinates.
    dest_icao : str | AirportCoords
        ICAO code for the destination airport or pre-fetched coordinates.

    Returns
    -------
    npt.NDArray[FLOAT_DTYPE]
        Array of candidate cruise flight levels in feet (e.g. ``[29000., 31000., ..., 41000.]``).
    """
    origin = AirportCoords.from_icao(origin_icao) if isinstance(origin_icao, str) else origin_icao
    dest = AirportCoords.from_icao(dest_icao) if isinstance(dest_icao, str) else dest_icao
    return _fl_choices(origin, dest)


def _build_dag(
    origin: AirportCoords,
    dest: AirportCoords,
    dag: HorizontalDAG | None,
    avoidance_regions: list[list[tuple[float, float]]] | None,
) -> HorizontalDAG:
    """Validate or build a DAG, then apply avoidance regions."""
    if dag is not None:
        # A custom DAG can have different dtypes; adjust here
        dag = HorizontalDAG(
            lon=dag.lon.astype(FLOAT_DTYPE, copy=False),
            lat=dag.lat.astype(FLOAT_DTYPE, copy=False),
            edge_dist=dag.edge_dist.astype(FLOAT_DTYPE, copy=False),
            h_origin=dag.h_origin,
            h_dest=dag.h_dest,
            adj_ptr=dag.adj_ptr,
            adj=dag.adj,
        )

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
    else:
        dag = HorizontalDAG.from_poisson(
            *origin.coords,
            *dest.coords,
            dtype=FLOAT_DTYPE,
        ).prune_unreachable()

    if avoidance_regions:
        dag = dag.exclude_polygons(avoidance_regions)

    return dag


class Optimizer:
    """Trajectory optimizer for a single origin-destination pair based on the PS model.

    The optimizer builds a horizontal directed acyclic graph between two airports,
    optionally interpolates met data onto edge sample points, and solves for the minimum-cost
    path across valid flight levels and Mach numbers.

    Parameters
    ----------
    origin_icao : str | AirportCoords
        ICAO code for the origin airport (e.g. ``"KLAX"``), or an ``AirportCoords`` instance.
    dest_icao : str | AirportCoords
        ICAO code for the destination airport, or an ``AirportCoords`` instance.
    aircraft_type : str
        Aircraft type key in the PS model parameter table (e.g. ``"A320"``).
    takeoff_time : pd.Timestamp
        Departure time, used for met interpolation.
    met : MetDataset | xr.Dataset | None, default None
        Gridded met data with ``air_temperature``, ``eastward_wind``, and ``northward_wind``.
        If *None*, cruise performance uses ISA temperatures and zero wind.
    dag : HorizontalDAG or None, default None
        Pre-built DAG. If *None*, a DAG is generated via Poisson-disk sampling along the
        great circle. The DAG origin and destination must agree with the airport coordinates.
        The DAG is expected to be pruned (``HorizontalDAG.prune()``) but this is not enforced.
    cost_index : float, default 60.0
        Fuel-vs-time tradeoff in kg per minute. Higher values penalize time more,
        favoring faster (and more fuel-intensive) routes.
    dollar_tonne_co2e : float, default 0.0
        Carbon price in US dollars per tonne (1000kg) of CO2-equivalent. A value of
        0.0 disables the carbon cost term. If positive, the ``met`` parameter must be provided
        with a ``eef_per_m`` variable giving the expected effective energy forcing in J per meter.
    dollar_kg_fuel : float, default 1.0
        Fuel price in US dollars per kg. Only used to convert the carbon cost into the
        fuel-equivalent units of the objective function. Ignored if ``dollar_tonne_co2e`` is 0.0.
    met_spacing_m : float, default 20_000.0
        Spacing in meters between met sample points along each edge.
    flight_hours : int or None, default None
        Upper-bound flight duration in hours for met time window. If None, estimated from
        the aircraft type. Providing an explicit value decouples the met lookup from the
        aircraft, allowing the user to call the ``solve()`` method with a different aircraft
        type without re-initializing the optimizer.
    allow_cooling_credit : bool, default False
        If True, negative EEF (cooling contrails) reduces cost when ``dollar_tonne_co2e`` is set.
        If False, negative EEF is clipped to zero in the cost function but still reported
        in the output flight. Only used if ``dollar_tonne_co2e`` is set.
    avoidance_regions : list of polygon coordinate lists, or None
        Polygons to exclude from the search, defined as lists of ``(lon, lat)`` vertices.
        Edges intersecting any polygon are removed and the DAG is re-pruned.
    """

    def __init__(
        self,
        origin_icao: str | AirportCoords,
        dest_icao: str | AirportCoords,
        aircraft_type: str,
        takeoff_time: pd.Timestamp,
        *,
        met: MetDataset | xr.Dataset | None = None,
        dag: HorizontalDAG | None = None,
        cost_index: float = 60.0,
        dollar_tonne_co2e: float = 0.0,
        dollar_kg_fuel: float = 1.0,
        met_spacing_m: float = 20_000.0,
        flight_hours: int | None = None,
        allow_cooling_credit: bool = False,
        avoidance_regions: list[list[tuple[float, float]]] | None = None,
    ) -> None:
        self.origin = (
            AirportCoords.from_icao(origin_icao) if isinstance(origin_icao, str) else origin_icao
        )
        self.dest = AirportCoords.from_icao(dest_icao) if isinstance(dest_icao, str) else dest_icao

        if takeoff_time.tzinfo:
            takeoff_time = takeoff_time.tz_convert("UTC").tz_localize(None)
        self.takeoff_time = takeoff_time
        self.cost_index = cost_index
        self.dollar_tonne_co2e = dollar_tonne_co2e
        self.dollar_kg_fuel = dollar_kg_fuel
        self.allow_cooling_credit = allow_cooling_credit
        self.aircraft_type = aircraft_type
        self.atyp = ps_aircraft_params.load_aircraft_engine_params()[aircraft_type]

        if dollar_tonne_co2e > 0.0:
            if met is None:
                raise ValueError("met must be provided when dollar_tonne_co2e is set")
            if "eef_per_m" not in met:
                raise ValueError("met must contain 'eef_per_m' when dollar_tonne_co2e is set")

        self.dag = _build_dag(self.origin, self.dest, dag, avoidance_regions)
        self.avoidance_regions = avoidance_regions

        self.fl_choices = cruise_flight_levels(origin_icao, dest_icao)
        self.mach_choices = _mach_choices(self.atyp)

        if met is not None:
            flight_hours = flight_hours or estimate_flight_hours(
                self.origin, self.dest, self.atyp.m_des
            )
            self.met_lookup = EdgeMetLookup.from_met(
                met=met,
                dag=self.dag,
                altitude_ft=self.fl_choices,
                takeoff_time=self.takeoff_time,
                flight_hours=flight_hours,
                spacing_m=met_spacing_m,
            )
        else:
            self.met_lookup = None

        self.result: DAGResult | None = None

    @classmethod
    def from_flight(
        cls,
        flight: Flight,
        origin_icao: str | None = None,
        dest_icao: str | None = None,
        aircraft_type: str | None = None,
        *,
        met: MetDataset | xr.Dataset | None = None,
        cost_index: float = 60.0,
        dollar_tonne_co2e: float = 0.0,
        dollar_kg_fuel: float = 1.0,
        met_spacing_m: float = 20_000.0,
        max_dist_m: float = 500_000.0,
        allow_cooling_credit: bool = False,
    ) -> Self:
        """Build a vertical-only optimizer from a ``pycontrails.Flight`` trajectory."""
        if not flight:
            raise ValueError("Flight must be non-empty")

        aircraft_type = aircraft_type or flight.get_constant("aircraft_type", None)
        if aircraft_type is None:
            raise ValueError("aircraft_type must be provided or present in flight.attrs")

        origin_icao = origin_icao or flight.get_constant("origin_airport", None)
        dest_icao = dest_icao or flight.get_constant("destination_airport", None)

        if origin_icao is None:
            origin_icao = airports.find_nearest_airport(
                airports.global_airport_database(),
                flight["longitude"][0].item(),
                flight["latitude"][0].item(),
                altitude=0.0,
                bbox=0.5,  # arbitrary
            )
            if origin_icao is None:
                raise ValueError("No airport found near flight origin")
        if dest_icao is None:
            dest_icao = airports.find_nearest_airport(
                airports.global_airport_database(),
                flight["longitude"][-1].item(),
                flight["latitude"][-1].item(),
                altitude=0.0,
                bbox=0.5,  # arbitrary
            )
            if dest_icao is None:
                raise ValueError("No airport found near flight destination")

        dag = HorizontalDAG.from_flight(flight, max_dist_m=max_dist_m)
        return cls(
            origin_icao,
            dest_icao,
            aircraft_type,
            pd.Timestamp(flight["time"][0]),
            met=met,
            dag=dag,
            cost_index=cost_index,
            dollar_tonne_co2e=dollar_tonne_co2e,
            dollar_kg_fuel=dollar_kg_fuel,
            met_spacing_m=met_spacing_m,
            allow_cooling_credit=allow_cooling_credit,
        )

    @property
    def eef_cost_factor(self) -> float:
        """Compute the kg-fuel-equivalent cost per J of effective energy forcing."""
        return self.dollar_tonne_co2e / (J_PER_TONNE_CO2 * self.dollar_kg_fuel)

    def __repr__(self) -> str:
        status = "solved" if self.result is not None else "unsolved"
        met = "with met" if self.met_lookup is not None else "no met"
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
            self.cost_index = cost_index
        if dollar_tonne_co2e is not None:
            self.dollar_tonne_co2e = dollar_tonne_co2e
        if allow_cooling_credit is not None:
            self.allow_cooling_credit = allow_cooling_credit
        if aircraft_type is not None:
            self.aircraft_type = aircraft_type
            self.atyp = ps_aircraft_params.load_aircraft_engine_params()[aircraft_type]
            self.mach_choices = _mach_choices(self.atyp)

        payload, reserve_fuel = _estimate_mass(
            payload,
            self.origin.icao_code,
            self.dest.icao_code,
            self.takeoff_time,
            self.aircraft_type,
            self.atyp,
        )
        landing_mass = self.atyp.amass_oew + payload + reserve_fuel

        fuel_estimate = _estimate_trip_fuel(self.origin, self.dest, self.atyp, landing_mass)
        amass_init = min(landing_mass + fuel_estimate, self.atyp.amass_mtow)

        for _ in range(n_iter):
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
            )
            ground_fi = len(self.fl_choices)
            amass_final = state.best_mass[self.dag.h_dest, ground_fi].item()
            if not np.isfinite(amass_final):
                raise ValueError(
                    "No feasible path found. DAG edges may be too short for the "
                    "initial climb or final descent. Try adjusting the max_dist_m "
                    "if providing a custom DAG."
                )

            trip_fuel = amass_init - amass_final
            new_amass_init = min(landing_mass + trip_fuel, self.atyp.amass_mtow)
            if abs(new_amass_init - amass_init) < 100.0:  # 100 kg convergence threshold
                break
            amass_init = new_amass_init

        self.result = DAGResult(
            state=state,
            amass_init=amass_init,
            trip_fuel=trip_fuel,
            payload=payload,
            reserve_fuel=reserve_fuel,
            landing_mass=landing_mass,
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
        state = self.result.state
        dag = self.dag
        n_fl = len(self.fl_choices)
        ground_fi = n_fl

        t_src = state.best_time[h_src, fi_src]
        t_dst = state.best_time[h_dst, fi_dst]

        edge_idx = dag.edge_index(h_src, h_dst)
        edge_dist = dag.edge_dist[edge_idx]

        # Sample range (skip source on subsequent edges to avoid duplication)
        s0 = met_lookup.edge_ptr[edge_idx]
        s1 = met_lookup.edge_ptr[edge_idx + 1]
        if skip_first:
            s0 += 1
        sample_idxs = np.arange(s0, s1)
        n_samp = len(sample_idxs)
        cum_dist = met_lookup.cum_dist[sample_idxs]

        is_first = fi_src == ground_fi
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

        # --- Met interpolation ---
        times_dt = np.datetime64(self.takeoff_time) + (elapsed * 1e9).astype("timedelta64[ns]")
        fl_arr = np.array([cruise_fi])
        interp = met_lookup(sample_idxs, times_dt[:, np.newaxis], fl_idx=fl_arr)

        eef_per_m = (
            interp.eef_per_m[:, 0]
            if interp.eef_per_m is not None
            else np.zeros(n_samp, dtype=FLOAT_DTYPE)
        )

        out = np.empty(n_samp, dtype=_WAYPOINT_DTYPE)
        out["longitude"] = met_lookup.sample_lon[sample_idxs]
        out["latitude"] = met_lookup.sample_lat[sample_idxs]
        out["altitude_ft"] = alt
        out["elapsed_s"] = elapsed
        out["mach_number"] = edge_mach
        out["eef_per_m"] = eef_per_m
        out["air_temperature"] = interp.air_temperature[:, 0]
        out["eastward_wind"] = interp.eastward_wind[:, 0]
        out["northward_wind"] = interp.northward_wind[:, 0]
        out["node_index"] = -1
        if not skip_first:
            out["node_index"][0] = h_src
        out["node_index"][-1] = h_dst
        out["sample_index"] = sample_idxs
        return out

    def to_flight(self) -> Flight:
        """Return the optimal trajectory as a `pycontrails.Flight`.

        When met data is available, waypoints are emitted at each edge sample
        point (~20 km spacing) with proper climb/descent altitude profiles and
        per-sample ``eef_per_m``. Without met, falls back to one waypoint per
        DAG node.

        The ``solve()`` method must be called first.
        """
        path_h, path_fl_idx, path_mach = self.reconstruct_path()
        state = self.result.state
        dag = self.dag
        n_fl = len(self.fl_choices)
        ground_fi = n_fl

        if self.met_lookup is None:
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

        wpts = np.concatenate(
            [
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
        )

        time = self.takeoff_time + pd.to_timedelta(wpts["elapsed_s"], unit="s")

        data: dict[str, npt.NDArray] = {
            "mach_number": wpts["mach_number"],
            "air_temperature": wpts["air_temperature"],
            "eastward_wind": wpts["eastward_wind"],
            "northward_wind": wpts["northward_wind"],
        }
        data["node_index"] = wpts["node_index"]
        data["sample_index"] = wpts["sample_index"]
        if "eef_per_m" in self.met_lookup.ds:
            data["eef_per_m"] = wpts["eef_per_m"]

        return Flight(
            longitude=wpts["longitude"],
            latitude=wpts["latitude"],
            altitude_ft=wpts["altitude_ft"],
            time=time,
            data=data,
            aircraft_type=self.aircraft_type,
        )

    def plot_met(
        self,
        altitude_ft: float | None = None,
        time: pd.Timestamp | None = None,
        ax: "GeoAxes | None" = None,
        **kwargs,
    ) -> "GeoAxes":
        """Plot met data on DAG nodes for a given flight level and time.

        Draws a wind quiver overlay. When ``eef_per_m`` is available in the
        met lookup, also draws a scatter plot colored by EEF.

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
        if self.met_lookup is None:
            raise ValueError("No met data available; pass met to Optimizer to use plot_met")

        if ax is None:
            ax = self.dag.plot()

        ds = self.met_lookup.ds

        if altitude_ft is None:
            altitude_ft = ds["altitude_ft"][0]
        if time is None:
            time = ds["time"][0]

        sel = ds.sel(altitude_ft=altitude_ft, time=time, method="nearest")

        # Get one sample index per node (first sample of each node's first outgoing edge)
        has_edges = self.dag.out_degree > 0
        node_sample_idx = self.met_lookup.edge_ptr[self.dag.adj_ptr[:-1][has_edges]]
        node_lon = self.dag.lon[has_edges]
        node_lat = self.dag.lat[has_edges]

        u = sel.eastward_wind.values[node_sample_idx]
        v = sel.northward_wind.values[node_sample_idx]

        kwargs.setdefault("alpha", 0.6)
        kwargs.setdefault("headwidth", 2)
        kwargs.setdefault("headlength", 2)
        kwargs.setdefault("headaxislength", 1.5)
        ax.quiver(
            node_lon,
            node_lat,
            u,
            v,
            transform=ax.projection,
            **kwargs,
        )

        if "eef_per_m" in ds:
            eef = sel.eef_per_m.values[node_sample_idx]
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

        fl = int(sel["altitude_ft"].item())
        t = pd.Timestamp(sel["time"].item())
        ax.set_title(f"FL{fl // 100} — {t:%Y-%m-%d %H:%M UTC}")
        return ax

    def animate_solve(self, display_fl_idx: int | None = None) -> "FuncAnimation":
        """Re-run the DP with converged mass and return a wavefront animation.

        ``solve()`` must be called first. This re-runs a single ``solve_dag``
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

        from contrailopt import slerp

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
        fig, ax = plt.subplots(figsize=(24, 12), subplot_kw={"projection": pc})
        dag.plot(ax=ax)

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

        fl_cmap = plt.colormaps["coolwarm"].resampled(n_fl)
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
