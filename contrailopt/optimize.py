"""Trajectory optimization with PS lookups on a horizontal DAG."""

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from pycontrails import Flight, MetDataset
from pycontrails.core import airports
from pycontrails.models.ps_model import ps_aircraft_params
from pycontrails.physics import geo, jet, units

from contrailopt import ps
from contrailopt.dag import AirportCoords, EdgeMetLookup, HorizontalDAG

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation

FLOAT_DTYPE = np.float32


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
    # Per-FL climb time for each sample's source: (n_sample, n_fl)
    sample_climb_s = climb_time[src_idx[sample_to_edge]]
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
) -> tuple[npt.NDArray[FLOAT_DTYPE], npt.NDArray[FLOAT_DTYPE], npt.NDArray[np.bool_]]:
    """Compute per-edge cruise fuel, time, and feasibility from met samples.

    Returns three arrays of shape ``(n_edge, n_fl, n_mach)`` containing the cruise fuel,
    time, and feasibility for each edge and FL/Mach choice.
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
    mass_3d = post_climb_mass[src_idx[sample_to_edge]][:, :, np.newaxis]
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
    return cruise_fuel, cruise_time, cruise_feasible


@dataclass(kw_only=True, slots=True, frozen=True)
class DAGState:
    """Working arrays for the wavefront dynamic program."""

    best_cost: npt.NDArray[FLOAT_DTYPE]  # (n_h, n_fl)
    best_mass: npt.NDArray[FLOAT_DTYPE]  # (n_h, n_fl) arrival mass at each state
    best_time: npt.NDArray[FLOAT_DTYPE]  # (n_h, n_fl) arrival time in seconds from takeoff
    best_mach: npt.NDArray[FLOAT_DTYPE]  # (n_h, n_fl) incoming cruise mach
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
    """Shared state passed to bootstrap and wavefront relaxation."""

    dag: HorizontalDAG
    fl_choices: npt.NDArray[FLOAT_DTYPE]
    mach_choices: npt.NDArray[FLOAT_DTYPE]
    atyp: ps_aircraft_params.PSAircraftEngineParams
    cost_index: float
    origin_elev_ft: float
    dest_elev_ft: float
    takeoff_time: pd.Timestamp
    descent: ps.DescentTable
    met_lookup: EdgeMetLookup | None


def _compute_climbs(
    fl_idxs: npt.NDArray[np.int64],
    fl_choices: npt.NDArray[FLOAT_DTYPE],
    src_masses: npt.NDArray[FLOAT_DTYPE],
    atyp: ps_aircraft_params.PSAircraftEngineParams,
    origin_elev_ft: float,
) -> tuple[
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[FLOAT_DTYPE],
    npt.NDArray[np.bool_],
    npt.NDArray[FLOAT_DTYPE],
]:
    """Compute climb dist/fuel/time from source FLs to each candidate FL.

    Handles two cases: origin-to-FL (ground source) and FL-to-FL (cruise source).

    Returns six arrays of shape ``(n_src, n_fl)``.
    """
    ground_fi = len(fl_choices)

    if (fl_idxs == ground_fi).any():
        # Origin wavefront: single ground source, needs full climb_to_target
        if len(fl_idxs) != 1:
            raise RuntimeError("Only one origin node should be active in the first wavefront")

        src_fls = np.array([origin_elev_ft], dtype=FLOAT_DTYPE)
        base_alt = fl_choices[0]
        init_dist, init_fuel, init_time, base_mass = ps.climb_to_target(
            src_masses[0],
            origin_elev_ft,
            base_alt,
            atyp,
        )
        next_dist, next_fuel, next_time, post_climb_mass, feasible = ps.compute_climb_segment(
            base_alt,
            fl_choices,
            FLOAT_DTYPE(base_mass),
            atyp,
        )

        climb_dist = (init_dist + next_dist)[np.newaxis, :]  # (1, n_fl)
        climb_fuel = (init_fuel + next_fuel)[np.newaxis, :]  # (1, n_fl)
        climb_time = (init_time + next_time)[np.newaxis, :]  # (1, n_fl)
        post_climb_mass = post_climb_mass[np.newaxis, :]  # (1, n_fl)
        feasible = feasible[np.newaxis, :]  # (1, n_fl)
    else:
        # All cruise sources: FL-to-FL climbs
        src_fls = fl_choices[fl_idxs]

        # Each has shape (n_src, n_fl)
        climb_dist, climb_fuel, climb_time, post_climb_mass, feasible = ps.compute_climb_segment(
            src_fls[:, np.newaxis],
            fl_choices[np.newaxis, :],
            src_masses[:, np.newaxis],
            atyp,
        )
        feasible = feasible | (fl_choices[np.newaxis, :] <= src_fls[:, np.newaxis])

    return climb_dist, climb_fuel, climb_time, post_climb_mass, feasible, src_fls


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

    # Climb from src_fl to each dst_fl: (n_src, n_fl)
    climb_dist, climb_fuel, climb_time, post_climb_mass, feasible, src_fls = _compute_climbs(
        fl_idxs,
        fl_choices,
        src_masses,
        ctx.atyp,
        ctx.origin_elev_ft,
    )
    climb_cost = ctx.cost_index / 60.0 * climb_time + climb_fuel

    # Expand CSR adjacency for active sources into flat edge arrays
    flat_nbr, flat_dist, src_idx, flat_edge_idx = ctx.dag.expand_neighbors(h_idxs)
    flat_dist = flat_dist.astype(FLOAT_DTYPE, copy=False)  # custom dag may have different dtype

    # Descent: step-down + ground descent if neighbor is destination
    is_dest = (flat_nbr == ctx.dag.h_dest)[:, np.newaxis]
    dest_elev = np.array([ctx.dest_elev_ft], dtype=FLOAT_DTYPE)
    final_descent_dist, final_descent_time = ctx.descent(fl_choices, dest_elev)
    dd_step, dt_step = ctx.descent(src_fls[:, np.newaxis], fl_choices[np.newaxis, :])
    descent_dd = dd_step[src_idx] + is_dest * final_descent_dist
    descent_dt = dt_step[src_idx] + is_dest * final_descent_time

    # Cruise fuel/time: (n_edge, n_fl, n_mach)
    cruise_dist = flat_dist[:, np.newaxis] - climb_dist[src_idx] - descent_dd
    if ctx.met_lookup is not None:
        cruise_fuel, cruise_time, cruise_feasible = _calculate_cruise_at_samples(
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
            climb_dist[src_idx],
            descent_dd,
            flat_dist,
        )
    else:
        cruise_fuel, cruise_time, cruise_feasible = _isa_cruise(
            fl_choices,
            mach_choices,
            post_climb_mass[src_idx],
            ctx.atyp,
            cruise_dist,
        )

    valid = (
        feasible[src_idx, :, np.newaxis] & cruise_feasible & (cruise_dist[:, :, np.newaxis] > 0.0)
    )
    cruise_cost = ctx.cost_index / 60.0 * cruise_time + cruise_fuel
    descent_cost = ctx.cost_index / 60.0 * descent_dt
    total_cost = (
        src_costs[src_idx, np.newaxis, np.newaxis]
        + climb_cost[src_idx, :, np.newaxis]
        + descent_cost[:, :, np.newaxis]
        + cruise_cost
    )
    total_cost = np.where(valid, total_cost, np.inf)

    # Best mach per (edge, FL)
    best_mach_idx = np.argmin(total_cost, axis=2, keepdims=True)
    best_total = np.take_along_axis(total_cost, best_mach_idx, axis=2).squeeze(2)
    best_mach = mach_choices[best_mach_idx.squeeze(2)]
    best_cruise_fuel = np.take_along_axis(cruise_fuel, best_mach_idx, axis=2).squeeze(2)
    best_cruise_time = np.take_along_axis(cruise_time, best_mach_idx, axis=2).squeeze(2)
    arrival_mass = post_climb_mass[src_idx] - best_cruise_fuel
    arrival_time = src_elapsed[src_idx, None] + climb_time[src_idx] + best_cruise_time + descent_dt

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
    state.best_prev_h[flat_nbr[wi], wj] = h_idxs[src_idx[wi]]
    state.best_prev_fi[flat_nbr[wi], wj] = fl_idxs[src_idx[wi]]


def solve_dag(
    dag: HorizontalDAG,
    amass_init: float,
    fl_choices: npt.NDArray[FLOAT_DTYPE],
    mach_choices: npt.NDArray[FLOAT_DTYPE],
    atyp: ps_aircraft_params.PSAircraftEngineParams,
    cost_index: float,
    origin_elev_ft: float,
    dest_elev_ft: float,
    takeoff_time: pd.Timestamp,
    met_lookup: EdgeMetLookup | None = None,
    on_wavefront: Callable[[npt.NDArray[np.int64], DAGState], None] | None = None,
) -> DAGState:
    """Solve shortest-path DP on the topo-sorted DAG, tracking mass exactly."""
    ctx = _SolverCtx(
        dag=dag,
        fl_choices=fl_choices.astype(FLOAT_DTYPE, copy=False),
        mach_choices=mach_choices.astype(FLOAT_DTYPE, copy=False),
        atyp=atyp,
        cost_index=cost_index,
        origin_elev_ft=origin_elev_ft,
        dest_elev_ft=dest_elev_ft,
        takeoff_time=takeoff_time,
        descent=ps.DescentTable(atyp),
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
    best_dest_fi = int(np.argmin(state.best_cost[dag.h_dest, :n_fl]))
    if np.isfinite(state.best_cost[dag.h_dest, best_dest_fi]):
        for arr in (
            state.best_cost,
            state.best_mass,
            state.best_time,
            state.best_mach,
            state.best_prev_h,
            state.best_prev_fi,
        ):
            arr[dag.h_dest, ground_fi] = arr[dag.h_dest, best_dest_fi]

    return state


def _estimate_flight_hours(
    origin: AirportCoords,
    dest: AirportCoords,
    atyp: ps_aircraft_params.PSAircraftEngineParams,
) -> int:
    """Estimate upper-bound flight duration in hours for met time window."""
    dist = geo.haversine(*origin.coords, *dest.coords)
    min_mach = atyp.m_des - 0.03
    T_cold = units.m_to_T_isa(units.ft_to_m(40_000.0))
    min_tas = units.mach_number_to_tas(min_mach, T_cold)
    max_headwind = 70.0  # m/s, strong jet stream
    slow_gs = min_tas - max_headwind
    return int(np.ceil(dist / slow_gs / 3600.0))


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
        dag = HorizontalDAG.from_poisson(*origin.coords, *dest.coords, dtype=FLOAT_DTYPE).prune()

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
    origin_icao : str
        ICAO code for the origin airport (e.g. ``"KLAX"``).
    dest_icao : str
        ICAO code for the destination airport.
    aircraft_type : str
        Aircraft type key in the PS model parameter table (e.g. ``"A320"``).
    takeoff_time : pd.Timestamp
        Departure time, used for met interpolation.
    met : MetDataset or None, default None
        Gridded met data with ``air_temperature``, ``eastward_wind``, and ``northward_wind``.
        If *None*, cruise performance uses ISA temperatures and zero wind.
    dag : HorizontalDAG or None, default None
        Pre-built DAG. If *None*, a DAG is generated via Poisson-disk sampling along the
        great circle. The DAG origin and destination must agree with the airport coordinates.
        The DAG is expected to be pruned (``HorizontalDAG.prune()``) but this is not enforced.
    cost_index : float, default 60.0
        Fuel-vs-time tradeoff in kg per minute. Higher values penalize time more,
        favoring faster (and more fuel-intensive) routes.
    met_spacing_m : float, default 20_000.0
        Spacing in meters between met sample points along each edge.
    avoidance_regions : list of polygon coordinate lists, or None
        Polygons to exclude from the search, defined as lists of ``(lon, lat)`` vertices.
        Edges intersecting any polygon are removed and the DAG is re-pruned.
    **interp_kwargs
        Additional keyword arguments passed to :meth:`xarray.Dataset.interp`
        during met interpolation (e.g. ``kwargs={"fill_value": None}``).
    """

    def __init__(
        self,
        origin_icao: str,
        dest_icao: str,
        aircraft_type: str,
        takeoff_time: pd.Timestamp,
        *,
        met: MetDataset | None = None,
        dag: HorizontalDAG | None = None,
        cost_index: float = 60.0,
        met_spacing_m: float = 20_000.0,
        avoidance_regions: list[list[tuple[float, float]]] | None = None,
        **interp_kwargs: Any,
    ) -> None:
        self.origin = AirportCoords.from_icao(origin_icao)
        self.dest = AirportCoords.from_icao(dest_icao)

        if takeoff_time.tzinfo:
            takeoff_time = takeoff_time.tz_convert("UTC").tz_localize(None)
        self.takeoff_time = takeoff_time
        self.cost_index = cost_index
        self.atyp = ps_aircraft_params.load_aircraft_engine_params()[aircraft_type]

        self.dag = _build_dag(self.origin, self.dest, dag, avoidance_regions)
        self.avoidance_regions = avoidance_regions

        self.fl_choices = cruise_flight_levels(origin_icao, dest_icao)
        self.mach_choices = np.arange(
            self.atyp.m_des // 0.01 * 0.01 - 0.02,  # floor to 2 decimal places, minus a margin
            self.atyp.max_mach_num + 0.01,
            0.01,
            dtype=FLOAT_DTYPE,
        )

        if met is not None:
            flight_hours = _estimate_flight_hours(self.origin, self.dest, self.atyp)
            self.met_lookup = EdgeMetLookup.from_met(
                met=met,
                dag=self.dag,
                altitude_ft=self.fl_choices,
                takeoff_time=self.takeoff_time,
                flight_hours=flight_hours,
                spacing_m=met_spacing_m,
                **interp_kwargs,
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
        met: MetDataset | None = None,
        cost_index: float = 60.0,
        met_spacing_m: float = 20_000.0,
        max_dist_m: float = 500_000.0,
        **interp_kwargs: Any,
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
            met_spacing_m=met_spacing_m,
            **interp_kwargs,
        )

    def __repr__(self) -> str:
        status = "solved" if self.result is not None else "unsolved"
        met = "with met" if self.met_lookup is not None else "no met"
        name = type(self).__name__
        return (
            f"{name}({self.origin.icao_code} -> {self.dest.icao_code}, "
            f"{self.atyp.aircraft_type}, {self.takeoff_time}, "
            f"{self.dag.n_nodes} nodes, {met}, {status})"
        )

    def solve(
        self,
        n_iter: int = 3,
        cost_index: float | None = None,
        payload: float | None = None,
    ) -> DAGResult:
        """Solve the trajectory optimization via shortest-path dynamic programming on the DAG.

        This method iteratively re-solves the DAG to converge on takeoff mass.

        - Guess initial takeoff mass at 80% of the OEW-to-MTOW range
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
        payload : float or None, default None
            Aircraft payload in kg if known. If None, this is estimated with pycontrails.

        Returns
        -------
        DAGResult
            DP state, takeoff mass, trip fuel, payload, reserve fuel, and landing mass.
        """
        if cost_index is not None:
            self.cost_index = cost_index

        payload, reserve_fuel = _estimate_mass(
            payload,
            self.origin.icao_code,
            self.dest.icao_code,
            self.takeoff_time,
            self.atyp.aircraft_type,
            self.atyp,
        )
        landing_mass = self.atyp.amass_oew + payload + reserve_fuel
        amass_init = self.atyp.amass_oew + 0.8 * (self.atyp.amass_mtow - self.atyp.amass_oew)

        for _ in range(n_iter):
            state = solve_dag(
                self.dag,
                amass_init,
                self.fl_choices,
                self.mach_choices,
                self.atyp,
                self.cost_index,
                self.origin.elevation_ft,
                self.dest.elevation_ft,
                self.takeoff_time,
                self.met_lookup,
            )
            ground_fi = len(self.fl_choices)
            trip_fuel = amass_init - state.best_mass[self.dag.h_dest, ground_fi].item()
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
        waypoints; endpoints use a sentinel index (``len(fl_choices)``)
        representing ground level.

        Returns
        -------
        path_h : npt.NDArray[np.int64]
            Horizontal node indices along the path.
            The first and last entries are origin and destination.
        path_fl_idx : npt.NDArray[np.int64]
            Flight level index at each node. Endpoints use a special convention
            ``ground_fl_idx = len(fl_choices)``, interior nodes index into ``fl_choices``.
        path_mach : npt.NDArray[FLOAT_DTYPE]
            Cruise Mach number on each incoming leg. The first entry ``path_mach[0]`` is NaN
            (no incoming leg at the origin).
        """
        if self.result is None:
            raise ValueError("Call solve() first")

        state = self.result.state
        dag = self.dag
        ground_fi = len(self.fl_choices)

        if not np.isfinite(state.best_cost[dag.h_dest, ground_fi]):
            raise ValueError("No feasible path to destination")

        path_h, path_fl_idx, path_mach = [], [], []
        h, fi = dag.h_dest, ground_fi

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
            dag,
            self.result.amass_init,
            self.fl_choices,
            self.mach_choices,
            self.atyp,
            self.cost_index,
            self.origin.elevation_ft,
            self.dest.elevation_ft,
            self.takeoff_time,
            self.met_lookup,
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
