"""Support for aircraft performance modeling with the Poll-Schumann model chain."""

import numpy as np
import numpy.typing as npt
from pycontrails import JetA
from pycontrails.models.ps_model import ps_aircraft_params, ps_model, ps_operational_limits
from pycontrails.physics import units

ENGINE_DETERIORATION_FACTOR = 0.025
THRUST_FRACTION = 0.9
ROCD_CLIMB_THRESHOLD = 500.0
MAX_THRUST_BUFFER = 0.0


def mach_schedule(
    alt_ft: npt.NDArray[np.floating],
    atyp: ps_aircraft_params.PSAircraftEngineParams,
) -> npt.NDArray[np.floating]:
    """Compute the speed-limited Mach number as a function of altitude.

    Returns ``min(m_des, mach_limit)`` at each altitude, applying ATM speed limits
    below 10,000 ft. This is the single source of truth for the climb/descent speed
    schedule used throughout the optimizer.
    """
    air_pressure = units.ft_to_pl(alt_ft) * 100.0
    mach_lim = ps_operational_limits.max_mach_number_by_altitude(
        alt_ft,
        air_pressure,
        atyp.max_mach_num,
        np.float32(atyp.p_i_max),
        atyp.p_inf_co,
        atm_speed_limit=True,
        buffer=0.0,
    )
    return np.minimum(atyp.m_des, mach_lim)


def cruise_performance(
    alt_ft: npt.NDArray[np.floating],
    mach: npt.NDArray[np.floating],
    mass: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating],
    atyp: ps_aircraft_params.PSAircraftEngineParams,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool_]]:
    """Run PS model chain for cruise at a given Mach.

    Return a tuple of:
    - fuel flow in kg/s
    - a boolean feasibility mask

    Infeasible conditions are those where the required thrust exceeds the maximum continuous thrust.
    """
    air_pressure = units.ft_to_pl(alt_ft) * 100.0
    angle = np.float32(0.0)  # avoid needless promotion to float64 in ps_model.lift_coefficient

    rn = ps_model.reynolds_number(atyp.wing_surface_area, mach, air_temperature, air_pressure)
    c_f = ps_model.skin_friction_coefficient(rn)
    c_lift = ps_model.lift_coefficient(atyp.wing_surface_area, mass, air_pressure, mach, angle)
    c_drag_0 = ps_model.zero_lift_drag_coefficient(c_f, atyp.psi_0)
    e_ls = ps_model.oswald_efficiency_factor(c_drag_0, atyp)
    c_drag_w = ps_model.wave_drag_coefficient(mach, c_lift, atyp)
    c_drag = ps_model.airframe_drag_coefficient(
        c_drag_0, c_drag_w, c_lift, e_ls, atyp.wing_aspect_ratio
    )

    f_thrust = ps_model.thrust_force(mass, c_lift, c_drag, 0.0, angle)
    c_t = ps_model.engine_thrust_coefficient(f_thrust, mach, air_pressure, atyp.wing_surface_area)
    c_t_eta_b = ps_model.thrust_coefficient_at_max_efficiency(mach, atyp.m_des, atyp.c_t_des)
    c_t_max = ps_operational_limits.max_available_thrust_coefficient(
        air_temperature, mach, c_t_eta_b, atyp, buffer=MAX_THRUST_BUFFER
    )
    feasible = c_t <= c_t_max

    eta = ps_model.overall_propulsion_efficiency(
        mach,
        c_t,
        c_t_eta_b,
        atyp,
        engine_deterioration_factor=ENGINE_DETERIORATION_FACTOR,
    )
    fuel_flow = ps_model.fuel_mass_flow_rate(
        air_pressure,
        air_temperature,
        mach,
        c_t,
        eta,
        atyp.wing_surface_area,
        q_fuel=JetA.q_fuel,
    )
    return fuel_flow, feasible


def climb_performance(
    alt_ft: npt.NDArray[np.floating],
    mass: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating],
    atyp: ps_aircraft_params.PSAircraftEngineParams,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.bool_],
]:
    """Evaluate instantaneous climb performance at a single point.

    Uses a fixed fraction of maximum continuous thrust at the design Mach number, which is
    only valid at cruise altitudes where the design Mach does not exceed the operational speed
    limit. For integrated climb over an altitude range, use compute_climb_segment. For full
    climb integration from the ground with a realistic speed schedule, use climb_to_target.

    Return a tuple of:
    - fuel flow in kg/s
    - ROCD in ft/min
    - TAS in m/s
    - a boolean feasibility mask

    Infeasible conditions are those where ROCD is below the threshold.
    """
    mach = atyp.m_des
    air_pressure = units.ft_to_pl(alt_ft) * 100.0

    rn = ps_model.reynolds_number(atyp.wing_surface_area, mach, air_temperature, air_pressure)
    c_f = ps_model.skin_friction_coefficient(rn)
    # 0 degree climb angle: step-climbs at cruise are very mild
    c_lift = ps_model.lift_coefficient(atyp.wing_surface_area, mass, air_pressure, mach, 0.0)
    c_drag_0 = ps_model.zero_lift_drag_coefficient(c_f, atyp.psi_0)
    e_ls = ps_model.oswald_efficiency_factor(c_drag_0, atyp)
    c_drag_w = ps_model.wave_drag_coefficient(mach, c_lift, atyp)
    c_drag = ps_model.airframe_drag_coefficient(
        c_drag_0, c_drag_w, c_lift, e_ls, atyp.wing_aspect_ratio
    )

    c_t_eta_b = ps_model.thrust_coefficient_at_max_efficiency(mach, atyp.m_des, atyp.c_t_des)
    c_t_max = ps_operational_limits.max_available_thrust_coefficient(
        air_temperature, mach, c_t_eta_b, atyp, buffer=MAX_THRUST_BUFFER
    )
    c_t_climb = THRUST_FRACTION * c_t_max

    tas = units.mach_number_to_tas(mach, air_temperature)
    dh_dt = tas * (c_t_climb - c_drag) / c_lift  # can be negative if too heavy, but gets rocd filt
    rocd = units.m_to_ft(dh_dt) * 60.0

    feasible = rocd > ROCD_CLIMB_THRESHOLD

    eta = ps_model.overall_propulsion_efficiency(
        mach,
        c_t_climb,
        c_t_eta_b,
        atyp,
        engine_deterioration_factor=ENGINE_DETERIORATION_FACTOR,
    )
    fuel_flow = ps_model.fuel_mass_flow_rate(
        air_pressure,
        air_temperature,
        mach,
        c_t_climb,
        eta,
        atyp.wing_surface_area,
        q_fuel=JetA.q_fuel,
    )

    return fuel_flow, rocd, tas, feasible


def compute_climb_segment(
    src_alt_ft: npt.NDArray[np.floating],
    dst_alt_ft: npt.NDArray[np.floating],
    src_mass: npt.NDArray[np.floating],
    atyp: ps_aircraft_params.PSAircraftEngineParams,
    delta_isa: npt.NDArray[np.floating] | float,
    tailwind: npt.NDArray[np.floating] | float,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.bool_],
]:
    """Integrate climb from ``src_alt_ft`` to ``dst_alt_ft`` in 2000 ft steps using a midpoint rule.

    Uses climb_performance at each step, accumulating distance, fuel, and time while
    updating mass for fuel burn. Climb uses a fixed fraction of maximum continuous thrust
    at the design Mach number.

    Set ``delta_isa`` and ``tailwind`` to zero for a weather-agnostic climb estimate.

    Return a tuple of:
    - climb distance in m
    - climb fuel in kg
    - climb time in s
    - mass after climb in kg
    - a boolean feasibility mask (True where the target altitude was reached)

    Values for infeasible entries (where feasible is False) are corrupt and should not be used.
    """
    src_alt_ft, dst_alt_ft, src_mass, delta_isa, tailwind = np.broadcast_arrays(
        src_alt_ft, dst_alt_ft, src_mass, delta_isa, tailwind
    )

    alt_ft = src_alt_ft.copy()
    mass = src_mass.copy()

    total_dist = np.zeros_like(dst_alt_ft)  # m
    total_fuel = np.zeros_like(dst_alt_ft)  # kg
    total_time = np.zeros_like(dst_alt_ft)  # s
    feasible = np.ones_like(dst_alt_ft, dtype=bool)

    while True:
        active = (alt_ft < dst_alt_ft) & feasible
        if not np.any(active):
            break

        step = np.minimum(2000.0, dst_alt_ft[active] - alt_ft[active])
        mid_alt_ft = alt_ft[active] + step / 2.0

        air_temperature = units.m_to_T_isa(units.ft_to_m(mid_alt_ft)) + delta_isa[active]

        # Values where step_feasible is False do not make sense (negative ROCD, etc.)
        ff, rocd, tas, feas = climb_performance(mid_alt_ft, mass[active], air_temperature, atyp)

        feasible[active] &= feas

        dt_s = step / rocd * 60.0
        total_dist[active] += (tas + tailwind[active]) * dt_s
        total_fuel[active] += ff * dt_s
        total_time[active] += dt_s
        mass[active] -= ff * dt_s
        alt_ft[active] += step

    return total_dist, total_fuel, total_time, mass, feasible


def climb_to_target(
    mass: float,
    ground_alt_ft: float,
    target_alt_ft: float,
    atyp: ps_aircraft_params.PSAircraftEngineParams,
) -> tuple[float, float, float, float]:
    """Integrate climb from ground to target altitude in 1000 ft steps.

    This function always uses the ISA temperature profile and no wind.

    Unlike climb_performance and compute_climb_segment, this function applies a realistic
    IAS/Mach speed schedule with ATM speed limits below 10,000 ft.

    Uses a fixed fraction of maximum continuous thrust and a climb angle of 3 degrees.

    Return a tuple of:
    - climb distance in m
    - climb fuel in kg
    - climb time in s
    - mass after climb in kg

    Raises ValueError if ROCD drops below the threshold at any step.
    """
    # Build altitude bands and use the mid-point of each band for calculations.
    band_edges = np.arange(ground_alt_ft, target_alt_ft + 1000.0, 1000.0)
    band_edges[-1] = target_alt_ft  # last step may be < 1000 ft
    steps = np.diff(band_edges)
    mid_alt_ft = band_edges[:-1] + steps / 2.0
    n = len(steps)

    # Vectorize everything independent of the mass loop
    air_pressure = units.ft_to_pl(mid_alt_ft) * 100.0
    T_isa = units.m_to_T_isa(units.ft_to_m(mid_alt_ft))
    mach = mach_schedule(mid_alt_ft, atyp)
    tas = units.mach_number_to_tas(mach, T_isa)

    rn = ps_model.reynolds_number(atyp.wing_surface_area, mach, T_isa, air_pressure)
    c_f = ps_model.skin_friction_coefficient(rn)
    c_drag_0 = ps_model.zero_lift_drag_coefficient(c_f, atyp.psi_0)
    e_ls = ps_model.oswald_efficiency_factor(c_drag_0, atyp)
    c_t_eta_b = ps_model.thrust_coefficient_at_max_efficiency(mach, atyp.m_des, atyp.c_t_des)
    c_t_max = ps_operational_limits.max_available_thrust_coefficient(
        T_isa,
        mach,
        c_t_eta_b,
        atyp,
        buffer=MAX_THRUST_BUFFER,
    )
    c_t_climb = THRUST_FRACTION * c_t_max

    eta = ps_model.overall_propulsion_efficiency(
        mach,
        c_t_climb,
        c_t_eta_b,
        atyp,
        engine_deterioration_factor=ENGINE_DETERIORATION_FACTOR,
    )
    ff = ps_model.fuel_mass_flow_rate(
        air_pressure,
        T_isa,
        mach,
        c_t_climb,
        eta,
        atyp.wing_surface_area,
        q_fuel=JetA.q_fuel,
    )

    # Loop only for mass-dependent quantities: lift, drag, ROCD
    climb_angle = 3.0
    total_dist = 0.0
    total_fuel = 0.0
    total_time = 0.0

    for i in range(n):
        c_lift = ps_model.lift_coefficient(
            atyp.wing_surface_area,
            mass,
            air_pressure[i],
            mach[i],
            climb_angle,
        )
        c_drag_w = ps_model.wave_drag_coefficient(mach[i], c_lift, atyp)
        c_drag = ps_model.airframe_drag_coefficient(
            c_drag_0[i],
            c_drag_w,
            c_lift,
            e_ls[i],
            atyp.wing_aspect_ratio,
        )

        dh_dt = tas[i] * (c_t_climb[i] - c_drag) / c_lift
        rocd = units.m_to_ft(dh_dt) * 60.0
        if rocd < ROCD_CLIMB_THRESHOLD:
            raise ValueError(f"ROCD {rocd:.0f} ft/min below threshold at {mid_alt_ft[i]:.0f} ft")

        dt_s = (steps[i] / rocd) * 60.0
        total_dist += tas[i] * dt_s
        total_fuel += ff[i] * dt_s
        total_time += dt_s
        mass -= ff[i] * dt_s

    return float(total_dist), float(total_fuel), float(total_time), float(mass)


def descent_performance(
    alt_ft: npt.NDArray[np.floating],
    mass: npt.NDArray[np.floating],
    air_temperature: npt.NDArray[np.floating],
    atyp: ps_aircraft_params.PSAircraftEngineParams,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """Evaluate instantaneous 3-degree descent performance at a single point.

    https://en.wikipedia.org/wiki/Rule_of_three_(aeronautics)

    The descent angle and speed schedule are fixed. Fuel flow comes from the PS model chain
    with idle fuel flow as a floor. This is the counterpart of :func:`climb_performance`.

    Return a tuple of:
    - fuel flow in kg/s
    - ROCD in ft/min, negative
    - TAS in m/s
    """
    descent_angle = np.float32(-3.0)
    air_pressure = units.ft_to_pl(alt_ft) * 100.0

    mach = mach_schedule(alt_ft, atyp)
    tas = units.mach_number_to_tas(mach, air_temperature)

    rn = ps_model.reynolds_number(atyp.wing_surface_area, mach, air_temperature, air_pressure)
    c_f = ps_model.skin_friction_coefficient(rn)
    c_lift = ps_model.lift_coefficient(
        atyp.wing_surface_area, mass, air_pressure, mach, descent_angle
    )
    c_drag_0 = ps_model.zero_lift_drag_coefficient(c_f, atyp.psi_0)
    e_ls = ps_model.oswald_efficiency_factor(c_drag_0, atyp)
    c_drag_w = ps_model.wave_drag_coefficient(mach, c_lift, atyp)
    c_drag = ps_model.airframe_drag_coefficient(
        c_drag_0, c_drag_w, c_lift, e_ls, atyp.wing_aspect_ratio
    )

    # When the aircraft is heavy, f_thrust can be 0 at low altitudes, which makes c_t 0 and eta
    # and ff nan. ff is clipped to the idle floor below so it does not matter, but clip the
    # thrust to 1 to keep numpy from warning about dividing by zero.
    f_thrust = np.maximum(ps_model.thrust_force(mass, c_lift, c_drag, 0.0, descent_angle), 1.0)

    c_t = ps_model.engine_thrust_coefficient(f_thrust, mach, air_pressure, atyp.wing_surface_area)
    c_t_eta_b = ps_model.thrust_coefficient_at_max_efficiency(mach, atyp.m_des, atyp.c_t_des)
    eta = ps_model.overall_propulsion_efficiency(
        mach, c_t, c_t_eta_b, atyp, engine_deterioration_factor=ENGINE_DETERIORATION_FACTOR
    )
    ff = ps_model.fuel_mass_flow_rate(
        air_pressure, air_temperature, mach, c_t, eta, atyp.wing_surface_area, q_fuel=JetA.q_fuel
    )
    ff = np.maximum(ff, ps_operational_limits.fuel_flow_idle(atyp.ff_idle_sls, alt_ft))

    rocd = units.m_to_ft(tas * np.tan(np.deg2rad(descent_angle))) * 60.0
    return ff, rocd, tas


def final_descent(
    src_alt_ft: npt.NDArray[np.floating],
    ground_alt_ft: float,
    atyp: ps_aircraft_params.PSAircraftEngineParams,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute a 3 degree descent from cruise FL to the ground.

    https://en.wikipedia.org/wiki/Rule_of_three_(aeronautics)

    Fixes the descent angle and speed schedule, then solves for the required
    thrust from the force balance at each 1000 ft altitude band. Fuel flow
    is computed through the PS model chain, with idle fuel flow as a floor.

    Distance and time are mass-independent (fixed geometry and speed schedule).
    Fuel has weak but nonzero sensitivity to mass: heavier aircraft need less
    thrust (gravity does more work), so descent fuel *decreases* with mass.
    But the sensitivity is small enough that we can just use the max landing weight for now.

    Parameters
    ----------
    src_alt_ft : npt.NDArray[np.floating]
        Starting altitude(s) in feet, e.g. one per candidate flight level.
    ground_alt_ft : float
        Ground altitude at the destination in feet.
    atyp : ps_aircraft_params.PSAircraftEngineParams
        Aircraft/engine parameters.

    Returns
    -------
    descent_dist : npt.NDArray[np.floating]
        Horizontal distance consumed by the descent, in meters. Same shape as ``src_alt_ft``.
    descent_fuel : npt.NDArray[np.floating]
        Fuel burned during the descent, in kg.
    descent_time : npt.NDArray[np.floating]
        Time for the descent, in seconds.
    """
    max_alt = 45_000.0
    mass = atyp.amass_mlw
    descent_angle = np.float32(-3.0)  # constant descent angle of 3 degrees

    band_start = ground_alt_ft // 1000.0 * 1000.0
    band_edges = np.arange(band_start, max_alt + 1000.0, 1000.0, dtype=src_alt_ft.dtype)
    band_mids = band_edges[:-1] + 500.0

    T_isa = units.m_to_T_isa(units.ft_to_m(band_mids))
    tas = units.mach_number_to_tas(mach_schedule(band_mids, atyp), T_isa)

    # Geometry of the bands
    band_dist = units.ft_to_m(1000.0) / np.tan(np.deg2rad(-descent_angle))
    band_time = band_dist / tas

    ff, _, _ = descent_performance(band_mids, np.full_like(band_mids, mass), T_isa, atyp)
    band_fuel = ff * band_time

    # Cumulative sums from ground up: band_edges[k] = k * 1000 ft
    cum_dist = np.zeros_like(band_edges)
    cum_time = np.zeros_like(band_edges)
    cum_fuel = np.zeros_like(band_edges)
    cum_dist[1:] = np.arange(1, len(band_edges), dtype=src_alt_ft.dtype) * band_dist
    np.cumsum(band_time, out=cum_time[1:])
    np.cumsum(band_fuel, out=cum_fuel[1:])

    # Interpolate at source and ground altitudes, take the difference
    descent_dist = np.interp(src_alt_ft, band_edges, cum_dist).astype(src_alt_ft.dtype) - np.interp(
        ground_alt_ft, band_edges, cum_dist
    ).astype(src_alt_ft.dtype)
    descent_fuel = np.interp(src_alt_ft, band_edges, cum_fuel).astype(src_alt_ft.dtype) - np.interp(
        ground_alt_ft, band_edges, cum_fuel
    ).astype(src_alt_ft.dtype)
    descent_time = np.interp(src_alt_ft, band_edges, cum_time).astype(src_alt_ft.dtype) - np.interp(
        ground_alt_ft, band_edges, cum_time
    ).astype(src_alt_ft.dtype)

    return descent_dist, descent_fuel, descent_time
