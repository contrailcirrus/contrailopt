"""Test the ps module."""

import numpy as np
import pytest
from pycontrails.models.ps_model import ps_aircraft_params
from pycontrails.models.ps_model.ps_aircraft_params import PSAircraftEngineParams as PSParams
from pycontrails.physics import units

from contrailopt.ps import (
    climb_performance,
    climb_to_target,
    compute_climb_segment,
    cruise_performance,
    final_descent,
)


@pytest.fixture
def atyp() -> PSParams:
    return ps_aircraft_params.load_aircraft_engine_params()["A320"]


class TestCruisePerformance:
    def test_fuel_flow_reasonable(self, atyp: PSParams) -> None:
        alt_ft = np.array([35000.0])
        mach = np.array([atyp.m_des])
        mass = np.array([65000.0])
        air_temperature = units.m_to_T_isa(units.ft_to_m(alt_ft))

        fuel_flow, feasible = cruise_performance(alt_ft, mach, mass, air_temperature, atyp)

        assert fuel_flow.shape == (1,)
        assert feasible.shape == (1,)
        assert fuel_flow[0] == pytest.approx(0.64, abs=0.01)
        assert feasible[0]

    def test_heavier_aircraft_burns_more_fuel(self, atyp: PSParams) -> None:
        alt_ft = np.array([35000.0, 35000.0])
        mach = np.array([atyp.m_des, atyp.m_des])
        mass = np.array([55000.0, 75000.0])
        air_temperature = units.m_to_T_isa(units.ft_to_m(alt_ft))

        fuel_flow, feasible = cruise_performance(alt_ft, mach, mass, air_temperature, atyp)
        assert np.all(feasible)
        assert fuel_flow[1] > fuel_flow[0]

    def test_higher_mach_burns_more_fuel(self, atyp: PSParams) -> None:
        alt_ft = np.array([35000.0, 35000.0])
        mach = np.array([atyp.m_des - 0.02, atyp.m_des + 0.02])
        mass = np.array([65000.0, 65000.0])
        air_temperature = units.m_to_T_isa(units.ft_to_m(alt_ft))

        fuel_flow, feasible = cruise_performance(alt_ft, mach, mass, air_temperature, atyp)
        assert np.all(feasible)
        assert fuel_flow[1] > fuel_flow[0]

    def test_colder_air_burns_less_fuel(self, atyp: PSParams) -> None:
        alt_ft = np.array([35000.0, 35000.0])
        mach = np.array([atyp.m_des, atyp.m_des])
        mass = np.array([65000.0, 65000.0])
        air_temperature = units.m_to_T_isa(units.ft_to_m(alt_ft)) + np.array([0.0, -20.0])

        fuel_flow, feasible = cruise_performance(alt_ft, mach, mass, air_temperature, atyp)
        assert np.all(feasible)
        assert fuel_flow[1] < fuel_flow[0]

    def test_feasibility_ceiling_at_mtow(self, atyp: PSParams) -> None:
        alt_ft = np.arange(30000.0, 45000.0, 1000.0)
        mach = np.full_like(alt_ft, atyp.m_des)
        mass = np.full_like(alt_ft, atyp.amass_mtow)
        air_temperature = units.m_to_T_isa(units.ft_to_m(alt_ft))

        _, feasible = cruise_performance(alt_ft, mach, mass, air_temperature, atyp)
        assert np.all(feasible[:10])  # FL300 through FL390
        assert not np.any(feasible[10:])  # at FL400 and above not feasible to cruise at MTOW


class TestClimbPerformance:
    def test_rocd_and_fuel_flow_reasonable(self, atyp: PSParams) -> None:
        alt_ft = np.array([30000.0])
        mass = np.array([65000.0])
        air_temperature = units.m_to_T_isa(units.ft_to_m(alt_ft))

        fuel_flow, rocd, tas, feasible = climb_performance(alt_ft, mass, air_temperature, atyp)
        assert feasible[0]
        assert rocd[0] == pytest.approx(1085.4, abs=1.0)
        assert fuel_flow[0] == pytest.approx(0.954, abs=0.01)
        assert tas[0] == pytest.approx(228.2, abs=1.0)

    def test_heavier_aircraft_has_lower_rocd(self, atyp: PSParams) -> None:
        alt_ft = np.array([30000.0, 30000.0])
        mass = np.array([55000.0, 75000.0])
        air_temperature = units.m_to_T_isa(units.ft_to_m(alt_ft))

        fuel_flow, rocd, _, feasible = climb_performance(alt_ft, mass, air_temperature, atyp)
        assert np.all(feasible)
        assert rocd[0] > rocd[1]
        # Fuel flow is mass-independent: thrust is a fixed fraction of c_t_max,
        # which depends only on altitude/temperature, not mass.
        assert fuel_flow[0] == fuel_flow[1]

    def test_rocd_decreases_with_altitude(self, atyp: PSParams) -> None:
        alt_ft = np.array([25000.0, 35000.0])
        mass = np.array([65000.0, 65000.0])
        air_temperature = units.m_to_T_isa(units.ft_to_m(alt_ft))

        fuel_flow, rocd, _, feasible = climb_performance(alt_ft, mass, air_temperature, atyp)
        assert np.all(feasible)
        assert rocd[0] > rocd[1]
        # Fuel flow decreases with altitude: thinner air means lower pressure,
        # which reduces fuel mass flow rate even at the same thrust coefficient.
        assert fuel_flow[0] > fuel_flow[1]

    def test_feasibility_ceiling_at_mtow(self, atyp: PSParams) -> None:
        alt_ft = np.arange(27000.0, 45000.0, 1000.0)
        mass = np.full_like(alt_ft, atyp.amass_mtow)
        air_temperature = units.m_to_T_isa(units.ft_to_m(alt_ft))

        _, _, _, feasible = climb_performance(alt_ft, mass, air_temperature, atyp)
        assert np.all(feasible[:9])  # FL270 through FL350
        assert not np.any(feasible[9:])  # at FL360 and above not feasible to climb at MTOW


class TestComputeClimbSegment:
    def test_no_climb_needed(self, atyp: PSParams) -> None:
        src_alt_ft = dst_alt_ft = np.array([35000.0])
        src_mass = np.array([70000.0])

        dist, fuel, time, mass, feasible = compute_climb_segment(
            src_alt_ft, dst_alt_ft, src_mass, atyp, delta_isa=0.0, tailwind=0.0
        )
        assert feasible[0]
        assert dist[0] == 0.0
        assert fuel[0] == 0.0
        assert time[0] == 0.0
        assert mass[0] == 70000.0

    def test_fuel_and_distance_reasonable(self, atyp: PSParams) -> None:
        src_alt_ft = np.array([30000.0])
        dst_alt_ft = np.array([35000.0])
        src_mass = np.array([65000.0])

        dist, fuel, time, mass, feasible = compute_climb_segment(
            src_alt_ft, dst_alt_ft, src_mass, atyp, delta_isa=0.0, tailwind=0.0
        )
        assert feasible[0]
        assert dist[0] == pytest.approx(75916, abs=100)
        assert fuel[0] == pytest.approx(297.5, abs=1.0)
        assert time[0] == pytest.approx(336.6, abs=1.0)
        assert mass[0] == pytest.approx(src_mass[0] - fuel[0], rel=1e-6)

    def test_larger_climb_uses_more_fuel(self, atyp: PSParams) -> None:
        src_alt_ft = np.array([33000.0, 30000.0])
        dst_alt_ft = np.array([35000.0, 35000.0])
        src_mass = np.array([65000.0, 65000.0])

        dist, fuel, time, mass, feasible = compute_climb_segment(
            src_alt_ft, dst_alt_ft, src_mass, atyp, delta_isa=0.0, tailwind=0.0
        )
        assert np.all(feasible)
        assert fuel[1] > fuel[0]
        assert dist[1] > dist[0]
        assert time[1] > time[0]
        assert mass[1] < mass[0]

    def test_heavier_aircraft_uses_more_fuel(self, atyp: PSParams) -> None:
        src_alt_ft = np.array([30000.0, 30000.0])
        dst_alt_ft = np.array([35000.0, 35000.0])
        src_mass = np.array([65000.0, 75000.0])

        dist, fuel, time, _, feasible = compute_climb_segment(
            src_alt_ft, dst_alt_ft, src_mass, atyp, delta_isa=0.0, tailwind=0.0
        )
        assert np.all(feasible)
        # Heavier aircraft has lower ROCD, spending more time climbing,
        # burning more fuel over a longer distance.
        assert fuel[1] > fuel[0]
        assert dist[1] > dist[0]
        assert time[1] > time[0]

    def test_infeasible_at_high_altitude(self, atyp: PSParams) -> None:
        src_alt_ft = np.array([30000.0])
        dst_alt_ft = np.array([37000.0])
        src_mass = np.array([atyp.amass_mtow])

        _, _, _, _, feasible = compute_climb_segment(
            src_alt_ft, dst_alt_ft, src_mass, atyp, delta_isa=0.0, tailwind=0.0
        )
        assert not feasible[0]


class TestClimbToTarget:
    def test_fuel_and_distance_reasonable(self, atyp: PSParams) -> None:
        mass = 70000.0
        ground_alt_ft = 0.0
        target_alt_ft = 35000.0

        dist, fuel, time, mass_after = climb_to_target(
            mass=mass, ground_alt_ft=ground_alt_ft, target_alt_ft=target_alt_ft, atyp=atyp
        )
        assert dist == pytest.approx(310000, abs=2000)
        assert fuel == pytest.approx(1680, abs=10)
        assert time == pytest.approx(1440, abs=10)
        assert mass_after == pytest.approx(mass - fuel, rel=1e-6)

    def test_no_climb(self, atyp: PSParams) -> None:
        mass = 70000.0
        target_alt_ft = 35000.0

        dist, fuel, time, mass_after = climb_to_target(
            mass=mass, ground_alt_ft=target_alt_ft, target_alt_ft=target_alt_ft, atyp=atyp
        )
        assert dist == 0.0
        assert fuel == 0.0
        assert time == 0.0
        assert mass_after == mass

    def test_higher_target_uses_more_fuel(self, atyp: PSParams) -> None:
        mass = 70000.0
        ground_alt_ft = 0.0

        _, fuel_low, _, _ = climb_to_target(
            mass=mass, ground_alt_ft=ground_alt_ft, target_alt_ft=30000.0, atyp=atyp
        )
        _, fuel_high, _, _ = climb_to_target(
            mass=mass, ground_alt_ft=ground_alt_ft, target_alt_ft=35000.0, atyp=atyp
        )
        assert fuel_high > fuel_low

    def test_heavier_aircraft_uses_more_fuel(self, atyp: PSParams) -> None:
        ground_alt_ft = 0.0
        target_alt_ft = 35000.0

        _, fuel_light, _, _ = climb_to_target(
            mass=70000.0, ground_alt_ft=ground_alt_ft, target_alt_ft=target_alt_ft, atyp=atyp
        )
        _, fuel_heavy, _, _ = climb_to_target(
            mass=78000.0, ground_alt_ft=ground_alt_ft, target_alt_ft=target_alt_ft, atyp=atyp
        )
        assert fuel_heavy > fuel_light


class TestFinalDescent:
    def test_distance_fuel_time_reasonable(self, atyp: PSParams) -> None:
        src_alt_ft = np.array([35000.0])
        dist, fuel, time = final_descent(src_alt_ft, 0.0, atyp)
        assert dist[0] == pytest.approx(203558, abs=100)
        assert time[0] == pytest.approx(1063.7, abs=1.0)
        assert fuel[0] > 0.0

    def test_nonzero_ground_alt(self, atyp: PSParams) -> None:
        src_alt_ft = np.array([35000.0])
        dist_sea, fuel_sea, time_sea = final_descent(src_alt_ft, 0.0, atyp)
        dist_high, fuel_high, time_high = final_descent(src_alt_ft, 5000.0, atyp)
        assert dist_high[0] < dist_sea[0]
        assert fuel_high[0] < fuel_sea[0]
        assert time_high[0] < time_sea[0]

    def test_larger_descent_covers_more_distance(self, atyp: PSParams) -> None:
        src_alt_ft = np.array([35000.0, 40000.0])
        dist, fuel, time = final_descent(src_alt_ft, 0.0, atyp)
        assert dist[1] > dist[0]
        assert fuel[1] > fuel[0]
        assert time[1] > time[0]

    def test_fractional_ground_alt(self, atyp: PSParams) -> None:
        src_alt_ft = np.array([35000.0])
        dist_0, _, _ = final_descent(src_alt_ft, 0.0, atyp)
        dist_500, _, _ = final_descent(src_alt_ft, 500.0, atyp)
        dist_1000, _, _ = final_descent(src_alt_ft, 1000.0, atyp)
        # 500 ft ground alt should interpolate between 0 and 1000
        assert dist_1000[0] < dist_500[0] < dist_0[0]
