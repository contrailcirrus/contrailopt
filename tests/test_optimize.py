"""Tests for the optimize module."""

import types

import numpy as np
import pandas as pd
import pytest
from pycontrails import Flight
from pycontrails.models.ps_model import ps_aircraft_params
from pycontrails.models.ps_model.ps_aircraft_params import PSAircraftEngineParams as PSParams

from contrailopt.dag import AirportCoords, HorizontalDAG
from contrailopt.optimize import (
    FLOAT_DTYPE,
    DAGState,
    Optimizer,
    _build_dag,
    _compute_edge_climbs,
    _compute_ground_climbs,
    _cruise_zone_weights,
    _estimate_flight_hours,
    _estimate_mass,
    _estimate_sample_times,
    _expand_edge_samples,
    _isa_cruise,
    cruise_flight_levels,
    solve_dag,
)


@pytest.fixture
def atyp() -> PSParams:
    """Load A320 engine parameters."""
    return ps_aircraft_params.load_aircraft_engine_params()["A320"]


def _mock_lookup(**kwargs) -> types.SimpleNamespace:
    """Create a SimpleNamespace with EdgeMetLookup fields used by helpers."""
    return types.SimpleNamespace(**kwargs)


class TestExpandEdgeSamples:
    def test_subset_of_edges(self) -> None:
        """Selecting non-contiguous edges returns correct sample indices."""
        # 3 edges: 2, 3, and 2 samples respectively
        edge_ptr = np.array([0, 2, 5, 7])
        lookup = _mock_lookup(edge_ptr=edge_ptr)

        # Query edges 0 and 2, skipping edge 1
        sample_idxs, sample_to_edge, edge_bounds = _expand_edge_samples(lookup, np.array([0, 2]))

        assert sample_idxs.shape == (4,)
        np.testing.assert_array_equal(sample_idxs, [0, 1, 5, 6])
        np.testing.assert_array_equal(sample_to_edge, [0, 0, 1, 1])
        np.testing.assert_array_equal(edge_bounds, [0, 2])

    def test_single_edge(self) -> None:
        """Single-edge query returns that edge's samples."""
        edge_ptr = np.array([0, 4, 6])
        lookup = _mock_lookup(edge_ptr=edge_ptr)

        sample_idxs, sample_to_edge, edge_bounds = _expand_edge_samples(lookup, np.array([1]))

        np.testing.assert_array_equal(sample_idxs, [4, 5])
        np.testing.assert_array_equal(sample_to_edge, [0, 0])
        np.testing.assert_array_equal(edge_bounds, [0])

    def test_all_edges(self) -> None:
        """Querying all edges concatenates all samples in order."""
        edge_ptr = np.array([0, 3, 5])
        lookup = _mock_lookup(edge_ptr=edge_ptr)

        sample_idxs, sample_to_edge, edge_bounds = _expand_edge_samples(lookup, np.array([0, 1]))

        np.testing.assert_array_equal(sample_idxs, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(sample_to_edge, [0, 0, 0, 1, 1])
        np.testing.assert_array_equal(edge_bounds, [0, 3])


class TestEstimateSampleTimes:
    def test_shape_and_monotonicity(self) -> None:
        """Times have shape (n_sample, n_fl) and increase along an edge."""
        cum_dist = np.array([0.0, 100_000.0, 200_000.0])
        lookup = _mock_lookup(cum_dist=cum_dist)

        times = _estimate_sample_times(
            met_lookup=lookup,
            sample_idxs=np.array([0, 1, 2]),
            sample_to_edge=np.array([0, 0, 0]),
            src_idx=np.array([0]),
            takeoff_time=pd.Timestamp("2024-01-01"),
            src_elapsed=np.array([0.0]),
            climb_time=np.array([[60.0, 120.0]]),  # (1, 2)
            fl_choices=np.array([33_000.0, 37_000.0]),
            mach_choices=np.array([0.78]),
        )

        assert times.shape == (3, 2)  # times should be (n_sample, n_fl)
        assert times.dtype == np.dtype("datetime64[ns]")
        # Monotonically increasing along edge for each FL
        assert np.all(times[1] > times[0])
        assert np.all(times[2] > times[1])

    def test_higher_fl_has_later_start(self) -> None:
        """Higher FL starts later due to longer climb time."""
        cum_dist = np.array([0.0])
        lookup = _mock_lookup(cum_dist=cum_dist)

        times = _estimate_sample_times(
            met_lookup=lookup,
            sample_idxs=np.array([0]),
            sample_to_edge=np.array([0]),
            src_idx=np.array([0]),
            takeoff_time=pd.Timestamp("2024-01-01"),
            src_elapsed=np.array([0.0]),
            climb_time=np.array([[60.0, 300.0]]),
            fl_choices=np.array([29_000.0, 37_000.0]),
            mach_choices=np.array([0.78]),
        )

        assert times[0, 1] > times[0, 0]

    def test_second_edge_later_than_first(self) -> None:
        """Source with more elapsed time produces later sample times."""
        cum_dist = np.array([0.0, 50_000.0, 0.0, 50_000.0])
        lookup = _mock_lookup(cum_dist=cum_dist)

        times = _estimate_sample_times(
            met_lookup=lookup,
            sample_idxs=np.array([0, 1, 2, 3]),
            sample_to_edge=np.array([0, 0, 1, 1]),
            src_idx=np.array([0, 1]),
            takeoff_time=pd.Timestamp("2024-01-01"),
            src_elapsed=np.array([0.0, 3600.0]),
            climb_time=np.array([[60.0], [60.0]]),
            fl_choices=np.array([33_000.0]),
            mach_choices=np.array([0.78]),
        )

        assert times.shape == (4, 1)
        # Second edge's first sample should be later than first edge's last
        assert times[2, 0] > times[1, 0]


class TestCruiseZoneWeights:
    def test_known_weights(self) -> None:
        """Verify exact weights for known climb/descent boundaries."""
        cum_dist = np.array([0.0, 100_000.0, 200_000.0, 300_000.0])
        delta_dist = np.array([100_000.0, 100_000.0, 100_000.0, 0.0])
        lookup = _mock_lookup(cum_dist=cum_dist, delta_dist=delta_dist)

        weights = _cruise_zone_weights(
            met_lookup=lookup,
            sample_idxs=np.array([0, 1, 2, 3]),
            sample_to_edge=np.array([0, 0, 0, 0]),
            climb_dist=np.array([[50_000.0, 150_000.0]]),
            descent_dd=np.array([[50_000.0, 50_000.0]]),
            flat_dist=np.array([400_000.0]),
        )

        # FL0 cruise zone: [50k, 350k]; FL1 cruise zone: [150k, 350k]
        expected = np.array(
            [
                [0.5, 0.0],  # [0, 100k]: partial/none
                [1.0, 0.5],  # [100k, 200k]: full/partial
                [1.0, 1.0],  # [200k, 300k]: full/full
                [0.0, 0.0],  # delta_dist = 0
            ]
        )
        np.testing.assert_allclose(weights, expected)

    def test_all_cruise(self) -> None:
        """Zero climb and descent gives full cruise weight on interior samples."""
        cum_dist = np.array([0.0, 100_000.0, 200_000.0])
        delta_dist = np.array([100_000.0, 100_000.0, 0.0])
        lookup = _mock_lookup(cum_dist=cum_dist, delta_dist=delta_dist)

        weights = _cruise_zone_weights(
            met_lookup=lookup,
            sample_idxs=np.array([0, 1, 2]),
            sample_to_edge=np.array([0, 0, 0]),
            climb_dist=np.array([[0.0]]),
            descent_dd=np.array([[0.0]]),
            flat_dist=np.array([300_000.0]),
        )

        np.testing.assert_allclose(weights, [[1.0], [1.0], [0.0]])

    def test_climb_exceeds_edge(self) -> None:
        """Climb longer than edge gives all weights zero."""
        cum_dist = np.array([0.0, 100_000.0])
        delta_dist = np.array([100_000.0, 0.0])
        lookup = _mock_lookup(cum_dist=cum_dist, delta_dist=delta_dist)

        weights = _cruise_zone_weights(
            met_lookup=lookup,
            sample_idxs=np.array([0, 1]),
            sample_to_edge=np.array([0, 0]),
            climb_dist=np.array([[500_000.0]]),
            descent_dd=np.array([[0.0]]),
            flat_dist=np.array([200_000.0]),
        )

        np.testing.assert_allclose(weights, [[0.0], [0.0]])

    def test_descent_exceeds_edge(self) -> None:
        """Descent longer than edge gives all weights zero."""
        cum_dist = np.array([0.0, 100_000.0])
        delta_dist = np.array([100_000.0, 0.0])
        lookup = _mock_lookup(cum_dist=cum_dist, delta_dist=delta_dist)

        weights = _cruise_zone_weights(
            met_lookup=lookup,
            sample_idxs=np.array([0, 1]),
            sample_to_edge=np.array([0, 0]),
            climb_dist=np.array([[0.0]]),
            descent_dd=np.array([[500_000.0]]),
            flat_dist=np.array([200_000.0]),
        )

        np.testing.assert_allclose(weights, [[0.0], [0.0]])


class TestComputeGroundClimbs:
    def test_shapes(self, atyp: PSParams) -> None:
        """Origin wavefront returns correct shapes and dtypes."""
        fl_choices = np.array([29_000.0, 33_000.0, 37_000.0], dtype=FLOAT_DTYPE)

        dist, fuel, time, mass, feasible = _compute_ground_climbs(
            fl_choices, 70_000.0, atyp, 1_000.0
        )

        assert dist.shape == (3,)
        assert fuel.shape == (3,)
        assert time.shape == (3,)
        assert mass.shape == (3,)
        assert feasible.shape == (3,)
        assert dist.dtype == FLOAT_DTYPE
        assert fuel.dtype == FLOAT_DTYPE
        assert mass.dtype == FLOAT_DTYPE

    def test_monotonic(self, atyp: PSParams) -> None:
        """Higher FL requires more distance, fuel, and time from ground."""
        fl_choices = np.array([29_000.0, 33_000.0, 37_000.0], dtype=FLOAT_DTYPE)

        dist, fuel, time, _, _ = _compute_ground_climbs(fl_choices, 70_000.0, atyp, 1_000.0)

        assert np.all(np.diff(dist) > 0)
        assert np.all(np.diff(fuel) > 0)
        assert np.all(np.diff(time) > 0)


class TestComputeEdgeClimbs:
    def test_level_flight(self, atyp: PSParams) -> None:
        """Same source and dest FL -> zero climb."""
        fl_choices = np.array([33_000.0], dtype=FLOAT_DTYPE)

        dist, fuel, time, mass, feasible = _compute_edge_climbs(
            fl_idxs=np.array([0]),
            fl_choices=fl_choices,
            src_idx=np.array([0]),
            src_masses=np.array([65_000.0], dtype=FLOAT_DTYPE),
            src_elapsed=np.array([0.0], dtype=FLOAT_DTYPE),
            flat_edge_idx=np.array([0]),
            atyp=atyp,
            takeoff_time=pd.Timestamp("2024-01-01"),
            met_lookup=None,
        )

        assert dist[0, 0] == 0.0
        assert fuel[0, 0] == 0.0
        assert time[0, 0] == 0.0
        assert mass[0, 0] == 65_000.0
        assert feasible[0, 0]

    def test_step_climb(self, atyp: PSParams) -> None:
        """FL290 -> FL330 should have positive climb distance and fuel."""
        fl_choices = np.array([29_000.0, 33_000.0], dtype=FLOAT_DTYPE)

        dist, fuel, _, _, feasible = _compute_edge_climbs(
            fl_idxs=np.array([0]),
            fl_choices=fl_choices,
            src_idx=np.array([0]),
            src_masses=np.array([65_000.0], dtype=FLOAT_DTYPE),
            src_elapsed=np.array([0.0], dtype=FLOAT_DTYPE),
            flat_edge_idx=np.array([0]),
            atyp=atyp,
            takeoff_time=pd.Timestamp("2024-01-01"),
            met_lookup=None,
        )

        assert dist.shape == (1, 2)
        # Level flight at FL290
        assert dist[0, 0] == 0.0
        # Step climb to FL330
        assert dist[0, 1] > 0.0
        assert fuel[0, 1] > 0.0
        assert feasible[0, 1]


class TestIsaCruise:
    def test_shapes(self, atyp: PSParams) -> None:
        """Output shapes are (n_edge, n_fl, n_mach) with correct dtypes."""
        fl_choices = np.array([29_000.0, 33_000.0, 37_000.0], dtype=FLOAT_DTYPE)
        mach_choices = np.array([0.76, 0.78], dtype=FLOAT_DTYPE)
        edge_mass = np.full((4, 3), 65_000.0, dtype=FLOAT_DTYPE)
        cruise_dist = np.full((4, 3), 200_000.0, dtype=FLOAT_DTYPE)

        fuel, time, feasible = _isa_cruise(fl_choices, mach_choices, edge_mass, atyp, cruise_dist)

        assert fuel.shape == (4, 3, 2)
        assert time.shape == (4, 3, 2)
        assert feasible.shape == (4, 3, 2)
        assert fuel.dtype == FLOAT_DTYPE
        assert time.dtype == FLOAT_DTYPE

    def test_fuel_and_time_positive(self, atyp: PSParams) -> None:
        """Positive cruise distance yields positive fuel and time."""
        fl_choices = np.array([33_000.0, 35_000.0])
        mach_choices = np.array([0.78])
        edge_mass = np.array([[65_000.0, 65_000.0]])
        cruise_dist = np.array([[200_000.0, 200_000.0]])

        fuel, time, feasible = _isa_cruise(fl_choices, mach_choices, edge_mass, atyp, cruise_dist)

        assert np.all(fuel > 0)
        assert np.all(time > 0)
        assert np.all(feasible)

    def test_heavier_burns_more(self, atyp: PSParams) -> None:
        """Heavier aircraft burns more fuel for the same distance."""
        fl_choices = np.array([33_000.0])
        mach_choices = np.array([0.78])
        edge_mass = np.array([[55_000.0], [75_000.0]])
        cruise_dist = np.array([[200_000.0], [200_000.0]])

        fuel, time, _ = _isa_cruise(fl_choices, mach_choices, edge_mass, atyp, cruise_dist)

        assert fuel[1, 0, 0] > fuel[0, 0, 0]
        # Same distance, Mach, temperature -> same TAS -> same time
        np.testing.assert_allclose(time[0], time[1])

    def test_zero_distance_zero_output(self, atyp: PSParams) -> None:
        """Zero cruise distance produces zero fuel and time."""
        fl_choices = np.array([33_000.0])
        mach_choices = np.array([0.78])
        edge_mass = np.array([[65_000.0]])
        cruise_dist = np.array([[0.0]])

        fuel, time, _ = _isa_cruise(fl_choices, mach_choices, edge_mass, atyp, cruise_dist)

        assert fuel[0, 0, 0] == 0.0
        assert time[0, 0, 0] == 0.0

    def test_higher_mach_shorter_time(self, atyp: PSParams) -> None:
        """Higher Mach number results in shorter cruise time."""
        fl_choices = np.array([33_000.0])
        mach_choices = np.array([0.76, 0.80])
        edge_mass = np.array([[65_000.0]])
        cruise_dist = np.array([[300_000.0]])

        _, time, _ = _isa_cruise(fl_choices, mach_choices, edge_mass, atyp, cruise_dist)

        # Higher mach -> higher TAS -> less time
        assert time[0, 0, 1] < time[0, 0, 0]


class TestSolveDag:
    @pytest.fixture
    def line3(self) -> HorizontalDAG:
        """3-node line DAG: 0 -> 1 -> 2, each edge ~530 km."""
        lon = np.array([-80.0, -73.0, -66.0], dtype=FLOAT_DTYPE)
        lat = np.array([40.0, 40.0, 40.0], dtype=FLOAT_DTYPE)
        dag = HorizontalDAG.from_points(lon, lat, max_angle_deg=60.0, max_dist_m=600_000.0)
        assert dag.n_nodes == 3
        assert dag.n_edges == 2
        return dag

    def test_isa_produces_finite_result(self, atyp: PSParams, line3: HorizontalDAG) -> None:
        """ISA solver produces a finite cost at the destination."""
        fl_choices = np.array([29_000.0, 33_000.0, 37_000.0])
        mach_choices = np.array([0.76, 0.78])

        result = solve_dag(
            line3,
            amass_init=70_000.0,
            fl_choices=fl_choices,
            mach_choices=mach_choices,
            atyp=atyp,
            cost_index=60.0,
            eef_cost_factor=0.0,
            origin_elev_ft=1_000.0,
            dest_elev_ft=500.0,
            takeoff_time=pd.Timestamp("2024-01-01"),
        )

        ground_fi = len(fl_choices)
        assert isinstance(result, DAGState)
        assert np.isfinite(result.best_cost[line3.h_dest, ground_fi])
        assert result.best_mass[line3.h_dest, ground_fi] < 70_000.0
        assert result.best_time[line3.h_dest, ground_fi] > 0.0

    def test_backpointers_form_valid_path(self, atyp: PSParams, line3: HorizontalDAG) -> None:
        """Backpointers trace a valid origin-to-destination path."""
        fl_choices = np.array([29_000.0, 33_000.0, 37_000.0])
        mach_choices = np.array([0.76, 0.78])

        result = solve_dag(
            line3,
            amass_init=70_000.0,
            fl_choices=fl_choices,
            mach_choices=mach_choices,
            atyp=atyp,
            cost_index=60.0,
            eef_cost_factor=0.0,
            origin_elev_ft=1_000.0,
            dest_elev_ft=500.0,
            takeoff_time=pd.Timestamp("2024-01-01"),
        )

        ground_fi = len(fl_choices)
        h, fi = line3.h_dest, ground_fi
        path = [(h, fi)]
        while True:
            prev_h = int(result.best_prev_h[h, fi])
            prev_fi = int(result.best_prev_fi[h, fi])
            if prev_fi == -1:
                break
            h, fi = prev_h, prev_fi
            path.append((h, fi))

        path.reverse()
        assert path[0][0] == line3.h_origin
        assert path[0][1] == ground_fi
        assert path[-1][0] == line3.h_dest
        assert path[-1][1] == ground_fi
        assert len(path) == 3  # origin -> waypoint -> dest

    def test_higher_cost_index_prefers_faster(self, atyp: PSParams, line3: HorizontalDAG) -> None:
        """Higher CI weights time more heavily -> solver picks faster Mach -> shorter flight."""
        fl_choices = np.array([29_000.0, 33_000.0])
        mach_choices = np.array([0.74, 0.76, 0.78, 0.80])
        common = {
            "dag": line3,
            "amass_init": 70_000.0,
            "fl_choices": fl_choices,
            "mach_choices": mach_choices,
            "atyp": atyp,
            "eef_cost_factor": 0.0,
            "origin_elev_ft": 1_000.0,
            "dest_elev_ft": 500.0,
            "takeoff_time": pd.Timestamp("2024-01-01"),
        }

        result_low = solve_dag(**common, cost_index=10.0)
        result_high = solve_dag(**common, cost_index=200.0)

        ground_fi = len(fl_choices)
        time_low = result_low.best_time[line3.h_dest, ground_fi]
        time_high = result_high.best_time[line3.h_dest, ground_fi]
        assert np.isfinite(time_low)
        assert np.isfinite(time_high)
        assert time_high <= time_low

    def test_dtypes(self, atyp: PSParams, line3: HorizontalDAG) -> None:
        """State arrays use FLOAT_DTYPE throughout."""
        fl_choices = np.array([29_000.0, 33_000.0, 37_000.0])
        mach_choices = np.array([0.76, 0.78])

        result = solve_dag(
            line3,
            amass_init=70_000.0,
            fl_choices=fl_choices,
            mach_choices=mach_choices,
            atyp=atyp,
            cost_index=60.0,
            eef_cost_factor=0.0,
            origin_elev_ft=1_000.0,
            dest_elev_ft=500.0,
            takeoff_time=pd.Timestamp("2024-01-01"),
        )

        assert result.best_cost.dtype == FLOAT_DTYPE
        assert result.best_mass.dtype == FLOAT_DTYPE
        assert result.best_time.dtype == FLOAT_DTYPE
        assert result.best_mach.dtype == FLOAT_DTYPE


class TestEstimateFlightHours:
    def test_returns_positive_int(self, atyp: PSParams) -> None:
        """Cross-country flight estimate is a positive integer."""
        origin = AirportCoords(
            icao_code="KJFK", longitude=-73.78, latitude=40.64, elevation_ft=13.0
        )
        dest = AirportCoords(
            icao_code="KLAX", longitude=-118.41, latitude=33.94, elevation_ft=128.0
        )
        hours = _estimate_flight_hours(origin, dest, atyp)
        assert isinstance(hours, int)
        assert hours > 0

    def test_short_flight(self, atyp: PSParams) -> None:
        """Short regional flight still estimates at least 1 hour."""
        origin = AirportCoords(
            icao_code="KJFK", longitude=-73.78, latitude=40.64, elevation_ft=13.0
        )
        dest = AirportCoords(icao_code="KBOS", longitude=-71.01, latitude=42.36, elevation_ft=20.0)
        hours = _estimate_flight_hours(origin, dest, atyp)
        assert hours >= 1


class TestEstimateMass:
    def test_explicit_payload(self, atyp: PSParams) -> None:
        """Explicit payload is passed through unchanged."""
        payload, reserve_fuel = _estimate_mass(
            payload=15_000.0,
            origin_icao="KJFK",
            dest_icao="KLAX",
            takeoff_time=pd.Timestamp("2024-06-01"),
            aircraft_type="A320",
            atyp=atyp,
        )
        assert payload == 15_000.0
        assert reserve_fuel > 0.0

    def test_estimated_payload(self, atyp: PSParams) -> None:
        """Estimated payload from pycontrails is positive."""
        payload, reserve_fuel = _estimate_mass(
            payload=None,
            origin_icao="KJFK",
            dest_icao="KLAX",
            takeoff_time=pd.Timestamp("2024-06-01"),
            aircraft_type="A320",
            atyp=atyp,
        )
        assert payload > 0.0
        assert reserve_fuel > 0.0


class TestCruiseFlightLevels:
    def test_eastbound(self) -> None:
        """Eastbound flight gets odd FLs starting at FL290."""
        fls = cruise_flight_levels("KJFK", "EGLL")
        assert fls[0] == 29_000.0
        assert np.all(np.diff(fls) == 2000.0)
        assert fls[-1] < 44_000.0
        assert fls.dtype == FLOAT_DTYPE

    def test_westbound(self) -> None:
        """Westbound flight gets even FLs starting at FL280."""
        fls = cruise_flight_levels("EGLL", "KJFK")
        assert fls[0] == 28_000.0
        assert np.all(np.diff(fls) == 2000.0)


class TestBuildDag:
    def test_generates_dag_when_none(self) -> None:
        """Passing dag=None generates a new Poisson DAG."""
        origin = AirportCoords(
            icao_code="KJFK", longitude=-73.78, latitude=40.64, elevation_ft=13.0
        )
        dest = AirportCoords(icao_code="KBOS", longitude=-71.01, latitude=42.36, elevation_ft=20.0)
        dag = _build_dag(origin, dest, dag=None, avoidance_regions=None)
        assert dag.n_nodes > 2
        assert dag.n_edges > 0

    def test_accepts_matching_dag(self) -> None:
        """Matching pre-built DAG is passed through unchanged."""
        origin = AirportCoords(
            icao_code="KJFK", longitude=-73.78, latitude=40.64, elevation_ft=13.0
        )
        dest = AirportCoords(icao_code="KBOS", longitude=-71.01, latitude=42.36, elevation_ft=20.0)
        exist = HorizontalDAG.from_poisson(*origin.coords, *dest.coords, dtype=FLOAT_DTYPE).prune()
        dag = _build_dag(origin, dest, dag=exist, avoidance_regions=None)
        assert dag == exist

    def test_rejects_mismatched_dag(self) -> None:
        """DAG with wrong endpoints raises ValueError."""
        origin = AirportCoords(
            icao_code="KJFK", longitude=-73.78, latitude=40.64, elevation_ft=13.0
        )
        dest = AirportCoords(icao_code="KBOS", longitude=-71.01, latitude=42.36, elevation_ft=20.0)
        wrong = HorizontalDAG.from_poisson(-118.0, 34.0, -87.0, 42.0).prune()
        with pytest.raises(ValueError, match="does not agree"):
            _build_dag(origin, dest, dag=wrong, avoidance_regions=None)


class TestOptimizer:
    def test_init_no_met(self) -> None:
        """Constructor without met builds DAG and sets choices."""
        opt = Optimizer("KJFK", "KBOS", "A320", pd.Timestamp("2024-06-01"))
        assert opt.origin.icao_code == "KJFK"
        assert opt.dest.icao_code == "KBOS"
        assert opt.met_lookup is None
        assert opt.dag.n_nodes > 2
        assert len(opt.fl_choices) > 0
        assert len(opt.mach_choices) > 0

    def test_init_tz_aware(self) -> None:
        """Tz-aware takeoff_time is normalized to naive UTC."""
        t_utc = pd.Timestamp("2024-06-01 12:00", tz="US/Eastern")
        opt = Optimizer("KJFK", "KBOS", "A320", t_utc)
        assert opt.takeoff_time.tzinfo is None
        assert opt.takeoff_time == pd.Timestamp("2024-06-01 16:00")

    def test_repr(self) -> None:
        """Repr shows airports and unsolved status."""
        opt = Optimizer("KJFK", "KBOS", "A320", pd.Timestamp("2024-06-01"))
        r = repr(opt)
        assert "KJFK" in r
        assert "KBOS" in r
        assert "unsolved" in r

    def test_solve_and_reconstruct(self) -> None:
        """Solve produces valid result and path traces back to origin."""
        opt = Optimizer("KJFK", "KORD", "A320", pd.Timestamp("2024-06-01"))
        result = opt.solve(n_iter=2, payload=15_000.0)
        assert result.trip_fuel > 0.0
        assert result.payload == 15_000.0
        assert result.reserve_fuel > 0.0
        assert "solved" in repr(opt)

        path_h, _, path_mach = opt.reconstruct_path()
        assert path_h[0] == opt.dag.h_origin
        assert path_h[-1] == opt.dag.h_dest
        assert len(path_h) >= 2
        assert np.isnan(path_mach[0])
        assert np.all(np.isfinite(path_mach[1:]))

    def test_solve_converges(self) -> None:
        """Mass iteration converges within bounds."""
        opt = Optimizer("KJFK", "KORD", "A320", pd.Timestamp("2024-06-01"))
        result = opt.solve(n_iter=5, payload=15_000.0)
        assert result.amass_init <= opt.atyp.amass_mtow
        assert result.landing_mass > opt.atyp.amass_oew

    def test_cost_index_update_on_solve(self) -> None:
        """Passing cost_index to solve updates the instance attribute."""
        opt = Optimizer("KJFK", "KORD", "A320", pd.Timestamp("2024-06-01"), cost_index=60.0)
        opt.solve(n_iter=1, cost_index=120.0, payload=15_000.0)
        assert opt.cost_index == 120.0

    def test_to_flight(self) -> None:
        """The to_flight returns a Flight with expected structure."""
        opt = Optimizer("KJFK", "KORD", "A320", pd.Timestamp("2024-06-01"))
        opt.solve(n_iter=1, payload=15_000.0)
        fl = opt.to_flight()

        assert isinstance(fl, Flight)
        assert len(fl) >= 2

        assert "mach_number" in fl
        assert fl["longitude"][0] == pytest.approx(opt.origin.longitude, abs=0.01)
        assert fl["latitude"][0] == pytest.approx(opt.origin.latitude, abs=0.01)
        assert fl["longitude"][-1] == pytest.approx(opt.dest.longitude, abs=0.01)
        assert fl["latitude"][-1] == pytest.approx(opt.dest.latitude, abs=0.01)
        assert pd.DatetimeIndex(fl["time"]).is_monotonic_increasing
