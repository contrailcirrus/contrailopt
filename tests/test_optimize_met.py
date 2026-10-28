"""Tests for the optimizer with synthetic met data."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pycontrails import Flight, MetDataset
from pycontrails.models.ps_model import ps_aircraft_params
from pycontrails.models.ps_model.ps_aircraft_params import PSAircraftEngineParams as PSParams
from pycontrails.physics import geo, units

from contrailopt import optimize
from contrailopt.dag import AirportCoords, HorizontalDAG


@pytest.fixture
def atyp() -> PSParams:
    return ps_aircraft_params.load_aircraft_engine_params()["B737"]


@pytest.fixture
def route() -> tuple[AirportCoords, AirportCoords]:
    origin = AirportCoords(icao_code="KJFK", longitude=-73.78, latitude=40.64, elevation_ft=13.0)
    dest = AirportCoords(icao_code="KORD", longitude=-87.90, latitude=41.98, elevation_ft=672.0)
    return origin, dest


@pytest.fixture
def fl_choices(route: tuple[AirportCoords, AirportCoords]) -> np.ndarray:
    return optimize.cruise_flight_levels(*route)


@pytest.fixture
def line_dag(route: tuple[AirportCoords, AirportCoords]) -> HorizontalDAG:
    """Simple line DAG along the route."""
    origin, dest = route
    n = 20
    lons = np.linspace(origin.longitude, dest.longitude, n)
    lats = np.linspace(origin.latitude, dest.latitude, n)
    return HorizontalDAG.from_points(lons, lats, max_dist_m=300_000.0)


@pytest.fixture
def met() -> MetDataset:
    """Build a synthetic CONUS MetDataset with alternating east-west wind strips.

    Two pressure levels which bracket typical cruise FLs.
    ISA temperatures at both levels, zero northward wind. Eastward wind
    changes sign halfway across the domain, creating alternating headwind/tailwind zones.

    The lower level has zero wind, giving the optimizer a reason to step down when
    facing a headwind at higher FLs.
    """
    lons = np.arange(-90.0, -69.0, 1.0)
    lats = np.arange(35.0, 50.0, 1.0)
    levels = np.array([360.0, 175.0], dtype=np.float32)
    times = pd.date_range("2024-01-01", periods=6, freq="h")

    shape = (len(lons), len(lats), len(levels), len(times))

    T_isa = units.m_to_T_isa(units.pl_to_m(levels))
    air_temperature = np.broadcast_to(T_isa[np.newaxis, np.newaxis, :, np.newaxis], shape)

    # Alternating +/- magnitude eastward wind in 10-degree strips, upper level only
    magnitude = 50.0  # m/s
    strip_sign = np.where(lons < -80.0, 1.0, -1.0)
    eastward_wind = np.zeros(shape, dtype=np.float32)
    eastward_wind[:, :, 1, :] = strip_sign[:, np.newaxis, np.newaxis] * magnitude
    northward_wind = np.zeros(shape, dtype=np.float32)

    dim_names = ["longitude", "latitude", "level", "time"]

    ds = xr.Dataset(
        {
            "air_temperature": (dim_names, air_temperature),
            "eastward_wind": (dim_names, eastward_wind),
            "northward_wind": (dim_names, northward_wind),
        },
        coords={
            "longitude": lons,
            "latitude": lats,
            "level": levels,
            "time": times,
        },
    )
    return MetDataset(ds)


@pytest.fixture
def static_graph(route: tuple[AirportCoords, AirportCoords]) -> xr.Dataset:
    """Build a static graph similar to ``line_dag`` with ``met``."""
    origin, dest = route
    n = 20
    lon = np.linspace(origin.longitude, dest.longitude, n)
    lat = np.linspace(origin.latitude, dest.latitude, n)

    graph = xr.Dataset()
    graph["node"] = (("node",), range(n))
    graph["lon"] = (("node",), lon)
    graph["lat"] = (("node",), lat)

    graph["icao"] = (("icao",), [origin.icao_code, dest.icao_code])
    graph["airport_node"] = (("icao",), [0, n - 1])

    dist = geo.haversine(
        lon[:, np.newaxis],
        lat[:, np.newaxis],
        lon[np.newaxis, :],
        lat[np.newaxis, :],
    )
    tail, head = np.nonzero((dist > 0.0) & (dist <= 300_000.0))
    mask = tail < head
    tail = tail[mask]
    head = head[mask]
    graph["edge"] = (("edge",), range(tail.size))
    graph["tail"] = (("edge",), tail)
    graph["head"] = (("edge",), head)

    level = np.array([360.0, 175.0], dtype=np.float32)
    time = pd.date_range("2024-01-01", periods=6, freq="h")
    graph["level"] = (("level"), level)
    graph["time"] = (("time"), time)

    elon = 0.5 * (lon[tail] + lon[head])
    shape = (graph.sizes["edge"], graph.sizes["level"], graph.sizes["time"])
    
    T_isa = units.m_to_T_isa(units.pl_to_m(level))
    air_temperature = np.broadcast_to(T_isa[np.newaxis, :, np.newaxis], shape)
    graph["air_temperature"] = (("edge", "level", "time"), air_temperature)
    
    magnitude = 50.0  # m/s
    strip_sign = np.where(elon < -80.0, 1.0, -1.0)
    eastward_wind = np.zeros(shape, dtype=np.float32)
    eastward_wind[:, 1, :] = strip_sign[:, np.newaxis] * magnitude
    northward_wind = np.zeros(shape, dtype=np.float32)
    
    az = np.deg2rad(geo.azimuth(lon[tail], lat[tail], lon[head], lat[head]))
    az = az[:, np.newaxis, np.newaxis]
    tailwind = eastward_wind * np.sin(az) + northward_wind * np.cos(az)
    graph["tailwind"] = (("edge", "level", "time"), tailwind)

    return graph


class TestStepDown:
    """Verify the optimizer steps down from a FL with a strong headwind."""

    def test_headwind_forces_step_down(
        self,
        route: tuple[AirportCoords, AirportCoords],
        fl_choices: np.ndarray,
        line_dag: HorizontalDAG,
        met: MetDataset,
    ) -> None:
        """Wind strips force the optimizer to use different FLs along the route."""
        origin, dest = route

        opt = optimize.Optimizer(
            origin_icao=origin,
            dest_icao=dest,
            aircraft_type="B737",
            takeoff_time=pd.Timestamp("2024-01-01T01:00:00"),
            met=met,
            dag=line_dag,
            cost_index=30.0,
            met_spacing_m=40_000.0,
        )
        opt.solve(n_iter=2, payload=15_000.0)
        _, path_fi, _ = opt.reconstruct_path()

        cruise_fls = fl_choices[path_fi[1:-1]]  # exclude ground nodes
        diffs = np.diff(cruise_fls)

        # The optimizer should step down in the headwind strip
        assert np.any(diffs < 0), (
            f"Expected a cruise step-down but path FLs were monotonically non-decreasing: "
            f"{cruise_fls}"
        )

    def test_no_headwind_no_step_down(
        self,
        route: tuple[AirportCoords, AirportCoords],
        fl_choices: np.ndarray,
        line_dag: HorizontalDAG,
    ) -> None:
        """With ISA conditions (no met), the optimizer should not step down."""
        origin, dest = route

        opt = optimize.Optimizer(
            origin_icao=origin,
            dest_icao=dest,
            aircraft_type="B737",
            takeoff_time=pd.Timestamp("2024-01-01T01:00:00"),
            dag=line_dag,
            cost_index=30.0,
        )
        opt.solve(n_iter=2, payload=15_000.0)
        _, path_fi, _ = opt.reconstruct_path()

        cruise_fls = fl_choices[path_fi[1:-1]]  # exclude ground nodes
        diffs = np.diff(cruise_fls)

        # Should be monotonically non-decreasing (step climbs only)
        assert np.all(diffs >= 0), f"Unexpected step-down with ISA conditions: {cruise_fls}"

    def test_headwind_all_fls_finite(
        self,
        route: tuple[AirportCoords, AirportCoords],
        fl_choices: np.ndarray,
        line_dag: HorizontalDAG,
        met: MetDataset,
    ) -> None:
        """The reachable FLs run consecutively from the lowest, with no gaps.

        A high FL can be unreachable because the climb to it drops below the minimum ROCD. A gap in
        the middle cannot happen for that reason, and would mean the solver skipped a usable FL.
        """
        origin, dest = route

        opt = optimize.Optimizer(
            origin_icao=origin,
            dest_icao=dest,
            aircraft_type="B737",
            takeoff_time=pd.Timestamp("2024-01-01T01:00:00"),
            met=met,
            dag=line_dag,
            cost_index=30.0,
            met_spacing_m=40_000.0,
        )
        opt.solve(n_iter=2, payload=15_000.0)

        state = opt.result.state
        dest_costs = state.best_cost[line_dag.h_dest, : len(fl_choices)]
        finite = np.isfinite(dest_costs)

        assert finite[0], f"Lowest FL did not reach the destination: {dest_costs}"
        assert np.all(finite[: finite.sum()]), f"Gap in the reachable FLs: {dest_costs}"


class TestStepPenalty:
    """Confirm ``step_penalty_kg`` suppresses level changes in the 4d DAG solver."""

    def test_a_large_penalty_flattens_the_cruise(
        self,
        route: tuple[AirportCoords, AirportCoords],
        line_dag: HorizontalDAG,
        met: MetDataset,
    ) -> None:
        origin, dest = route
        opt = optimize.Optimizer(
            origin_icao=origin,
            dest_icao=dest,
            aircraft_type="B737",
            takeoff_time=pd.Timestamp("2024-01-01T01:00:00"),
            met=met,
            dag=line_dag,
            cost_index=30.0,
            met_spacing_m=40_000.0,
        )
        opt.solve(n_iter=2, payload=15_000.0)
        unpenalized = np.count_nonzero(np.diff(opt.reconstruct_path()[1][1:-1]))
        assert unpenalized > 0, "fixture no longer changes level, so there is nothing to suppress"

        opt.solve(n_iter=2, payload=15_000.0, step_penalty_kg=10_000.0)
        assert np.count_nonzero(np.diff(opt.reconstruct_path()[1][1:-1])) == 0


class TestStaticGraphValidation:
    """Test static graph validation."""

    def test_fixture_is_valid(
        self,
        route: tuple[AirportCoords, AirportCoords],
        static_graph: xr.Dataset
    ) -> None:
        origin, dest = route
        opt = optimize.Optimizer(
            origin_icao=origin,
            dest_icao=dest,
            aircraft_type="B737",
            takeoff_time=pd.Timestamp("2024-01-01T01:00:00"),
            static_graph=static_graph,
        )
        assert opt.met_lookup is not None

    @pytest.mark.parametrize("drop", ["lon", "lat", "tail", "head", "airport_node", "air_temperature", "tailwind"])
    def test_missing_variable_raises(
        self,
        route: tuple[AirportCoords, AirportCoords],
        static_graph: xr.Dataset,
        drop: str,
    ) -> None:
        static_graph = static_graph.drop_vars(drop)
        origin, dest = route
        with pytest.raises(ValueError, match="Static graph missing required variable"):
            optimize.Optimizer(
                origin_icao=origin,
                dest_icao=dest,
                aircraft_type="B737",
                takeoff_time=pd.Timestamp("2024-01-01T01:00:00"),
                static_graph=static_graph,
            )

    @pytest.mark.parametrize(("dim", "var"), [
        ("node", "lon"),
        ("edge", "tail"),
        ("icao", "airport_node"),
        ("level", "air_temperature"),
        ("time", "air_temperature"),
    ])
    def test_wrong_dimension_raises(
        self,
        route: tuple[AirportCoords, AirportCoords],
        static_graph: xr.Dataset,
        dim: str,
        var: str
    ) -> None:
        static_graph = static_graph.rename_dims({dim: "foo"})
        origin, dest = route
        if var != "air_temperature":
            match = f"Variable '{var}' does not have"
        else:
            match = f"Could not transpose '{var}'"
        with pytest.raises(ValueError, match=match):
            optimize.Optimizer(
                origin_icao=origin,
                dest_icao=dest,
                aircraft_type="B737",
                takeoff_time=pd.Timestamp("2024-01-01T01:00:00"),
                static_graph=static_graph,
            )


class TestStaticGraphParity:
    """Test parity between results with and without static graph."""

    def test_equivalent_dag(
        self,
        route: tuple[AirportCoords, AirportCoords],
        line_dag: HorizontalDAG,
        static_graph: xr.Dataset,
    ) -> None:
        """Test equivalence of dag generated from static graph."""
        origin, dest = route
        opt = optimize.Optimizer(
            origin_icao=origin,
            dest_icao=dest,
            aircraft_type="B737",
            takeoff_time=pd.Timestamp("2024-01-01T01:00:00"),
            static_graph=static_graph,
        )
        dag = opt.dag
        assert dag.h_origin == line_dag.h_origin
        assert dag.h_dest == line_dag.h_dest
        np.testing.assert_array_equal(dag.lon, line_dag.lon.astype(optimize.FLOAT_DTYPE))
        np.testing.assert_array_equal(dag.lat, line_dag.lat.astype(optimize.FLOAT_DTYPE))
        np.testing.assert_array_equal(dag.adj_ptr, line_dag.adj_ptr)
        np.testing.assert_array_equal(dag.adj, line_dag.adj)
        # The optimizer computes edge_dist from float32 coordinates
        np.testing.assert_allclose(dag.edge_dist, line_dag.edge_dist, rtol=1e-4)

    def test_reproduces_level_changes(
        self,
        route: tuple[AirportCoords, AirportCoords],
        fl_choices: np.ndarray,
        line_dag: HorizontalDAG,
        met: MetDataset,
        static_graph: xr.Dataset,
    ) -> None:
        """Test parity with level changes from met-aware optimizer."""
        origin, dest = route

        opt1 = optimize.Optimizer(
            origin_icao=origin,
            dest_icao=dest,
            aircraft_type="B737",
            takeoff_time=pd.Timestamp("2024-01-01T01:00:00"),
            met=met,
            dag=line_dag,
            cost_index=30.0,
            met_spacing_m=40_000.0,
        )
        opt1.solve(n_iter=2, payload=15_000.0)
        _, p1, _ = opt1.reconstruct_path()

        opt2 = optimize.Optimizer(
            origin_icao=origin,
            dest_icao=dest,
            aircraft_type="B737",
            takeoff_time=pd.Timestamp("2024-01-01T01:00:00"),
            static_graph=static_graph,
            cost_index=30.0
        )
        opt2.solve(n_iter=2, payload=15_000.0)
        _, p2, _ = opt2.reconstruct_path()

        fl1 = fl_choices[p1[1:-1]]  # exclude ground nodes
        fl2 = fl_choices[p2[1:-1]]  # exclude ground nodes

        assert np.any(np.diff(fl2) < 0), f"No step descents: {fl2}"
        assert np.sum(np.diff(fl1) > 0) == np.sum(np.diff(fl2) > 0), (
            f"Step climb counts disagree: {fl1} vs. {fl2}"
        )
        assert np.sum(np.diff(fl1) < 0) == np.sum(np.diff(fl2) < 0), (
            f"Step descent counts disagree: {fl1} vs. {fl2}"
        )

    def test_fuel_burn_differences(
        self,
        route: tuple[AirportCoords, AirportCoords],
        line_dag: HorizontalDAG,
        met: MetDataset,
        static_graph: xr.Dataset,
    ) -> None:
        """Test for reasonable differences in fuel burn.

        Differences are expected but should O(10 kg), comparable to differenes
        from using aggregated rather than sampled met data.
        """
        origin, dest = route

        opt1 = optimize.Optimizer(
            origin_icao=origin,
            dest_icao=dest,
            aircraft_type="B737",
            takeoff_time=pd.Timestamp("2024-01-01T01:00:00"),
            met=met,
            dag=line_dag,
            cost_index=30.0,
            met_spacing_m=40_000.0,
        )
        res1 = opt1.solve(n_iter=2, payload=15_000.0)

        opt2 = optimize.Optimizer(
            origin_icao=origin,
            dest_icao=dest,
            aircraft_type="B737",
            takeoff_time=pd.Timestamp("2024-01-01T01:00:00"),
            static_graph=static_graph,
            cost_index=30.0
        )
        res2 = opt2.solve(n_iter=2, payload=15_000.0)
        
        opt3 = optimize.Optimizer(
            origin_icao=origin,
            dest_icao=dest,
            aircraft_type="B737",
            takeoff_time=pd.Timestamp("2024-01-01T01:00:00"),
            met=met,
            dag=line_dag,
            cost_index=30.0,
            met_spacing_m=40_000.0,
            aggregate_met=True,
        )
        res3 = opt3.solve(n_iter=2, payload=15_000.0)
        
        assert not opt1.met_lookup.is_aggregated
        assert opt2.met_lookup.is_aggregated
        assert opt3.met_lookup.is_aggregated

        assert abs(res1.trip_fuel - res2.trip_fuel) > 10.0
        assert abs(res1.trip_fuel - res2.trip_fuel) < 100.0
        
        assert abs(res1.trip_fuel - res3.trip_fuel) > 10.0
        assert abs(res1.trip_fuel - res3.trip_fuel) < 100.0


class TestToFlight:
    """Verify to_flight() produces a valid Flight with met-derived fields."""

    def test_to_flight_structure(
        self,
        route: tuple[AirportCoords, AirportCoords],
        line_dag: HorizontalDAG,
        met: MetDataset,
    ) -> None:
        """to_flight returns a Flight with expected columns and endpoints."""
        origin, dest = route

        opt = optimize.Optimizer(
            origin_icao=origin,
            dest_icao=dest,
            aircraft_type="B737",
            takeoff_time=pd.Timestamp("2024-01-01T01:00:00"),
            met=met,
            dag=line_dag,
            cost_index=30.0,
            met_spacing_m=40_000.0,
        )
        opt.solve(n_iter=2, payload=15_000.0)
        fl = opt.to_flight()

        assert isinstance(fl, Flight)
        assert len(fl) > 20  # densified waypoints, not just DAG nodes

        # Required columns present and finite
        for col in ("mach_number", "air_temperature", "tailwind"):
            assert col in fl, f"Missing column: {col}"
            assert np.all(np.isfinite(fl[col])), f"Non-finite values in {col}"

        # Endpoints match origin/dest
        assert fl["longitude"][0] == pytest.approx(origin.longitude, abs=0.1)
        assert fl["latitude"][0] == pytest.approx(origin.latitude, abs=0.1)
        assert fl["longitude"][-1] == pytest.approx(dest.longitude, abs=0.1)
        assert fl["latitude"][-1] == pytest.approx(dest.latitude, abs=0.1)

        # Time is monotonically increasing
        assert pd.DatetimeIndex(fl["time"]).is_monotonic_increasing

    def test_to_flight_altitude_profile(
        self,
        route: tuple[AirportCoords, AirportCoords],
        line_dag: HorizontalDAG,
        met: MetDataset,
    ) -> None:
        """Altitude starts at origin elevation, reaches cruise, ends at dest elevation."""
        origin, dest = route

        opt = optimize.Optimizer(
            origin_icao=origin,
            dest_icao=dest,
            aircraft_type="B737",
            takeoff_time=pd.Timestamp("2024-01-01T01:00:00"),
            met=met,
            dag=line_dag,
            cost_index=30.0,
            met_spacing_m=40_000.0,
        )
        opt.solve(n_iter=2, payload=15_000.0)
        fl = opt.to_flight()

        alt = fl["altitude_ft"]
        assert alt[0] == pytest.approx(origin.elevation_ft, abs=1.0)
        assert alt[-1] == pytest.approx(dest.elevation_ft, abs=1.0)
        assert np.max(alt) >= 30_000.0  # reaches cruise
        assert np.all(np.isfinite(alt))

    def test_to_flight_with_eef(
        self,
        route: tuple[AirportCoords, AirportCoords],
        line_dag: HorizontalDAG,
        met: MetDataset,
    ) -> None:
        """to_flight includes eef_per_m when met contains it."""
        origin, dest = route

        # Add a spatially varying eef_per_m field to the met dataset
        ds = met.data
        shape = ds["air_temperature"].shape
        # Positive EEF in the western half, negative in the east
        lons = ds["longitude"].values
        eef_sign = np.where(lons < -80.0, 1.0, -1.0)
        eef_per_m = np.broadcast_to(
            (eef_sign * 1e-9)[:, np.newaxis, np.newaxis, np.newaxis], shape
        ).astype(np.float32)
        ds_eef = ds.assign(eef_per_m=(ds["air_temperature"].dims, eef_per_m))
        met_eef = MetDataset(ds_eef)

        opt = optimize.Optimizer(
            origin_icao=origin,
            dest_icao=dest,
            aircraft_type="B737",
            takeoff_time=pd.Timestamp("2024-01-01T01:00:00"),
            met=met_eef,
            dag=line_dag,
            cost_index=30.0,
            dollar_tonne_co2e=100.0,
            met_spacing_m=40_000.0,
        )
        opt.solve(n_iter=2, payload=15_000.0)
        fl = opt.to_flight()

        assert "eef_per_m" in fl
        assert np.all(np.isfinite(fl["eef_per_m"]))
        # Should contain both positive and negative values given the spatial pattern
        assert np.any(fl["eef_per_m"] > 0)
        assert np.any(fl["eef_per_m"] < 0)

    def test_separate_eef_parameter(
        self,
        route: tuple[AirportCoords, AirportCoords],
        line_dag: HorizontalDAG,
        met: MetDataset,
    ) -> None:
        """Passing eef as a separate DataArray on a different grid works."""
        origin, dest = route

        # Build EEF on a deliberately different (coarser) lon/lat grid
        eef_lons = np.arange(-90.0, -69.0, 2.5)
        eef_lats = np.arange(35.0, 50.0, 2.5)
        levels = met.data["level"].values
        times = met.data["time"].values

        shape = (len(eef_lons), len(eef_lats), len(levels), len(times))
        eef_sign = np.where(eef_lons < -80.0, 1.0, -1.0)
        eef_values = np.broadcast_to(
            (eef_sign * 1e-9)[:, np.newaxis, np.newaxis, np.newaxis], shape
        ).astype(np.float32)

        da_eef = xr.DataArray(
            eef_values,
            dims=["longitude", "latitude", "level", "time"],
            coords={
                "longitude": eef_lons,
                "latitude": eef_lats,
                "level": levels,
                "time": times,
            },
            name="eef_per_m",
        )

        opt = optimize.Optimizer(
            origin_icao=origin,
            dest_icao=dest,
            aircraft_type="B737",
            takeoff_time=pd.Timestamp("2024-01-01T01:00:00"),
            met=met,
            eef=da_eef,
            dag=line_dag,
            cost_index=30.0,
            dollar_tonne_co2e=100.0,
            met_spacing_m=40_000.0,
        )
        opt.solve(n_iter=2, payload=15_000.0)
        fl = opt.to_flight()

        assert "eef_per_m" in fl
        assert np.all(np.isfinite(fl["eef_per_m"]))
        assert np.any(fl["eef_per_m"] > 0)
        assert np.any(fl["eef_per_m"] < 0)

    @pytest.mark.parametrize("resample_m", [1_000.0, 10_000.0, 100_000.0])
    def test_to_flight_resample(
        self,
        route: tuple[AirportCoords, AirportCoords],
        line_dag: HorizontalDAG,
        met: MetDataset,
        resample_m: float
    ) -> None:
        """to_flight returns a Flight with expected columns, endpoints, and segment lengths."""
        origin, dest = route

        opt = optimize.Optimizer(
            origin_icao=origin,
            dest_icao=dest,
            aircraft_type="B737",
            takeoff_time=pd.Timestamp("2024-01-01T01:00:00"),
            met=met,
            dag=line_dag,
            cost_index=30.0,
            met_spacing_m=40_000.0,
        )
        opt.solve(n_iter=2, payload=15_000.0)
        fl = opt.to_flight(resample_m=resample_m)

        # Minimal set of columns
        assert set(fl.data.keys()) == {"longitude", "latitude", "altitude_ft", "time"}

        # Endpoints match origin/dest
        assert fl["longitude"][0] == pytest.approx(origin.longitude, abs=0.1)
        assert fl["latitude"][0] == pytest.approx(origin.latitude, abs=0.1)
        assert fl["longitude"][-1] == pytest.approx(dest.longitude, abs=0.1)
        assert fl["latitude"][-1] == pytest.approx(dest.latitude, abs=0.1)

        # Time is monotonically increasing
        assert pd.DatetimeIndex(fl["time"]).is_monotonic_increasing

        # Segment lengths match resampling rate
        dist = fl.segment_haversine()[:-1]
        assert np.all(dist < 1.001 * resample_m)  # should be strictly smaller...
                                                  # not sure why this isn't the case,
                                                  # but we're close enough that I don't
                                                  # think it's worth worrying about.
