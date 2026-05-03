"""Tests for the optimizer with synthetic met data."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pycontrails import MetDataset
from pycontrails.models.ps_model import ps_aircraft_params
from pycontrails.models.ps_model.ps_aircraft_params import PSAircraftEngineParams as PSParams
from pycontrails.physics import units

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
        """All FLs at the destination should have finite cost (solver explored them)."""
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
        n_finite = np.isfinite(dest_costs).sum()
        assert n_finite == len(fl_choices), (
            f"Only {n_finite}/{len(fl_choices)} FLs reached destination: {dest_costs}"
        )
