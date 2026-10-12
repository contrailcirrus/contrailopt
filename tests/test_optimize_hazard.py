"""Tests for hazard (e.g. turbulence) avoidance: hard ``hazard_max`` and soft ``hazard_cost_fn``."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pycontrails import MetDataset
from pycontrails.physics import units

from contrailopt import optimize
from contrailopt.dag import AirportCoords

TAKEOFF = pd.Timestamp("2024-01-01T01:00:00")
PAYLOAD = 15_000.0


@pytest.fixture
def route() -> tuple[AirportCoords, AirportCoords]:
    origin = AirportCoords(icao_code="KJFK", longitude=-73.78, latitude=40.64, elevation_ft=13.0)
    dest = AirportCoords(icao_code="KORD", longitude=-87.90, latitude=41.98, elevation_ft=672.0)
    return origin, dest


@pytest.fixture
def met() -> MetDataset:
    """Calm ISA atmosphere over CONUS: hazard alone decides any detour."""
    lons = np.arange(-95.0, -64.0, 0.5)
    lats = np.arange(30.0, 55.0, 0.5)
    levels = np.array([360.0, 175.0], dtype=np.float32)
    times = pd.date_range("2024-01-01", periods=8, freq="h")
    shape = (len(lons), len(lats), len(levels), len(times))
    t_isa = units.m_to_T_isa(units.pl_to_m(levels)).astype(np.float32)
    dims = ["longitude", "latitude", "level", "time"]
    ds = xr.Dataset(
        {
            "air_temperature": (dims, np.broadcast_to(t_isa[None, None, :, None], shape)),
            "eastward_wind": (dims, np.zeros(shape, np.float32)),
            "northward_wind": (dims, np.zeros(shape, np.float32)),
        },
        coords={"longitude": lons, "latitude": lats, "level": levels, "time": times},
    )
    return MetDataset(ds)


def _hazard(met: MetDataset, box: tuple[float, float, float, float] | None, value: float = 1.0):
    """Hazard field equal to ``value`` inside ``box`` (lon0, lon1, lat0, lat1), 0 outside.

    ``box=None`` fills the whole domain.
    """
    ds = met.data
    lon, lat = ds["longitude"].values, ds["latitude"].values
    inside = np.ones((len(lon), len(lat)), bool)
    if box is not None:
        lon0, lon1, lat0, lat1 = box
        inside = ((lon >= lon0) & (lon <= lon1))[:, None] & ((lat >= lat0) & (lat <= lat1))[None, :]
    shape = (len(lon), len(lat), ds.sizes["level"], ds.sizes["time"])
    values = np.broadcast_to(np.where(inside, value, 0.0)[:, :, None, None], shape)
    return xr.DataArray(
        values.astype(np.float32),
        dims=["longitude", "latitude", "level", "time"],
        coords={k: ds[k].values for k in ("longitude", "latitude", "level", "time")},
        name="hazard",
    )


# Straddles the JFK-ORD great circle around 80W, well clear of both airports, and leaves room
# inside the DAG corridor (about 39-44N here) to pass south of it
BLOB = (-81.5, -78.5, 41.2, 42.8)


def _solve(route, met, **kwargs) -> optimize.Optimizer:
    origin, dest = route
    solve_kw = {k: kwargs.pop(k) for k in ("hazard_max", "hazard_cost_fn") if k in kwargs}
    opt = optimize.Optimizer(
        origin_icao=origin,
        dest_icao=dest,
        aircraft_type="B737",
        takeoff_time=TAKEOFF,
        met=met,
        cost_index=30.0,
        met_spacing_m=20_000.0,
        rng=7,
        **kwargs,
    )
    opt.solve(n_iter=2, payload=PAYLOAD, **solve_kw)
    return opt


def _cruise_hazard(opt: optimize.Optimizer) -> np.ndarray:
    """Reported hazard where it is constrained (to_flight leaves climb/descent as NaN)."""
    h = opt.to_flight()["hazard"]
    return h[np.isfinite(h)]


def test_zero_hazard_matches_baseline(route, met) -> None:
    """An all-zero hazard field, a cap and a cost function change nothing."""
    base = _solve(route, met)
    with_zero = _solve(
        route,
        met,
        hazard=_hazard(met, BLOB, value=0.0),
        hazard_max=0.5,
        hazard_cost_fn=lambda h: 1.0 * h,
    )
    for a, b in zip(base.reconstruct_path(), with_zero.reconstruct_path(), strict=True):
        np.testing.assert_array_equal(a, b)
    assert base.result.trip_fuel == with_zero.result.trip_fuel


def test_hard_constraint_avoids_blob(route, met) -> None:
    """Without a cap the route crosses the blob; with one it never exceeds the cap at cruise."""
    hazard = _hazard(met, BLOB)
    uncapped = _solve(route, met, hazard=hazard)
    assert _cruise_hazard(uncapped).max() > 0.5  # the cost-optimal route goes through it

    capped = _solve(route, met, hazard=hazard, hazard_max=0.5)
    assert _cruise_hazard(capped).max() <= 0.5 + 1e-6
    assert capped.result.trip_fuel > uncapped.result.trip_fuel  # the detour costs fuel


def test_soft_cost_reduces_exposure(route, met) -> None:
    """A large shadow cost routes around the blob; a zero cost leaves the route unchanged."""
    hazard = _hazard(met, BLOB)
    base = _solve(route, met, hazard=hazard)
    zero = _solve(route, met, hazard=hazard, hazard_cost_fn=lambda h: 0.0 * h)
    np.testing.assert_array_equal(base.reconstruct_path()[0], zero.reconstruct_path()[0])

    # 1 kg fuel equivalent per meter at hazard 1: far more than any detour costs
    costly = _solve(route, met, hazard=hazard, hazard_cost_fn=lambda h: 1.0 * h)
    assert _cruise_hazard(costly).sum() < _cruise_hazard(base).sum()


def test_infeasible_when_hazard_everywhere(route, met) -> None:
    with pytest.raises(ValueError, match="hazard_max"):
        _solve(route, met, hazard=_hazard(met, None), hazard_max=0.5)


def test_climb_out_of_origin_not_constrained(route, met) -> None:
    """Hazard over the departure airport doesn't block the climb below cruise levels."""
    origin, _ = route
    box = (
        origin.longitude - 0.3,
        origin.longitude + 0.3,
        origin.latitude - 0.3,
        origin.latitude + 0.3,
    )
    opt = _solve(route, met, hazard=_hazard(met, box), hazard_max=0.5)
    assert np.isfinite(opt.result.trip_fuel)


def test_hazard_max_requires_field(route, met) -> None:
    with pytest.raises(ValueError, match="hazard"):
        _solve(route, met, hazard_max=0.5)


def test_to_flight_has_hazard_column(route, met) -> None:
    opt = _solve(route, met, hazard=_hazard(met, BLOB))
    fl = opt.to_flight()
    assert "hazard" in fl
    cruise = fl.altitude_ft >= opt.fl_choices[0] - 1.0
    assert np.isfinite(fl["hazard"][cruise]).any()
    assert np.isnan(fl["hazard"][fl.altitude_ft < opt.fl_choices[0] - 1.0]).all()  # climb/descent
