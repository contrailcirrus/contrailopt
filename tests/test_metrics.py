"""Test the metrics module."""

import numpy as np
import pandas as pd
import pytest
from pycontrails import Flight
from pycontrails.physics import constants

from contrailopt import slerp
from contrailopt.metrics import (
    _mean_altitude_deviation,
    _mean_lateral_deviation,
    _point_to_linestring_dist,
    flight_metrics,
)


def _make_flight(
    longitude: np.ndarray,
    latitude: np.ndarray,
    altitude_ft: float | np.ndarray = 35000.0,
) -> Flight:
    """Build a Flight from coordinate arrays."""
    n = len(longitude)
    time = pd.date_range("2025-06-01T12:00", periods=n, freq="1min")
    if isinstance(altitude_ft, (int, float)):
        altitude_ft = np.full(n, altitude_ft)
    return Flight(
        longitude=longitude,
        latitude=latitude,
        altitude_ft=altitude_ft,
        time=time,
    )


# Two simple flights for reuse
@pytest.fixture
def fl_gc() -> Flight:
    """A 10-point flight along a great circle from JFK to LAX."""
    frac = np.linspace(0, 1, 10)
    lons, lats = slerp.gc_interp(-73.78, 40.64, -118.41, 33.94, frac)
    return _make_flight(lons, lats)


@pytest.fixture
def fl_shifted() -> Flight:
    """Same route as fl_gc but shifted ~1 degree north."""
    frac = np.linspace(0, 1, 10)
    lons, lats = slerp.gc_interp(-73.78, 41.64, -118.41, 34.94, frac)
    return _make_flight(lons, lats)


class TestFlightMetrics:
    def test_identical_flights(self, fl_gc: Flight) -> None:
        result = flight_metrics(fl_gc, fl_gc)
        assert result["dist_cand_m"] == result["dist_base_m"]
        assert result["time_cand_s"] == result["time_base_s"]
        assert result["lateral_dev_m"] == pytest.approx(0.0, abs=1.0)
        assert result["altitude_dev_ft"] == pytest.approx(0.0, abs=0.01)

    def test_symmetry(self, fl_gc: Flight, fl_shifted: Flight) -> None:
        ab = flight_metrics(fl_gc, fl_shifted)
        ba = flight_metrics(fl_shifted, fl_gc)
        assert ab["lateral_dev_m"] == pytest.approx(ba["lateral_dev_m"], rel=1e-6)
        assert ab["altitude_dev_ft"] == pytest.approx(ba["altitude_dev_ft"], rel=1e-6)

    def test_dist_sign(self, fl_gc: Flight) -> None:
        lons = fl_gc["longitude"]
        lats = fl_gc["latitude"]

        # Make a longer flight by adding a detour point
        lons_detour = np.r_[lons[:5], -96.0, lons[5:]]
        lats_detour = np.r_[lats[:5], 50.0, lats[5:]]
        fl_long = _make_flight(lons_detour, lats_detour)
        result = flight_metrics(fl_long, fl_gc)
        assert result["dist_cand_m"] > result["dist_base_m"] * 1.1  # detour is longer

    def test_output_keys(self, fl_gc: Flight, fl_shifted: Flight) -> None:
        result = flight_metrics(fl_gc, fl_shifted)
        expected_keys = {
            "dist_cand_m",
            "dist_base_m",
            "time_cand_s",
            "time_base_s",
            "lateral_dev_m",
            "altitude_dev_ft",
        }
        assert result.keys() == expected_keys


class TestLateralDeviation:
    def test_shifted_flight_has_positive_deviation(self, fl_gc: Flight, fl_shifted: Flight) -> None:
        dev = _mean_lateral_deviation(fl_gc, fl_shifted)
        # 1 degree of latitude ~ 111 km
        assert dev == pytest.approx(constants.radius_earth * np.radians(1.0), rel=0.05)

    def test_different_resolutions(self) -> None:
        """Coarse and fine flights along same great circle should have near-zero deviation."""
        frac_fine = np.linspace(0, 1, 500)
        lons_fine, lats_fine = slerp.gc_interp(-73.78, 40.64, -118.41, 33.94, frac_fine)
        frac_coarse = np.linspace(0, 1, 15)
        lons_coarse, lats_coarse = slerp.gc_interp(-73.78, 40.64, -118.41, 33.94, frac_coarse)

        fl_fine = _make_flight(lons_fine, lats_fine)
        fl_coarse = _make_flight(lons_coarse, lats_coarse)

        dev = _mean_lateral_deviation(fl_fine, fl_coarse)
        assert dev == pytest.approx(0.0, abs=1.0)


class TestAltitudeDeviation:
    def test_symmetric(self) -> None:
        frac = np.linspace(0, 1, 20)
        lons, lats = slerp.gc_interp(0.0, 40.0, 10.0, 45.0, frac)
        fl_high = _make_flight(lons, lats, altitude_ft=37000.0)
        fl_low = _make_flight(lons, lats, altitude_ft=35000.0)

        ab = _mean_altitude_deviation(fl_high, fl_low)
        ba = _mean_altitude_deviation(fl_low, fl_high)
        assert ab == pytest.approx(ba, rel=1e-6)

    def test_constant_offset(self) -> None:
        frac = np.linspace(0, 1, 20)
        lons, lats = slerp.gc_interp(0.0, 40.0, 10.0, 45.0, frac)
        fl_high = _make_flight(lons, lats, altitude_ft=37000.0)
        fl_low = _make_flight(lons, lats, altitude_ft=35000.0)

        dev = _mean_altitude_deviation(fl_high, fl_low)
        assert dev == pytest.approx(2000.0, abs=0.01)

    def test_different_resolutions(self) -> None:
        """Same altitude profile at different resolutions should give ~zero."""
        frac_fine = np.linspace(0, 1, 500)
        lons_fine, lats_fine = slerp.gc_interp(0.0, 40.0, 10.0, 45.0, frac_fine)
        frac_coarse = np.linspace(0, 1, 15)
        lons_coarse, lats_coarse = slerp.gc_interp(0.0, 40.0, 10.0, 45.0, frac_coarse)

        fl_fine = _make_flight(lons_fine, lats_fine, altitude_ft=35000.0)
        fl_coarse = _make_flight(lons_coarse, lats_coarse, altitude_ft=35000.0)

        dev = _mean_altitude_deviation(fl_fine, fl_coarse)
        assert dev == pytest.approx(0.0, abs=1.0)


class TestPointToLinestringDist:
    def test_multiple_points(self) -> None:
        """Multiple query points each find their closest segment independently."""
        # L-shaped polyline: east along equator, then north
        line_lon = np.array([0.0, 10.0, 10.0])
        line_lat = np.array([0.0, 0.0, 10.0])

        # Three points: one near first segment, one near second, one near the corner
        p_lon = np.array([5.0, 10.0, 15.0])
        p_lat = np.array([1.0, 5.0, 0.0])
        dist = _point_to_linestring_dist(p_lon, p_lat, line_lon, line_lat)
        R = constants.radius_earth

        # Point 0: 1 degree north of first segment
        assert dist[0] == pytest.approx(R * np.radians(1.0), rel=1e-3)
        # Point 1: exactly on second segment
        assert dist[1] == pytest.approx(0.0, abs=100.0)
        # Point 2: 5 degrees east of endpoint (10,0)
        assert dist[2] == pytest.approx(R * np.radians(5.0), rel=1e-3)

    def test_degenerate_segment(self) -> None:
        """A duplicate vertex (zero-length segment) should not produce zero distance."""
        # Polyline: [0,0] -> [5,0] -> [5,0] -> [10,0] with a degenerate segment in the middle
        line_lon = np.array([0.0, 5.0, 5.0, 10.0])
        line_lat = np.array([0.0, 0.0, 0.0, 0.0])

        # Point 1 degree north of the midpoint
        p_lon = np.array([5.0])
        p_lat = np.array([1.0])
        dist = _point_to_linestring_dist(p_lon, p_lat, line_lon, line_lat)
        R = constants.radius_earth
        assert dist[0] == pytest.approx(R * np.radians(1.0), rel=1e-3)
