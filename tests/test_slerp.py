"""Compare SLERP helpers against pyproj (WGS84 ellipsoid) as ground truth."""

import numpy as np
import pyproj
import pytest

from contrailopt.slerp import gc_interp, gc_npts, spherical_fwd

geod = pyproj.Geod(ellps="WGS84")


OD_PAIRS = [
    # (lon1, lat1, lon2, lat2)
    (-73.78, 40.64, -118.41, 33.94),  # JFK -> LAX
    (0.46, 51.47, 23.84, 37.94),  # LHR -> ATH
    (-73.78, 40.64, 2.55, 49.01),  # JFK -> CDG (transatlantic)
    (25.27, 65.02, 24.96, 60.32),  # Oulu -> Helsinki (high lat, short)
    (-43.17, -22.91, 55.52, -21.12),  # GIG -> MRU (cross-equator-ish)
]


class TestSphericalFwd:
    @pytest.mark.parametrize(("lon1", "lat1", "lon2", "lat2"), OD_PAIRS)
    def test_fwd_matches_pyproj(self, lon1: float, lat1: float, lon2: float, lat2: float) -> None:
        az, _, dist = geod.inv(lon1, lat1, lon2, lat2)
        expected_lon, expected_lat, _ = geod.fwd(lon1, lat1, az, dist)
        result_lon, result_lat = spherical_fwd(lon1, lat1, az, dist)

        assert result_lon == pytest.approx(expected_lon, abs=0.2)
        assert result_lat == pytest.approx(expected_lat, abs=0.2)

    def test_short_distance(self) -> None:
        lon1, lat1 = -73.78, 40.64
        az = 240.0
        dist = 100_000.0  # short projection should be close to pyproj even on a sphere
        expected_lon, expected_lat, _ = geod.fwd(lon1, lat1, az, dist)
        result_lon, result_lat = spherical_fwd(lon1, lat1, az, dist)
        assert result_lon == pytest.approx(expected_lon, abs=0.005)
        assert result_lat == pytest.approx(expected_lat, abs=0.005)

    def test_vectorized_fwd(self) -> None:
        lon = np.array([-73.78, 0.46, 25.27])
        lat = np.array([40.64, 51.47, 65.02])
        az = np.array([270.0, 120.0, 180.0])
        dist = np.array([500_000.0, 1_000_000.0, 200_000.0])
        expected_lon, expected_lat, _ = geod.fwd(lon, lat, az, dist)
        result_lon, result_lat = spherical_fwd(lon, lat, az, dist)
        np.testing.assert_allclose(result_lon, expected_lon, atol=0.05)
        np.testing.assert_allclose(result_lat, expected_lat, atol=0.05)

    def test_float32_inputs(self) -> None:
        lon = np.float32(-73.78)
        lat = np.float32(40.64)
        az = np.float32(240.0)
        dist = np.float32(500_000.0)
        lon_f32, lat_f32 = spherical_fwd(lon, lat, az, dist)
        assert lon_f32.dtype == np.float32
        assert lat_f32.dtype == np.float32

        lon_f64, lat_f64 = spherical_fwd(-73.78, 40.64, 240.0, 500_000.0)
        np.testing.assert_allclose(lon_f32, lon_f64, atol=0.01)
        np.testing.assert_allclose(lat_f32, lat_f64, atol=0.01)


class TestGCInterp:
    @pytest.mark.parametrize(("lon1", "lat1", "lon2", "lat2"), OD_PAIRS)
    def test_endpoints(self, lon1: float, lat1: float, lon2: float, lat2: float) -> None:
        frac = np.array([0.0, 1.0])  # frac = 0 and frac = 1 should return the original
        lon_out, lat_out = gc_interp(lon1, lat1, lon2, lat2, frac)
        assert lon_out[0] == pytest.approx(lon1, abs=1e-10)
        assert lat_out[0] == pytest.approx(lat1, abs=1e-10)
        assert lon_out[1] == pytest.approx(lon2, abs=1e-10)
        assert lat_out[1] == pytest.approx(lat2, abs=1e-10)

    @pytest.mark.parametrize(("lon1", "lat1", "lon2", "lat2"), OD_PAIRS)
    def test_interior(self, lon1: float, lat1: float, lon2: float, lat2: float) -> None:
        n = 50
        frac = np.linspace(0.0, 1.0, n + 2)[1:-1]
        result_lon, result_lat = gc_interp(lon1, lat1, lon2, lat2, frac)

        pts = geod.npts(lon1, lat1, lon2, lat2, n)
        expected_lon = np.array([p[0] for p in pts])
        expected_lat = np.array([p[1] for p in pts])
        np.testing.assert_allclose(result_lon, expected_lon, atol=0.1)
        np.testing.assert_allclose(result_lat, expected_lat, atol=0.1)

    def test_zero_length_arc(self) -> None:
        lon, lat = -73.78, 40.64
        frac = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        lon_out, lat_out = gc_interp(lon, lat, lon, lat, frac)
        np.testing.assert_allclose(lon_out, lon, atol=1e-10)
        np.testing.assert_allclose(lat_out, lat, atol=1e-10)

    def test_antipodal_points_not_crash(self) -> None:
        lon1, lat1 = 0.0, 0.0
        lon2, lat2 = 180.0, 0.0
        frac = np.array([0.0, 0.5, 1.0])
        lon_out, lat_out = gc_interp(lon1, lat1, lon2, lat2, frac)
        assert np.all(np.isfinite(lon_out))
        assert np.all(np.isfinite(lat_out))

    def test_extrapolation_outside_0_1_interval(self) -> None:
        lon1, lat1, lon2, lat2 = OD_PAIRS[0]
        frac = np.array([-0.5, 1.5])
        lon_out, lat_out = gc_interp(lon1, lat1, lon2, lat2, frac)
        assert np.all(np.isfinite(lon_out))
        assert np.all(np.isfinite(lat_out))

    def test_float32_inputs(self) -> None:
        lon1, lat1, lon2, lat2 = OD_PAIRS[0]
        frac = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        lon_f32, lat_f32 = gc_interp(
            np.float32(lon1),
            np.float32(lat1),
            np.float32(lon2),
            np.float32(lat2),
            frac,
        )
        assert lon_f32.dtype == np.float32  # dtypes preserved
        assert lat_f32.dtype == np.float32  # dtypes preserved

        lon_f64, lat_f64 = gc_interp(lon1, lat1, lon2, lat2, np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(lon_f32, lon_f64, atol=0.01)
        np.testing.assert_allclose(lat_f32, lat_f64, atol=0.01)


class TestGCNpts:
    @pytest.mark.parametrize(("lon1", "lat1", "lon2", "lat2"), OD_PAIRS)
    def test_npts_matches_pyproj(self, lon1: float, lat1: float, lon2: float, lat2: float) -> None:
        n = 10
        result_lon, result_lat = gc_npts(lon1, lat1, lon2, lat2, n)
        pts = geod.npts(lon1, lat1, lon2, lat2, n)
        expected_lon = np.array([p[0] for p in pts])
        expected_lat = np.array([p[1] for p in pts])
        assert len(result_lon) == n
        np.testing.assert_allclose(result_lon, expected_lon, atol=0.1)
        np.testing.assert_allclose(result_lat, expected_lat, atol=0.1)

    def test_single_midpoint(self) -> None:
        lon1, lat1, lon2, lat2 = OD_PAIRS[0]
        result_lon, result_lat = gc_npts(lon1, lat1, lon2, lat2, 1)
        assert len(result_lon) == 1
        mid_lon, mid_lat = gc_interp(lon1, lat1, lon2, lat2, 0.5)
        np.testing.assert_allclose(result_lon, mid_lon, atol=1e-10)
        np.testing.assert_allclose(result_lat, mid_lat, atol=1e-10)

    def test_zero_points(self) -> None:
        lon1, lat1, lon2, lat2 = OD_PAIRS[0]
        result_lon, result_lat = gc_npts(lon1, lat1, lon2, lat2, 0)
        assert len(result_lon) == 0
        assert len(result_lat) == 0
