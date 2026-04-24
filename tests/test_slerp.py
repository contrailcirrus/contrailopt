"""Compare SLERP helpers against pyproj (WGS84 ellipsoid) as ground truth."""

import numpy as np
import pyproj
import pytest

from contrailopt.slerp import gc_interp, gc_npts, spherical_azimuth, spherical_fwd

geod = pyproj.Geod(ellps="WGS84")


OD_PAIRS = [
    # (lon1, lat1, lon2, lat2)
    (-73.78, 40.64, -118.41, 33.94),  # JFK -> LAX
    (0.46, 51.47, 23.84, 37.94),  # LHR -> ATH
    (-73.78, 40.64, 2.55, 49.01),  # JFK -> CDG (transatlantic)
    (25.27, 65.02, 24.96, 60.32),  # Oulu -> Helsinki (high lat, short)
    (-43.17, -22.91, 55.52, -21.12),  # GIG -> MRU (cross-equator-ish)
]


class TestSphericalAzimuth:
    @pytest.mark.parametrize("lon1, lat1, lon2, lat2", OD_PAIRS)
    def test_scalar_azimuth(self, lon1, lat1, lon2, lat2):
        expected, _, _ = geod.inv(lon1, lat1, lon2, lat2)
        result = spherical_azimuth(lon1, lat1, lon2, lat2)
        assert result == pytest.approx(expected, abs=0.1)

    def test_vectorized_azimuth(self):
        lon1 = np.array([p[0] for p in OD_PAIRS])
        lat1 = np.array([p[1] for p in OD_PAIRS])
        lon2 = np.array([p[2] for p in OD_PAIRS])
        lat2 = np.array([p[3] for p in OD_PAIRS])

        expected, _, _ = geod.inv(lon1, lat1, lon2, lat2)
        result = spherical_azimuth(lon1, lat1, lon2, lat2)
        np.testing.assert_allclose(result, expected, atol=0.1)


class TestSphericalFwd:
    @pytest.mark.parametrize("lon1, lat1, lon2, lat2", OD_PAIRS)
    def test_fwd_matches_pyproj(self, lon1, lat1, lon2, lat2):
        az, _, dist = geod.inv(lon1, lat1, lon2, lat2)
        expected_lon, expected_lat, _ = geod.fwd(lon1, lat1, az, dist)
        result_lon, result_lat = spherical_fwd(lon1, lat1, az, dist)

        assert result_lon == pytest.approx(expected_lon, abs=0.2)
        assert result_lat == pytest.approx(expected_lat, abs=0.2)

    def test_short_distance(self):
        lon1, lat1 = -73.78, 40.64
        az = 240.0
        dist = 100_000.0  # short projection should be close to pyproj even on a sphere
        expected_lon, expected_lat, _ = geod.fwd(lon1, lat1, az, dist)
        result_lon, result_lat = spherical_fwd(lon1, lat1, az, dist)
        assert result_lon == pytest.approx(expected_lon, abs=0.005)
        assert result_lat == pytest.approx(expected_lat, abs=0.005)

    def test_vectorized_fwd(self):
        lon = np.array([-73.78, 0.46, 25.27])
        lat = np.array([40.64, 51.47, 65.02])
        az = np.array([270.0, 120.0, 180.0])
        dist = np.array([500_000.0, 1_000_000.0, 200_000.0])
        expected_lon, expected_lat, _ = geod.fwd(lon, lat, az, dist)
        result_lon, result_lat = spherical_fwd(lon, lat, az, dist)
        np.testing.assert_allclose(result_lon, expected_lon, atol=0.05)
        np.testing.assert_allclose(result_lat, expected_lat, atol=0.05)


class TestGCInterp:
    @pytest.mark.parametrize("lon1, lat1, lon2, lat2", OD_PAIRS)
    def test_endpoints(self, lon1, lat1, lon2, lat2):
        frac = np.array([0.0, 1.0])  # frac=0 and frac=1 should return the original
        lon_out, lat_out = gc_interp(lon1, lat1, lon2, lat2, frac)
        assert lon_out[0] == pytest.approx(lon1, abs=1e-10)
        assert lat_out[0] == pytest.approx(lat1, abs=1e-10)
        assert lon_out[1] == pytest.approx(lon2, abs=1e-10)
        assert lat_out[1] == pytest.approx(lat2, abs=1e-10)

    @pytest.mark.parametrize("lon1, lat1, lon2, lat2", OD_PAIRS)
    def test_interior(self, lon1, lat1, lon2, lat2):
        n = 50
        frac = np.linspace(0.0, 1.0, n + 2)[1:-1]
        result_lon, result_lat = gc_interp(lon1, lat1, lon2, lat2, frac)

        pts = geod.npts(lon1, lat1, lon2, lat2, n)
        expected_lon = np.array([p[0] for p in pts])
        expected_lat = np.array([p[1] for p in pts])
        np.testing.assert_allclose(result_lon, expected_lon, atol=0.1)
        np.testing.assert_allclose(result_lat, expected_lat, atol=0.1)


class TestGCNpts:
    @pytest.mark.parametrize("lon1, lat1, lon2, lat2", OD_PAIRS)
    def test_npts_matches_pyproj(self, lon1, lat1, lon2, lat2):
        n = 10
        result_lon, result_lat = gc_npts(lon1, lat1, lon2, lat2, n)
        pts = geod.npts(lon1, lat1, lon2, lat2, n)
        expected_lon = np.array([p[0] for p in pts])
        expected_lat = np.array([p[1] for p in pts])
        assert len(result_lon) == n
        np.testing.assert_allclose(result_lon, expected_lon, atol=0.1)
        np.testing.assert_allclose(result_lat, expected_lat, atol=0.1)

    def test_zero_points(self):
        lon1, lat1, lon2, lat2 = OD_PAIRS[0]
        result_lon, result_lat = gc_npts(lon1, lat1, lon2, lat2, 0)
        assert len(result_lon) == 0
        assert len(result_lat) == 0
