"""Spherical geometry helpers using SLERP (Spherical Linear Interpolation).

https://en.wikipedia.org/wiki/Spherical_linear_interpolation

These are rougher than pyproj in that they model Earth as a perfect sphere rather than
a WGS84 ellipsoid but significantly faster for vectorized numpy operations.
"""

import numpy as np
import numpy.typing as npt
from pycontrails.physics import constants


def spherical_azimuth(
    lon1: npt.NDArray[np.floating],
    lat1: npt.NDArray[np.floating],
    lon2: npt.NDArray[np.floating],
    lat2: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Forward azimuth (degrees) from ``(lon1, lat1)`` to ``(lon2, lat2)`` on a sphere.

    Equivalent to ``az, _, _ = geod.inv(lon1, lat1, lon2, lat2)``.
    """
    lon1r = np.deg2rad(lon1)
    lat1r = np.deg2rad(lat1)
    lon2r = np.deg2rad(lon2)
    lat2r = np.deg2rad(lat2)
    dlon = lon2r - lon1r

    az = np.arctan2(
        np.sin(dlon) * np.cos(lat2r),
        np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon),
    )

    return np.rad2deg(az)


def spherical_fwd(
    lon: npt.NDArray[np.floating],
    lat: npt.NDArray[np.floating],
    az: npt.NDArray[np.floating],
    dist: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Project from ``(lon, lat)`` along azimuth by dist meters on a sphere.

    Equivalent to ``lon2, lat2, _ = geod.fwd(lon, lat, az, dist)``.
    """
    lonr = np.deg2rad(lon)
    latr = np.deg2rad(lat)
    azr = np.deg2rad(az)
    d = dist / constants.radius_earth  # angular distance in radians

    sin_d = np.sin(d)
    cos_d = np.cos(d)
    sin_lat = np.sin(latr)
    cos_lat = np.cos(latr)

    lat2 = np.arcsin(sin_lat * cos_d + cos_lat * sin_d * np.cos(azr))
    lon2 = lonr + np.arctan2(
        np.sin(azr) * sin_d * cos_lat,
        cos_d - sin_lat * np.sin(lat2),
    )
    return np.rad2deg(lon2), np.rad2deg(lat2)


def gc_interp(
    lon1: npt.NDArray[np.floating],
    lat1: npt.NDArray[np.floating],
    lon2: npt.NDArray[np.floating],
    lat2: npt.NDArray[np.floating],
    frac: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Interpolate along great circles via SLERP.

    Equivalent to ``geod.fwd(lon1, lat1, az, frac * dist)`` but without
    requiring a separate ``geod.inv`` call to obtain ``az`` and ``dist``.

    Parameters:
        lon1, lat1: Source coordinates in degrees.
        lon2, lat2: Destination coordinates in degrees.
        frac: Fractional position along the arc, 0 = source, 1 = dest.

    Returns (lon, lat) in degrees.
    """
    lon1r = np.deg2rad(lon1)
    lat1r = np.deg2rad(lat1)
    lon2r = np.deg2rad(lon2)
    lat2r = np.deg2rad(lat2)

    # Central angle via haversine formula (numerically stable)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    sigma = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    sin_sigma = np.sin(sigma)
    # Guard against zero-length arcs
    safe = np.where(sin_sigma > 1e-12, sin_sigma, 1.0)
    A = np.sin((1.0 - frac) * sigma) / safe
    B = np.sin(frac * sigma) / safe

    # Cartesian SLERP
    cos_lat1 = np.cos(lat1r)
    cos_lat2 = np.cos(lat2r)
    x = A * cos_lat1 * np.cos(lon1r) + B * cos_lat2 * np.cos(lon2r)
    y = A * cos_lat1 * np.sin(lon1r) + B * cos_lat2 * np.sin(lon2r)
    z = A * np.sin(lat1r) + B * np.sin(lat2r)

    lat_out = np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2)))
    lon_out = np.rad2deg(np.arctan2(y, x))
    return lon_out, lat_out


def gc_npts(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
    n: int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Return n equally-spaced intermediate points along a great circle.

    Equivalent to ``geod.npts(lon1, lat1, lon2, lat2, n)``.
    """
    frac = np.linspace(0.0, 1.0, n + 2)[1:-1]
    return gc_interp(lon1, lat1, lon2, lat2, frac)
