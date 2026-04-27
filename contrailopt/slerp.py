"""Spherical geometry helpers using SLERP (Spherical Linear intERPolation).

https://en.wikipedia.org/wiki/Spherical_linear_interpolation

These are rougher than pyproj in that they model Earth as a perfect sphere rather than
a WGS84 ellipsoid but significantly faster for vectorized numpy operations.

The pycontrails library already includes ``geo.haversine`` and ``geo.azimuth``
functions for computing great circle distances and azimuths. The functions here extend
those utilities with forward projection and interpolation capabilities.
"""

import numpy as np
import numpy.typing as npt
from pycontrails.physics import constants


def spherical_fwd(
    lon: npt.NDArray[np.floating],
    lat: npt.NDArray[np.floating],
    az: npt.NDArray[np.floating],
    dist: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    r"""Project from ``(lon, lat)`` along azimuth ``az`` by ``dist`` meters on a sphere.

    Equivalent to ``lon2, lat2, _ = geod.fwd(lon, lat, az, dist)``.

    Parameters
    ----------
    lon : npt.NDArray[np.floating]
        Longitude of starting point, [:math:`\deg`].
    lat : npt.NDArray[np.floating]
        Latitude of starting point, [:math:`\deg`].
    az : npt.NDArray[np.floating]
        Forward azimuth, [:math:`\deg`].
    dist : npt.NDArray[np.floating]
        Distance to project, [:math:`m`].

    Returns
    -------
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
        Longitude and latitude of projected point, [:math:`\deg`].
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
    r"""Interpolate along great circles via SLERP.

    Equivalent to ``geod.fwd(lon1, lat1, az, frac * dist)`` but without
    requiring a separate ``geod.inv`` call to obtain ``az`` and ``dist``.

    Parameters
    ----------
    lon1 : npt.NDArray[np.floating]
        Longitude of source, [:math:`\deg`].
    lat1 : npt.NDArray[np.floating]
        Latitude of source, [:math:`\deg`].
    lon2 : npt.NDArray[np.floating]
        Longitude of destination, [:math:`\deg`].
    lat2 : npt.NDArray[np.floating]
        Latitude of destination, [:math:`\deg`].
    frac : npt.NDArray[np.floating]
        Fractional position along the arc, 0 = source, 1 = dest.

    Returns
    -------
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
        Interpolated longitude and latitude, [:math:`\deg`].
    """
    lon1r = np.deg2rad(lon1)
    lat1r = np.deg2rad(lat1)
    lon2r = np.deg2rad(lon2)
    lat2r = np.deg2rad(lat2)

    cos_lat1 = np.cos(lat1r)
    cos_lat2 = np.cos(lat2r)

    # Central angle via haversine
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + cos_lat1 * cos_lat2 * np.sin(dlon / 2) ** 2
    sigma = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    sin_sigma = np.sin(sigma)
    # Guard against zero-length arcs: as sigma -> 0, we have A -> 1 - frac and  B -> frac
    zero_arc = sin_sigma < 1e-12
    safe_sin_sigma = np.where(zero_arc, 1.0, sin_sigma)  # avoid division by zero
    A = np.where(zero_arc, 1.0 - frac, np.sin((1.0 - frac) * sigma) / safe_sin_sigma)
    B = np.where(zero_arc, frac, np.sin(frac * sigma) / safe_sin_sigma)

    # Cartesian SLERP
    x = A * cos_lat1 * np.cos(lon1r) + B * cos_lat2 * np.cos(lon2r)
    y = A * cos_lat1 * np.sin(lon1r) + B * cos_lat2 * np.sin(lon2r)
    z = A * np.sin(lat1r) + B * np.sin(lat2r)

    lat_out = np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2)))
    lon_out = np.rad2deg(np.arctan2(y, x))
    return lon_out, lat_out


def gc_npts(
    lon1: float, lat1: float, lon2: float, lat2: float, n: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Return n equally-spaced intermediate points along a great circle.

    Excludes the endpoints themselves. Equivalent to
    ``geod.npts(lon1, lat1, lon2, lat2, n)``.

    Parameters
    ----------
    lon1 : float
        Longitude of source, [:math:`\deg`].
    lat1 : float
        Latitude of source, [:math:`\deg`].
    lon2 : float
        Longitude of destination, [:math:`\deg`].
    lat2 : float
        Latitude of destination, [:math:`\deg`].
    n : int
        Number of intermediate points.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        Longitude and latitude of intermediate points, [:math:`\deg`].
        These arrays always have dtype ``np.float64``.
    """
    frac = np.linspace(0.0, 1.0, n + 2, dtype=np.float64)[1:-1]
    return gc_interp(lon1, lat1, lon2, lat2, frac)
