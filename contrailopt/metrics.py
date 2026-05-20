"""Metrics comparing two flights."""

import numpy as np
import numpy.typing as npt
from pycontrails import Flight
from pycontrails.physics import constants


def flight_metrics(fl_a: Flight, fl_b: Flight) -> dict[str, float]:
    """Compute similarity metrics between two flights.

    Parameters
    ----------
    fl_a : Flight
        First flight.
    fl_b : Flight
        Second flight (used as the reference for percentage differences).

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        - dist_a_m: total distance of fl_a in meters
        - dist_b_m: total distance of fl_b in meters
        - dist_diff_pct: percent difference (fl_a - fl_b) / fl_b
        - time_a_s: total duration of fl_a in seconds
        - time_b_s: total duration of fl_b in seconds
        - time_diff_s: duration difference in seconds
        - lateral_dev_m: symmetric mean lateral deviation in meters
        - altitude_dev_ft: symmetric mean altitude deviation in feet
    """
    dist_a = fl_a.segment_length()[:-1].sum().item()
    dist_b = fl_b.segment_length()[:-1].sum().item()

    time_a = fl_a.duration.total_seconds()
    time_b = fl_b.duration.total_seconds()

    lat_dev = _mean_lateral_deviation(fl_a, fl_b)
    alt_dev = _mean_altitude_deviation(fl_a, fl_b)

    return {
        "dist_a_m": dist_a,
        "dist_b_m": dist_b,
        "dist_diff_pct": (dist_a - dist_b) / dist_b * 100.0,
        "time_a_s": time_a,
        "time_b_s": time_b,
        "time_diff_s": time_a - time_b,
        "lateral_dev_m": lat_dev,
        "altitude_dev_ft": alt_dev,
    }


def _mean_lateral_deviation(fl_a: Flight, fl_b: Flight) -> float:
    """Compute symmetric mean point-to-linestring distance between two flights (meters)."""
    a_to_b = (
        _point_to_linestring_dist(
            fl_a["longitude"],
            fl_a["latitude"],
            fl_b["longitude"],
            fl_b["latitude"],
        )
        .mean()
        .item()
    )
    b_to_a = (
        _point_to_linestring_dist(
            fl_b["longitude"],
            fl_b["latitude"],
            fl_a["longitude"],
            fl_a["latitude"],
        )
        .mean()
        .item()
    )
    return (a_to_b + b_to_a) / 2.0


def _mean_altitude_deviation(fl_a: Flight, fl_b: Flight) -> float:
    """Symmetric mean absolute altitude difference between two flights (feet)."""

    def _cum_dist_frac(f: Flight) -> np.ndarray:
        segs = f.segment_length()
        segs[-1] = 0.0  # fill the terminal nan
        cd = np.cumsum(segs)
        total = cd[-1]
        if total == 0.0:  # degenerate, unexpected
            return cd
        return cd / total

    frac_a = _cum_dist_frac(fl_a)
    frac_b = _cum_dist_frac(fl_b)

    alt_b_interp = np.interp(frac_a, frac_b, fl_b.altitude_ft)
    a_to_b = np.abs(fl_a.altitude_ft - alt_b_interp).mean().item()

    alt_a_interp = np.interp(frac_b, frac_a, fl_a.altitude_ft)
    b_to_a = np.abs(fl_b.altitude_ft - alt_a_interp).mean().item()

    return (a_to_b + b_to_a) / 2.0


def _to_xyz(
    lon_deg: npt.NDArray[np.floating],
    lat_deg: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Convert lon/lat in degrees to unit 3D vectors on the sphere."""
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    cos_lat = np.cos(lat)

    x = cos_lat * np.cos(lon)
    y = cos_lat * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def _point_to_linestring_dist(
    p_lon: npt.NDArray[np.floating],
    p_lat: npt.NDArray[np.floating],
    line_lon: npt.NDArray[np.floating],
    line_lat: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute the great-circle distance from each point to a linestring (meters)."""
    P = _to_xyz(p_lon, p_lat)  # (n_pts, 3)
    A = _to_xyz(line_lon[:-1], line_lat[:-1])  # (n_seg, 3)
    B = _to_xyz(line_lon[1:], line_lat[1:])  # (n_seg, 3)

    # Compute the great circle normal to each segment
    N = np.cross(A, B)  # (n_seg, 3)
    N_norm = np.linalg.norm(N, axis=-1, keepdims=True)
    degen = N_norm.squeeze(-1) < 1e-12
    N_unit = N / np.maximum(N_norm, 1e-12)

    # Cross-track angular distance: |arcsin(P \cdot N_unit)| for each (point, seg)
    dot_pn = P @ N_unit.T  # (n_pts, n_seg)
    cross_track = np.abs(np.arcsin(np.clip(dot_pn, -1.0, 1.0)))  # (n_pts, n_seg)

    # Project each point onto each great circle plane, then normalize
    P_proj = P[:, np.newaxis, :] - dot_pn[:, :, np.newaxis] * N_unit[np.newaxis, :, :]
    P_proj_norm = np.linalg.norm(P_proj, axis=-1, keepdims=True)
    P_proj_unit = P_proj / np.maximum(P_proj_norm, 1e-12)

    # Check whether the projection falls on the arc (not the antipodal arc)
    wind_a = np.einsum("ijk,jk->ij", np.cross(A, P_proj_unit), N_unit)
    wind_b = np.einsum("ijk,jk->ij", np.cross(P_proj_unit, B), N_unit)
    on_arc = (wind_a >= -1e-10) & (wind_b >= -1e-10)

    # Angular distance to each endpoint
    dist_A = np.arccos(np.clip(P @ A.T, -1.0, 1.0))  # (n_pts, n_seg)
    dist_B = np.arccos(np.clip(P @ B.T, -1.0, 1.0))
    endpoint_dist = np.minimum(dist_A, dist_B)

    # Pick cross-track if projection is on-arc, otherwise nearest endpoint
    seg_dist = np.where(on_arc & ~degen, cross_track, endpoint_dist)

    return seg_dist.min(axis=1) * constants.radius_earth
