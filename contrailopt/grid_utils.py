"""Support for working with gridded data."""

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pycontrails import MetDataArray, MetDataset
from pycontrails.physics import units


def localize_horizontally(
    ds: xr.Dataset,
    sample_lon: npt.NDArray[np.floating],
    sample_lat: npt.NDArray[np.floating],
) -> xr.Dataset:
    """Crop ds to the lon/lat bounding box of the sample points.

    Across the antimeridian, sample_lon spans nearly [-180, 180), so the
    box degenerates to the full grid.
    """
    lon = ds["longitude"].values
    i0 = max(np.searchsorted(lon, sample_lon.min()).item() - 2, 0)
    i1 = np.searchsorted(lon, sample_lon.max()).item() + 2

    lat = ds["latitude"].values
    j0 = max(np.searchsorted(lat, sample_lat.min()).item() - 2, 0)
    j1 = np.searchsorted(lat, sample_lat.max()).item() + 2

    return ds.isel(longitude=slice(i0, i1), latitude=slice(j0, j1))


def bilinear_interp(
    ds: xr.Dataset,
    sample_lon: npt.NDArray[np.floating],
    sample_lat: npt.NDArray[np.floating],
) -> xr.Dataset:
    """Bilinear interpolation of ds onto sample points without full-grid intermediates.

    Assumes ds has dimensions (time, altitude_ft, latitude, longitude) and that
    longitude and latitude coordinates are regularly spaced and ascending.

    Returns a Dataset with a "sample" dimension replacing longitude/latitude.
    """
    lon_coord = ds["longitude"].values
    lat_coord = ds["latitude"].values

    # Find bounding indices
    i = np.searchsorted(lon_coord, sample_lon) - 1
    j = np.searchsorted(lat_coord, sample_lat) - 1
    np.clip(i, 0, len(lon_coord) - 2, out=i)
    np.clip(j, 0, len(lat_coord) - 2, out=j)

    # Fractional weights
    wx = (sample_lon - lon_coord[i]) / (lon_coord[i + 1] - lon_coord[i])
    wy = (sample_lat - lat_coord[j]) / (lat_coord[j + 1] - lat_coord[j])

    wx = wx.astype(np.float32)[:, np.newaxis, np.newaxis]
    wy = wy.astype(np.float32)[:, np.newaxis, np.newaxis]

    w00 = (1.0 - wx) * (1.0 - wy)
    w01 = wx * (1.0 - wy)
    w10 = (1.0 - wx) * wy
    w11 = wx * wy

    result_vars = {}
    for name, da in ds.items():
        # data shape: (longitude, latitude, altitude_ft, time)
        if da.dims != ("longitude", "latitude", "altitude_ft", "time"):
            raise ValueError(f"Unexpected dimensions for variable {name}: {da.dims}")

        v = da.values  # this materializes data into memory if not already loaded
        v = v.astype(np.float32, copy=False)  # some dask bug can cause float64 to leak thorugh
        f00 = v[i, j]
        f01 = v[i + 1, j]
        f10 = v[i, j + 1]
        f11 = v[i + 1, j + 1]

        val = w00 * f00 + w01 * f01 + w10 * f10 + w11 * f11
        result_vars[name] = (("sample", "altitude_ft", "time"), val)

    return xr.Dataset(
        result_vars,
        coords={
            "sample": np.arange(len(sample_lon), dtype=np.int64),
            "altitude_ft": ds["altitude_ft"],
            "time": ds["time"],
        },
    )


def select_flight_times(
    ds: xr.Dataset,
    takeoff_time: pd.Timestamp,
    flight_hours: int,
) -> pd.DatetimeIndex:
    """Return the hourly timestamps present in ds covering the flight window.

    Spans from the hour of ``takeoff_time`` through ``flight_hours`` later.
    Assumes met has hourly spacing, which could be relaxed. Raises if no
    timestamps are available and warns if coverage is only partial.
    """
    if takeoff_time.tzinfo:
        takeoff_time = takeoff_time.tz_convert("UTC").tz_localize(None)

    t0 = takeoff_time.floor("1h")
    extra = t0 < takeoff_time
    times = pd.date_range(t0, periods=flight_hours + extra + 1, freq="h")
    available = pd.DatetimeIndex(ds["time"])
    usable = times[times.isin(available)]

    if len(usable) == 0:
        raise ValueError(
            f"No met data available in the estimated flight window.\n"
            f"Required: {times[0]} ... {times[-1]}.\n"
            f"Available: {available[0]} ... {available[-1]}"
        )

    if len(usable) < len(times):
        warnings.warn(
            f"The met data covers {len(usable)} / {len(times)} estimated flight hours. "
            f"The met time extends to {usable[-1]}, but candidate flights may reach "
            f"{times[-1]}. If needed, met will be extrapolated outside its domain.",
            stacklevel=3,
        )

    return usable


def flight_profile_from_met(
    met: MetDataset | xr.Dataset,
    lon: npt.NDArray[np.floating],
    lat: npt.NDArray[np.floating],
    time: npt.NDArray[np.datetime64],
    altitude_ft: npt.NDArray[np.floating],
    eef: xr.DataArray | MetDataArray | None = None,
) -> xr.Dataset:
    """Interpolate gridded met onto a flight's waypoints at every candidate altitude.

    Produces the ``(waypoint, altitude_ft)`` profile that ``Optimizer.from_flight``
    consumes as its ``fl_profile``, so the gridded and profile entry points share a single
    downstream representation. Each waypoint's column is taken at that waypoint's own time,
    which freezes met against the supplied schedule.

    Parameters
    ----------
    met : MetDataset or xr.Dataset
        Gridded met with ``air_temperature``, ``eastward_wind``, ``northward_wind``,
        and optionally ``eef_per_m``.
    lon, lat : npt.NDArray[np.floating]
        Waypoint coordinates in degrees.
    time : npt.NDArray[np.datetime64]
        Waypoint times, used to select the met time step for each column.
    altitude_ft : npt.NDArray[np.floating]
        Candidate flight levels in feet.
    eef : xr.DataArray or MetDataArray or None
        Effective energy forcing per meter, if supplied separately from ``met``.

    Returns
    -------
    xr.Dataset
        Dims ``(waypoint, altitude_ft)`` with ``air_temperature``, ``u_wind``, ``v_wind``,
        and (when available) ``eef_per_m``; coords ``longitude``, ``latitude``, ``time``.
    """
    def _to_ds(obj: MetDataset | xr.Dataset) -> xr.Dataset:
        return obj.data if isinstance(obj, MetDataset) else MetDataset(obj).data

    ds = _to_ds(met)

    variables = ["air_temperature", "eastward_wind", "northward_wind"]
    if "eef_per_m" in ds and eef is None:
        variables.append("eef_per_m")
    ds = ds[variables]

    # Keep only the hourly steps bracketing the waypoint times
    t = pd.DatetimeIndex(time)
    usable = pd.DatetimeIndex(ds["time"])
    keep = (usable >= t.min().floor("1h")) & (usable <= t.max().ceil("1h"))
    if not keep.any():
        raise ValueError(
            f"No met data covers the flight window {t.min()} ... {t.max()}. "
            f"Available: {usable[0]} ... {usable[-1]}"
        )
    ds = ds.isel(time=keep)

    ds = localize_horizontally(ds, lon, lat)
    ds = to_altitude_ft(ds, altitude_ft)
    ds = bilinear_interp(ds, lon, lat)  # (sample, altitude_ft, time)

    if eef is not None:
        da_eef = eef.data if isinstance(eef, MetDataArray) else eef
        ds_eef = _to_ds(da_eef.to_dataset(name="eef_per_m"))
        ds_eef = ds_eef.isel(time=keep)
        ds_eef = localize_horizontally(ds_eef, lon, lat)
        ds_eef = to_altitude_ft(ds_eef, altitude_ft)
        # Bypass xarray coord alignment - eef altitude_ft may differ slightly from met's
        ds["eef_per_m"] = (
            ("sample", "altitude_ft", "time"),
            bilinear_interp(ds_eef, lon, lat)["eef_per_m"].values,
        )

    profile = _select_waypoint_times(ds, time)
    profile = profile.rename(eastward_wind="u_wind", northward_wind="v_wind")

    return profile.assign_coords(
        longitude=("waypoint", lon),
        latitude=("waypoint", lat),
        time=("waypoint", time),
    )


def _select_waypoint_times(
    ds: xr.Dataset,
    time: npt.NDArray[np.datetime64],
) -> xr.Dataset:
    """Collapse the time dim by interpolating each sample at its own waypoint time."""
    tc = ds["time"].values
    n = len(time)

    if len(tc) == 1:
        lo = hi = np.zeros(n, dtype=np.int64)
        w = np.zeros(n, dtype=np.float32)
    else:
        ts = (tc - tc[0]) / np.timedelta64(1, "s")
        query = (time - tc[0]) / np.timedelta64(1, "s")
        frac = np.interp(query, ts, np.arange(len(tc), dtype=np.float64))
        lo = np.floor(frac).astype(np.int64)
        hi = np.minimum(lo + 1, len(tc) - 1)
        w = (frac - lo).astype(np.float32)

    rows = np.arange(n)
    w2 = w[:, np.newaxis]
    data_vars = {}
    for name, da in ds.items():
        v = da.transpose("sample", "altitude_ft", "time").values
        col = v[rows, :, lo] * (1.0 - w2) + v[rows, :, hi] * w2
        data_vars[name] = (("waypoint", "altitude_ft"), col)

    return xr.Dataset(data_vars, coords={"altitude_ft": ds["altitude_ft"].values})


def to_altitude_ft(
    ds: xr.Dataset,
    altitude_ft: npt.NDArray[np.floating],
) -> xr.Dataset:
    """Convert ds from pressure ``level`` to ``altitude_ft`` and select onto ``altitude_ft``.

    Selects the nearest level within a 50 ft tolerance, falling back to vertical
    interpolation when the requested altitudes don't align with the grid.
    """
    ds_altitude_ft = units.pl_to_ft(ds["level"])
    ds = ds.assign_coords(altitude_ft=ds_altitude_ft).swap_dims(level="altitude_ft")
    try:
        return ds.sel(altitude_ft=altitude_ft, method="nearest", tolerance=50.0)
    except KeyError:
        # In this branch, the data gets materialized.
        # interp promotes dtype so immediately cast back
        return ds.interp(altitude_ft=altitude_ft).astype(np.float32)
