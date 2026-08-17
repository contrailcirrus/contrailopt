"""Test gridded-data helpers."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from contrailopt.grid_utils import fill_nan_spatial


@pytest.fixture
def met() -> xr.Dataset:
    """Gridded met on a small (lon, lat, level, time) grid with no NaN."""
    longitude = np.arange(-10.0, 10.0, 1.0)
    latitude = np.arange(30.0, 50.0, 1.0)
    level = np.array([200.0, 250.0, 300.0])
    time = pd.date_range("2025-02-01T12:00", periods=2, freq="1h")

    shape = (len(longitude), len(latitude), len(level), len(time))
    rng = np.random.default_rng(555444333)

    return xr.Dataset(
        {
            "air_temperature": (("longitude", "latitude", "level", "time"), rng.random(shape)),
            "eastward_wind": (("longitude", "latitude", "level", "time"), rng.random(shape)),
        },
        coords={
            "longitude": longitude,
            "latitude": latitude,
            "level": level,
            "time": time,
        },
    )


class TestFillNanSpatial:
    def test_no_nan(self, met: xr.Dataset) -> None:
        out = fill_nan_spatial(met)
        xr.testing.assert_identical(out, met)

    def test_takes_nearest(self, met: xr.Dataset) -> None:
        """An isolated NaN takes one of its four equidistant horizontal neighbors."""
        v = met["air_temperature"].values
        neighbors = {v[2, 5, 1, 0], v[4, 5, 1, 0], v[3, 4, 1, 0], v[3, 6, 1, 0]}
        met["air_temperature"][3, 5, 1, 0] = np.nan

        out = fill_nan_spatial(met)

        assert out["air_temperature"].values[3, 5, 1, 0] in neighbors
        assert not out.isnull().any().to_dataarray().any()

    def test_unambiguous_nearest(self, met: xr.Dataset) -> None:
        """With a single valid cell left in the slice, every cell takes its value."""
        met["air_temperature"][:, :, 1, 0] = np.nan
        met["air_temperature"][7, 9, 1, 0] = 4.0

        out = fill_nan_spatial(met)

        np.testing.assert_array_equal(out["air_temperature"].values[:, :, 1, 0], 4.0)

    def test_slices_are_independent(self, met: xr.Dataset) -> None:
        """A NaN column is filled horizontally, never from a neighboring level or time."""
        met["eastward_wind"][:, :, 0, 0] = np.nan
        met["eastward_wind"][0, 0, 0, 0] = 7.0

        out = fill_nan_spatial(met)

        np.testing.assert_array_equal(out["eastward_wind"].values[:, :, 0, 0], 7.0)
        xr.testing.assert_identical(
            out["eastward_wind"].isel(level=slice(1, None)),
            met["eastward_wind"].isel(level=slice(1, None)),
        )

    def test_does_not_mutate_input(self, met: xr.Dataset) -> None:
        met["air_temperature"][3, 5, 1, 0] = np.nan
        fill_nan_spatial(met)
        assert np.isnan(met["air_temperature"].values[3, 5, 1, 0])


class TestFillNanSpatialErrors:
    def test_all_nan_slice(self, met: xr.Dataset) -> None:
        met["air_temperature"][:, :, 1, 0] = np.nan
        with pytest.raises(ValueError, match="contains only NaN values"):
            fill_nan_spatial(met)

    def test_bad_dims(self, met: xr.Dataset) -> None:
        met["surface_pressure"] = met["air_temperature"].isel(level=0, drop=True)
        with pytest.raises(ValueError, match="has dims"):
            fill_nan_spatial(met)

    def test_horizontal_chunks(self, met: xr.Dataset) -> None:
        pytest.importorskip("dask")

        met["air_temperature"][3, 5, 1, 0] = np.nan
        with pytest.raises(ValueError, match="chunked along 'longitude'"):
            fill_nan_spatial(met.chunk({"longitude": 5}))


class TestFillNanSpatialDask:
    def test_lazy(self, met: xr.Dataset) -> None:
        pytest.importorskip("dask")

        met["air_temperature"][:, :, 1, 0] = np.nan
        met["air_temperature"][7, 9, 1, 0] = 4.0

        out = fill_nan_spatial(met.chunk({"time": 1, "level": 1}))

        assert out.chunksizes
        np.testing.assert_array_equal(out["air_temperature"].values[:, :, 1, 0], 4.0)
