"""Tests for contrailopt.waypoints."""

import io
import pathlib
import zipfile
from unittest import mock

import pandas as pd

from contrailopt import load_faa_waypoints


def _make_faa_index_html() -> bytes:
    return b'<a href="/NASR_Subscription/2025-01-01/">Jan</a>'


def _make_faa_zip() -> bytes:
    csv_content = (
        "FIX_ID,LONG_DECIMAL,LAT_DECIMAL,FIX_USE_CODE,ARTCC_ID_HIGH,CHARTS\n"
        "AAAME,-90.123,40.456,RP,ZAU,ENROUTE HIGH ALTITUDE US\n"
        "BBBME,-80.000,35.000,WP,ZTL,ENROUTE HIGH ALTITUDE US\n"
        "BADCODE,-60.000,25.000,XX,ZNY,ENROUTE HIGH ALTITUDE US\n"
        "LOWALT,-50.000,20.000,WP,ZNY,ENROUTE LOW ALTITUDE US\n"
    )
    inner_buf = io.BytesIO()
    with zipfile.ZipFile(inner_buf, "w") as inner:
        inner.writestr("FIX_BASE.csv", csv_content)

    outer_buf = io.BytesIO()
    with zipfile.ZipFile(outer_buf, "w") as outer:
        outer.writestr("28DaySub_CSV.zip", inner_buf.getvalue())
    return outer_buf.getvalue()


def _fake_download(url: str) -> bytes:
    if "NASR_Subscription" in url:
        return _make_faa_index_html()
    return _make_faa_zip()


class TestLoadFaaWaypoints:
    def test_downloads_parses_and_filters(self, tmp_path: pathlib.Path) -> None:
        cache_path = tmp_path / "waypoints.parquet"
        with (
            mock.patch("contrailopt.waypoints.CACHE_PATH", cache_path),
            mock.patch("contrailopt.waypoints._download_bytes", side_effect=_fake_download),
        ):
            df = load_faa_waypoints()

        assert df.columns.tolist() == ["name", "longitude", "latitude"]
        assert len(df) == 2
        assert df["name"].tolist() == ["AAAME", "BBBME"]
        assert cache_path.exists()

    def test_serves_from_cache_on_second_call(self, tmp_path: pathlib.Path) -> None:
        cache_path = tmp_path / "waypoints.parquet"
        with (
            mock.patch("contrailopt.waypoints.CACHE_PATH", cache_path),
            mock.patch("contrailopt.waypoints._download_bytes", side_effect=_fake_download) as dl,
        ):
            first = load_faa_waypoints()
            second = load_faa_waypoints()

        pd.testing.assert_frame_equal(first, second)
        assert dl.call_count == 2  # index page + zip, only on first call

    def test_refresh_cache_redownloads(self, tmp_path: pathlib.Path) -> None:
        cache_path = tmp_path / "waypoints.parquet"
        with (
            mock.patch("contrailopt.waypoints.CACHE_PATH", cache_path),
            mock.patch("contrailopt.waypoints._download_bytes", side_effect=_fake_download) as dl,
        ):
            load_faa_waypoints()
            load_faa_waypoints(refresh_cache=True)

        assert dl.call_count == 4  # 2 calls per download cycle, twice
