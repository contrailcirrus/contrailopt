"""Download and parse FAA NASR waypoint data."""

import io
import pathlib
import re
import zipfile
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import platformdirs
import requests

NASR_SUBSCRIPTION_URL = (
    "https://www.faa.gov/air_traffic/flight_info/aeronav/aero_data/NASR_Subscription/"
)
NASR_EFFECTIVE_DATE_RE = re.compile(r"NASR_Subscription/(\d{4}-\d{2}-\d{2})", flags=re.IGNORECASE)
CACHE_PATH = pathlib.Path(platformdirs.user_cache_dir("contrailopt")) / "faa_waypoints.parquet"
CACHE_MAX_AGE = timedelta(days=90)
HTTP_TIMEOUT_SECONDS = 60.0


def _download_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.content


def _latest_subscription_zip_url() -> str:
    html = _download_bytes(NASR_SUBSCRIPTION_URL).decode("utf-8", errors="replace")
    dates = sorted(date.fromisoformat(d) for d in NASR_EFFECTIVE_DATE_RE.findall(html))
    if not dates:
        raise ValueError("Could not find NASR subscription effective dates on the FAA page.")

    today = datetime.now(UTC).date()
    current_or_past = [d for d in dates if d <= today]
    effective_date = max(current_or_past) if current_or_past else max(dates)

    return (
        "https://nfdc.faa.gov/webContent/28DaySub/"
        f"28DaySubscription_Effective_{effective_date.isoformat()}.zip"
    )


def _cache_is_fresh(cache_path: pathlib.Path) -> bool:
    if not cache_path.exists():
        return False
    modified_at = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=UTC)
    return (datetime.now(UTC) - modified_at) <= CACHE_MAX_AGE


def _read_cached_waypoints(refresh_cache: bool) -> pd.DataFrame | None:
    if refresh_cache or not CACHE_PATH.exists():
        return None
    if not _cache_is_fresh(CACHE_PATH):
        return None
    return pd.read_parquet(CACHE_PATH)


def _write_waypoints_cache(waypoints: pd.DataFrame) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    waypoints.to_parquet(CACHE_PATH, index=False)


def _load_waypoints_from_zip_bytes(zip_bytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as outer_zip:
        for name in outer_zip.namelist():
            if name.lower().endswith("_csv.zip"):
                csv_bundle_name = name
                break
        else:
            raise ValueError("Could not find the CSV bundle in the FAA subscription archive.")
        with outer_zip.open(csv_bundle_name) as csv_bundle_file:
            csv_bundle_bytes = csv_bundle_file.read()

    with zipfile.ZipFile(io.BytesIO(csv_bundle_bytes)) as csv_bundle:
        names = csv_bundle.namelist()
        if "FIX_BASE.csv" not in names:
            raise ValueError("Could not find FIX_BASE.csv in the CSV bundle.")

        # Apply some filtering ... this should be revisited
        usecols = (
            "FIX_ID",
            "LONG_DECIMAL",
            "LAT_DECIMAL",
            "FIX_USE_CODE",
            "ARTCC_ID_HIGH",
            "CHARTS",
        )
        rename = {"FIX_ID": "name", "LONG_DECIMAL": "longitude", "LAT_DECIMAL": "latitude"}
        with csv_bundle.open("FIX_BASE.csv") as fix_csv_file:
            return (
                pd.read_csv(fix_csv_file, usecols=usecols)
                .query(
                    "CHARTS.str.contains('ENROUTE HIGH') and "
                    "FIX_USE_CODE.str.strip() in ('WP', 'RP', 'NRS')"
                )
                .dropna(subset="ARTCC_ID_HIGH")[list(rename)]
                .rename(columns=rename)
            )


def load_faa_waypoints(refresh_cache: bool = False) -> pd.DataFrame:
    """Load FAA FIX waypoints with ``CACHE_MAX_AGE`` cache refresh.

    If a cached DataFrame exists and is less than ``CACHE_MAX_AGE`` old, use it.
    Otherwise, download the latest FAA NASR subscription zip and rebuild the cache.
    """
    cached_waypoints = _read_cached_waypoints(refresh_cache=refresh_cache)
    if cached_waypoints is not None:
        return cached_waypoints

    zip_bytes = _download_bytes(_latest_subscription_zip_url())
    waypoints = _load_waypoints_from_zip_bytes(zip_bytes)
    _write_waypoints_cache(waypoints)
    return waypoints
