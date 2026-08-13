"""Generate the README figure from synthetic contrail forcing."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pycontrails import Flight
from pycontrails.physics import geo, units
from scipy.ndimage import gaussian_filter1d

from contrailopt import AirportCoords, Optimizer
from contrailopt.slerp import gc_npts

ORIGIN_ICAO = "KBOS"
DEST_ICAO = "EGLL"
AIRCRAFT_TYPE = "B737"
TAKEOFF_TIME = pd.Timestamp("2026-03-01T12:00:00")

CRUISE_FT = 35_000.0
WAYPOINT_FREQ = "60s"

FIELD_LEVELS = np.arange(28_000.0, 40_001.0, 1000.0)
FL_CHOICES = np.arange(28_000.0, 40_001.0, 2000.0)

BLOBS = (
    # (start fraction, end fraction, lower altitude, upper altitude)
    (0.05, 0.25, 33500.0, 34500.0),
    (0.55, 0.8, 37500.0, 38500.0),
)
EEF_RANGE = (2e8, 1e9)
EEF_CORRELATION = 0.01
SEED = 1234

# Add a small offset between the two profiles, so they stay distinct where they coincide
PROFILE_OFFSET_FT = 40.0

COST_INDEX = 80.0
DOLLAR_TONNE_CO2E = 10.0
STEP_PENALTY_KG = 0.0

OUTPUT = "docs/_static/profiles.png"


def synthetic_flight() -> Flight:
    """Build a great-circle track cruising the whole way at ``CRUISE_FT``."""
    origin = AirportCoords.from_icao(ORIGIN_ICAO)
    dest = AirportCoords.from_icao(DEST_ICAO)

    dist_m = geo.haversine(*origin.coords, *dest.coords)
    ground_speed = units.mach_number_to_tas(0.78, units.m_to_T_isa(units.ft_to_m(CRUISE_FT)))
    duration = pd.Timedelta(seconds=dist_m / ground_speed)

    time = pd.date_range(TAKEOFF_TIME, TAKEOFF_TIME + duration, freq=WAYPOINT_FREQ)
    lon, lat = gc_npts(*origin.coords, *dest.coords, len(time) - 2)
    lon = np.r_[origin.longitude, lon, dest.longitude]
    lat = np.r_[origin.latitude, lat, dest.latitude]

    return Flight(
        longitude=lon,
        latitude=lat,
        altitude_ft=np.full(len(time), CRUISE_FT),
        time=time,
        attrs={
            "aircraft_type": AIRCRAFT_TYPE,
            "origin_airport": ORIGIN_ICAO,
            "destination_airport": DEST_ICAO,
        },
    )


def forcing_field(flight: Flight) -> xr.DataArray:
    """Evaluate the blobs of forcing at every flight level along the track."""
    frac = np.linspace(0.0, 1.0, flight.size)
    rng = np.random.default_rng(SEED)
    noise = gaussian_filter1d(
        rng.standard_normal((len(frac), len(FIELD_LEVELS))),
        EEF_CORRELATION * len(frac),
        axis=0,
    )
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    eef = EEF_RANGE[0] + (EEF_RANGE[1] - EEF_RANGE[0]) * noise

    field = np.zeros((len(frac), len(FIELD_LEVELS)))
    for frac_start, frac_end, altitude_low, altitude_high in BLOBS:
        along = (frac >= frac_start) & (frac <= frac_end)
        across = (altitude_low <= FIELD_LEVELS) & (altitude_high >= FIELD_LEVELS)
        inside = along[:, np.newaxis] & across[np.newaxis, :]
        field[inside] = eef[inside]

    return xr.DataArray(
        field,
        dims=("waypoint", "altitude_ft"),
        coords={"altitude_ft": FIELD_LEVELS},
        name="Effective energy forcing per meter [J / m]",
    )


def synthetic_profile(flight: Flight, field: xr.DataArray) -> xr.Dataset:
    """Build a still, standard-atmosphere profile at the levels the optimizer may cruise at."""
    n_waypoints = flight.size

    air_temperature = np.broadcast_to(
        units.m_to_T_isa(units.ft_to_m(FL_CHOICES)), (n_waypoints, len(FL_CHOICES))
    )
    wind = np.zeros((n_waypoints, len(FL_CHOICES)))

    return xr.Dataset(
        {
            "air_temperature": (("waypoint", "altitude_ft"), air_temperature),
            "u_wind": (("waypoint", "altitude_ft"), wind),
            "v_wind": (("waypoint", "altitude_ft"), wind),
            "eef_per_m": field.sel(altitude_ft=FL_CHOICES),
        },
        coords={"altitude_ft": FL_CHOICES},
    )


def solve(flight: Flight, profile: xr.Dataset, dollar_tonne_co2e: float) -> Flight:
    """Optimize the vertical profile of ``flight`` under a given carbon price."""
    opt = Optimizer.from_flight(
        flight,
        fl_profile=profile,
        cost_index=COST_INDEX,
        dollar_tonne_co2e=dollar_tonne_co2e,
        use_flown_climb_descent=True,
    )
    opt.solve(step_penalty_kg=STEP_PENALTY_KG)
    return opt.to_flight()


def _offset(flight: Flight, feet: float) -> Flight:
    """Return a copy of ``flight`` shifted vertically for drawing."""
    shifted = flight.copy()
    shifted.update(altitude_ft=flight.altitude_ft + feet)
    return shifted


def plot(
    flight: Flight,
    field: xr.DataArray,
    fl_cost: Flight,
    fl_contrail: Flight,
) -> plt.Figure:
    """Draw both optimized profiles over the blobs of forcing."""
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "RdBu_r_warm", mpl.colormaps["RdBu_r"](np.linspace(0.5, 1.0, 256))
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    _offset(fl_cost, PROFILE_OFFSET_FT).plot_profile(ax=ax, label="Cost optimal")
    _offset(fl_contrail, -PROFILE_OFFSET_FT).plot_profile(ax=ax, label="Contrail aware")

    curtain = field.rename({"waypoint": "time"}).assign_coords(time=flight["time"])
    curtain.plot.pcolormesh(
        x="time",
        y="altitude_ft",
        ax=ax,
        cmap=cmap,
        vmin=0.0,
        vmax=EEF_RANGE[1],
    )

    ax.set_yticks(FL_CHOICES)
    ax.set_ylim(FL_CHOICES[0] - 1000.0, FL_CHOICES[-1] + 1000.0)
    ax.grid(True, axis="y")
    ax.legend(loc="upper left")

    return fig


def main() -> None:
    """Solve both trajectories and write the figure."""
    flight = synthetic_flight()
    field = forcing_field(flight)
    profile = synthetic_profile(flight, field)

    fl_cost = solve(flight, profile, 0.0)
    fl_contrail = solve(flight, profile, DOLLAR_TONNE_CO2E)

    fig = plot(flight, field, fl_cost, fl_contrail)
    fig.savefig(OUTPUT, dpi=120, bbox_inches="tight")
    print(f"Wrote to {OUTPUT}")


if __name__ == "__main__":
    main()
