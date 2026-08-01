"""Test the flight-profile optimizer path."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pycontrails import Flight, MetDataset
from pycontrails.models.ps_model import ps_aircraft_params
from pycontrails.physics import geo, units

from contrailopt import ps
from contrailopt.dag import AirportCoords, validate_flight_profile
from contrailopt.grid_utils import flight_profile_from_met
from contrailopt.optimize import (
    FLOAT_DTYPE,
    Optimizer,
    _ProfileArrays,
    _step_change,
)


@pytest.fixture
def track() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A curved track KJFK -> KOMA (chord and along-track distance differ)."""
    origin = AirportCoords.from_icao("KJFK")
    dest = AirportCoords.from_icao("KOMA")

    n = 40
    lon = np.linspace(origin.longitude, dest.longitude, n)
    lat = np.linspace(origin.latitude, dest.latitude, n) + 4.0 * np.sin(np.linspace(0, np.pi, n))

    seg = geo.haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
    elapsed = np.r_[0.0, np.cumsum(seg / 230.0)]  # 230 m/s groundspeed
    time = np.datetime64("2025-02-01T12:00", "ns") + (elapsed * 1e9).astype("timedelta64[ns]")

    return lon, lat, time


@pytest.fixture
def profile(track: tuple[np.ndarray, np.ndarray, np.ndarray]) -> xr.Dataset:
    """Flight profile with ISA temperatures.

    EEF confined to waypoints 15 - 19, and an along-track wind that is a tailwind over the
    second half at the lower FLs.
    """
    lon, lat, time = track
    altitude_ft = np.arange(31000.0, 39001.0, 2000.0)
    T_isa = units.m_to_T_isa(units.ft_to_m(altitude_ft))
    shape = (len(lon), len(altitude_ft))

    eef = np.zeros(shape, dtype=float)
    eef[15:20, :] = 5e9  # give every FL from the 15th to 19th waypoint high EF

    u = np.zeros(shape, dtype=float)
    u[20:, :2] = -50.0  # give lowest two FLs a strong tailwind

    return xr.Dataset(
        {
            "air_temperature": (("waypoint", "altitude_ft"), np.broadcast_to(T_isa, shape).copy()),
            "u_wind": (("waypoint", "altitude_ft"), u),
            "v_wind": (("waypoint", "altitude_ft"), np.zeros(shape)),
            "eef_per_m": (("waypoint", "altitude_ft"), eef),
        },
        coords={
            "longitude": ("waypoint", lon),
            "latitude": ("waypoint", lat),
            "time": ("waypoint", time),
            "altitude_ft": ("altitude_ft", altitude_ft),
        },
    )


@pytest.fixture
def flight(track: tuple[np.ndarray, np.ndarray, np.ndarray]) -> Flight:
    """The flight trajectory corresponding to the ``profile`` fixture."""
    lon, lat, time = track
    return Flight(
        longitude=lon,
        latitude=lat,
        time=time,
        altitude_ft=np.full_like(lon, 35_000.0),
        flight_id="test",
        aircraft_type="A320",
    )


class TestProfileValidation:
    """Test ``validate_flight_profile``."""

    @pytest.mark.parametrize("name", ["air_temperature", "u_wind", "v_wind"])
    def test_nan_weather_raises(self, profile: xr.Dataset, name: str) -> None:
        """Confirm an error is raised if weather contains NaNs."""
        profile[name].values[3, 1] = np.nan
        with pytest.raises(ValueError, match=name):
            validate_flight_profile(profile, profile.sizes["waypoint"])

    def test_nan_eef_is_zero_filled(self, profile: xr.Dataset) -> None:
        """Confirm eef nan-values are 0 filled."""
        profile["eef_per_m"].values[3, 1] = np.nan
        out = validate_flight_profile(profile, profile.sizes["waypoint"])
        assert not np.isnan(out["eef_per_m"].values).any()
        assert out["eef_per_m"].values[3, 1] == 0.0

    def test_raw_ef_is_rejected_with_a_warning(self, profile: xr.Dataset) -> None:
        """Confirm a warning is issue if EF names other than 'eef_per_m' is passed."""
        profile = profile.rename(eef_per_m="ef_per_m")
        with pytest.warns(UserWarning, match="ef_per_m"):
            out = validate_flight_profile(profile, profile.sizes["waypoint"])
        assert "eef_per_m" not in out

    def test_waypoint_count_must_match(self, profile: xr.Dataset) -> None:
        with pytest.raises(ValueError, match="waypoints"):
            validate_flight_profile(profile, profile.sizes["waypoint"] + 1)


class TestOptimizedFlightFollowsTrack:
    """Confirm the track solver optimizes FL and Mach number but keeps horizontal path."""

    @pytest.fixture
    def solved(self, flight: Flight, profile: xr.Dataset) -> Optimizer:
        opt = Optimizer.from_flight(
            flight,
            fl_profile=profile,
            aircraft_type="A320",
            dollar_tonne_co2e=5000.0,
        )
        opt.solve()
        return opt

    def test_output_stays_on_the_original_footprint(
        self,
        solved: Optimizer,
        track: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Confirm every output waypoint lies on the flown track."""
        lon, lat, _ = track
        out = solved.to_flight()

        np.testing.assert_allclose(out["longitude"], lon, atol=1e-4)
        np.testing.assert_allclose(out["latitude"], lat, atol=1e-4)

    def test_departure_is_anchored_and_time_advances(
        self, solved: Optimizer, track: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Departure matches the flown takeoff; arrival is free, since Mach is optimized.

        The old solver pinned the whole schedule; the 2D solver chooses Mach, so arrival
        time deviates from the flown flight. Only the takeoff anchor is preserved, and
        elapsed time must increase monotonically along the output.
        """
        _, _, time = track
        out = solved.to_flight()
        assert abs((out["time"][0] - time[0]) / np.timedelta64(1, "s")) < 1.0
        dt = np.diff(out["time"]) / np.timedelta64(1, "s")
        assert np.all(dt > 0.0)

    def test_mach_is_chosen_and_reported(self, solved: Optimizer) -> None:
        """Mach is a decision variable, so the output carries finite, in-envelope values."""
        out = solved.to_flight()
        mach = out["mach_number"]
        assert np.all(np.isfinite(mach))
        assert np.all((mach > 0.5) & (mach <= solved.atyp.max_mach_num))

    def test_cruise_stays_high_then_drops_into_the_tailwind(self, solved: Optimizer) -> None:
        """The tailwind lives at the two lowest FLs over waypoints 20-39, so flying low is cheap
        in the back half; the windless first half favours a higher, more efficient FL. So the
        cruise FL should be higher over the first half than over the second."""
        out = solved.to_flight()

        alt = out.altitude_ft
        first_half = alt[8:18].mean()  # windless first half, efficient high cruise
        second_half = alt[22:32].mean()  # inside the windy waypoints 20-39
        assert first_half > second_half


class TestFlownClimbDescent:
    """Confirm ``use_flown_climb_descent`` flies the flown climb/descent and optimizes cruise."""

    @pytest.fixture
    def climbing_flight(self, track: tuple[np.ndarray, np.ndarray, np.ndarray]) -> Flight:
        """A flight that climbs from the ground to cruise and back, crossing the candidate FLs."""
        lon, lat, time = track
        alt = np.full_like(lon, 37_000.0)
        ramp = np.linspace(0.0, 37_000.0, 8)

        alt[:8] = ramp
        alt[-8:] = ramp[::-1]

        return Flight(
            longitude=lon,
            latitude=lat,
            time=time,
            altitude_ft=alt,
            flight_id="climb",
            aircraft_type="A320",
        )

    @pytest.fixture
    def solved(self, climbing_flight: Flight, profile: xr.Dataset) -> Optimizer:
        opt = Optimizer.from_flight(
            climbing_flight,
            fl_profile=profile,
            aircraft_type="A320",
            use_flown_climb_descent=True,  # hand off at the lowest candidate FL (31,000 in profile)
        )
        opt.solve()
        return opt

    def test_only_cruise_is_optimized(self, solved: Optimizer, profile: xr.Dataset) -> None:
        """The DAG spans only the waypoints above the hand-off, not the whole flight."""
        assert solved.cruise_elev_ft == 31_000.0
        assert solved.dag.n_nodes < profile.sizes["waypoint"]

    def test_output_is_the_full_flight(self, solved: Optimizer, climbing_flight: Flight) -> None:
        """Confirm to_flight splices the flown climb/descent, so every flown waypoint appears."""
        out = solved.to_flight()
        assert len(out["longitude"]) == len(climbing_flight["longitude"])
        dt = np.diff(out["time"]) / np.timedelta64(1, "s")
        assert np.all(dt > 0.0)

    def test_below_handoff_matches_flown(self, solved: Optimizer, climbing_flight: Flight) -> None:
        """Waypoints below the hand-off are taken from the flown flight unchanged."""
        out = solved.to_flight()
        flown_alt = climbing_flight.altitude_ft
        below = flown_alt < 31_000.0
        np.testing.assert_allclose(out["altitude_ft"][below], flown_alt[below], atol=1.0)

    def test_cruise_stays_above_handoff(self, solved: Optimizer) -> None:
        """The optimized cruise never dips below the hand-off altitude."""
        out = solved.to_flight()
        node = out["node_index"] >= 0  # optimized cruise waypoints
        assert np.all(out["altitude_ft"][node] >= 31_000.0 - 1.0)


class TestStepDescentGeometry:
    """Confirm a step down is spread over the distance the descent covers."""

    N = 200  # ~10 km waypoint spacing, shorter than a 2000 ft step descent
    ALT_FT = np.arange(31000.0, 39001.0, 2000.0)

    @staticmethod
    def _track(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        origin = AirportCoords.from_icao("KJFK")
        dest = AirportCoords.from_icao("KOMA")
        lon = np.linspace(origin.longitude, dest.longitude, n)
        lat = np.linspace(origin.latitude, dest.latitude, n)
        seg = geo.haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
        elapsed = np.r_[0.0, np.cumsum(seg / 230.0)]
        time = np.datetime64("2025-02-01T12:00", "ns") + (elapsed * 1e9).astype("timedelta64[ns]")
        return lon, lat, time

    def _solve(self, u: np.ndarray) -> Optimizer:
        lon, lat, time = self._track(self.N)
        shape = (self.N, len(self.ALT_FT))
        T_isa = units.m_to_T_isa(units.ft_to_m(self.ALT_FT))
        profile = xr.Dataset(
            {
                "air_temperature": (("waypoint", "altitude_ft"), np.broadcast_to(T_isa, shape)),
                "u_wind": (("waypoint", "altitude_ft"), u),
                "v_wind": (("waypoint", "altitude_ft"), np.zeros(shape)),
            },
            coords={
                "longitude": ("waypoint", lon),
                "latitude": ("waypoint", lat),
                "time": ("waypoint", time),
                "altitude_ft": ("altitude_ft", self.ALT_FT),
            },
        )
        flight = Flight(
            longitude=lon,
            latitude=lat,
            time=time,
            altitude_ft=np.full_like(lon, 35_000.0),
            flight_id="step",
            aircraft_type="A320",
        )
        opt = Optimizer.from_flight(flight, fl_profile=profile, aircraft_type="A320")
        opt.solve()
        return opt

    @pytest.fixture
    def stepping_down(self) -> Optimizer:
        """The tailwind moves from the top FLs to the bottom ones, forcing a mid-route step down."""
        shape = (self.N, len(self.ALT_FT))
        u = np.zeros(shape, dtype=float)
        u[: self.N // 2, 3:] = -50.0
        u[self.N // 2 : self.N - 40, :2] = -50.0
        return self._solve(u)

    def test_step_down_covers_real_distance(self, stepping_down: Optimizer) -> None:
        """Confirm a step descent advances more than one waypoint beyond the current one."""
        path_h, path_fi, _ = stepping_down.reconstruct_path()

        # Skip the origin, whose ground sentinel sits above every real FL
        step_downs = [
            (path_h[k], path_h[k + 1])
            for k in range(1, len(path_h) - 1)
            if path_fi[k + 1] < path_fi[k]
        ]
        assert step_downs, f"Expected a step down but path FLs were {path_fi}"
        assert all(h_dst - h_src > 1 for h_src, h_dst in step_downs)

    def test_no_descent_is_steeper_than_the_model(self, stepping_down: Optimizer) -> None:
        """No descent in the output is steeper than a 3 degree path allows."""
        out = stepping_down.to_flight()
        dt = np.diff(out["time"]) / np.timedelta64(1, "s")
        rocd = np.diff(out.altitude_ft) / dt * 60.0

        # TAS is highest at the lowest candidate FL, and the fixture's tailwind peaks at 50 m/s
        tas = units.mach_number_to_tas(
            stepping_down.atyp.m_des, units.m_to_T_isa(units.ft_to_m(self.ALT_FT.min()))
        )
        limit = units.m_to_ft((tas + 50.0) * np.tan(np.deg2rad(3.0))) * 60.0
        assert rocd.min() > -limit


class TestStepChange:
    ALT_FT = np.arange(29000.0, 39001.0, 2000.0)

    @classmethod
    def _profile_arrays(cls, u_wind: np.ndarray) -> _ProfileArrays:
        """A due-east track of 10 km segments at ISA, with the given wind per waypoint.

        Due east makes the along-track wind exactly ``u_wind``, so ground speed is TAS plus it.
        """
        n = len(u_wind)
        shape = (n, len(cls.ALT_FT))
        seg = np.full(n - 1, 10_000.0, dtype=FLOAT_DTYPE)
        T_isa = units.m_to_T_isa(units.ft_to_m(cls.ALT_FT))
        return _ProfileArrays(
            air_temperature=np.broadcast_to(T_isa, shape).astype(FLOAT_DTYPE),
            u_wind=np.broadcast_to(u_wind[:, np.newaxis], shape).astype(FLOAT_DTYPE),
            v_wind=np.zeros(shape, dtype=FLOAT_DTYPE),
            eef_per_m=None,
            cum_dist=np.r_[0.0, np.cumsum(seg)].astype(FLOAT_DTYPE),
            seg_dist=seg,
            seg_azimuth=np.full(n - 1, np.pi / 2, dtype=FLOAT_DTYPE),
        )

    @classmethod
    def _step(cls, u_wind: np.ndarray, src: float, tgt: float, mass: float = 60_000.0) -> tuple:
        atyp = ps_aircraft_params.load_aircraft_engine_params()["A320"]
        return _step_change(
            cls._profile_arrays(np.asarray(u_wind, dtype=FLOAT_DTYPE)),
            0,
            np.array([src], dtype=FLOAT_DTYPE),
            np.array([tgt], dtype=FLOAT_DTYPE),
            np.array([mass], dtype=FLOAT_DTYPE),
            cls.ALT_FT.astype(FLOAT_DTYPE),
            atyp,
        )

    def test_wind_is_read_along_the_climb_not_at_its_start(self) -> None:
        """Two climbs differing only after the first waypoint should not come out the same.

        Both start with the same 50 m/s tailwind. One holds it constant, the other reverses
        to a headwind immediately after.
        """
        held = np.full(120, 50.0)
        reversing = np.full(120, -50.0)
        reversing[0] = 50.0

        dist_held = self._step(held, 31000.0, 35000.0)[0][0]
        dist_reversing = self._step(reversing, 31000.0, 35000.0)[0][0]

        assert dist_reversing < 0.8 * dist_held, (
            f"climb covered {dist_reversing / 1000:.1f} km against {dist_held / 1000:.1f} km, "
            "so the wind is being taken from the starting waypoint"
        )

    def test_descent_follows_the_three_degree_path_at_idle(self) -> None:
        """A descent follows the 3 degree geometry and burns less than cruise fuel."""
        dist, fuel, time, _, _, feasible = self._step(np.zeros(120), 35000.0, 31000.0)

        assert feasible[0]
        # A 4000 ft drop on a 3 degree path, flown in still air
        assert dist[0] == pytest.approx(units.ft_to_m(4000.0) / np.tan(np.deg2rad(3.0)), rel=0.05)

        # Descending burns real fuel, but far less per second than cruising
        atyp = ps_aircraft_params.load_aircraft_engine_params()["A320"]
        cruise_ff, _ = ps.cruise_performance(
            np.array([33000.0]),
            np.array([atyp.m_des]),
            np.array([60_000.0]),
            units.m_to_T_isa(units.ft_to_m(np.array([33000.0]))),
            atyp,
        )
        assert 0.0 < fuel[0] / time[0] < 0.5 * cruise_ff[0]

    def test_level_transition_does_nothing(self) -> None:
        """Staying at the same level costs nothing and advances exactly one waypoint."""
        dist, fuel, time, _, arrival, feasible = self._step(np.zeros(120), 35000.0, 35000.0)

        assert feasible[0]
        assert dist[0] == 0.0
        assert fuel[0] == 0.0
        assert time[0] == 0.0
        assert arrival[0] == 1  # a plain cruise advances exactly one waypoint

    def test_running_out_of_track_is_infeasible(self) -> None:
        """A climb needing more track than remains cannot be flown."""
        _, _, _, _, _, feasible = self._step(np.zeros(4), 29000.0, 39000.0)
        assert not feasible[0]


class TestProfileFromMet:
    """Confirm ``flight_profile_from_met`` builds the profile the track solver reads."""

    # Pressure levels whose pl_to_ft altitudes bracket the candidate flight levels, so
    # to_altitude_ft snaps to them exactly (nearest within tolerance, no interpolation).
    LEVELS = np.array([200.0, 250.0, 300.0])
    ALT_FT = np.sort(units.pl_to_ft(LEVELS)).astype(float)
    TEMP_BASE = 200.0  # K at the met's first time step
    TEMP_RATE = 10.0  # K per hour; a field linear in time interpolates exactly
    U_WIND = 7.0
    V_WIND = -4.0
    EEF_IN_MET = 3.0e9

    @classmethod
    def _met(
        cls,
        track: tuple[np.ndarray, np.ndarray, np.ndarray],
        n_times: int = 4,
        with_eef: bool = False,
    ) -> MetDataset:
        """A gridded met covering ``track``: temperature linear in time, winds constant.

        Temperature is uniform in space and level and rises ``TEMP_RATE`` K per hour, so
        linear time interpolation onto a waypoint's own time is exact and predictable.
        """
        lon, lat, time = track
        lons = np.arange(np.floor(lon.min()) - 1.0, np.ceil(lon.max()) + 2.0, 2.0)
        lats = np.arange(np.floor(lat.min()) - 1.0, np.ceil(lat.max()) + 2.0, 1.0)
        t0 = pd.Timestamp(time[0]).floor("1h")
        times = pd.date_range(t0, periods=n_times, freq="h")

        shape = (len(lons), len(lats), len(cls.LEVELS), len(times))
        hours = (times - times[0]) / np.timedelta64(1, "h")
        temp = np.broadcast_to(cls.TEMP_BASE + cls.TEMP_RATE * hours, shape)

        dims = ["longitude", "latitude", "level", "time"]
        data = {
            "air_temperature": (dims, temp),
            "eastward_wind": (dims, np.full(shape, cls.U_WIND)),
            "northward_wind": (dims, np.full(shape, cls.V_WIND)),
        }
        if with_eef:
            data["eef_per_m"] = (dims, np.full(shape, cls.EEF_IN_MET))

        ds = xr.Dataset(
            data,
            coords={"longitude": lons, "latitude": lats, "level": cls.LEVELS, "time": times},
        )
        return MetDataset(ds)

    def test_time_interpolation_is_exact_for_linear_field(
        self, track: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Confirm each column is read at its waypoint's time."""
        lon, lat, time = track
        prof = flight_profile_from_met(self._met(track), lon, lat, time, self.ALT_FT)

        elapsed_h = (time - time[0]) / np.timedelta64(1, "h")
        expected = self.TEMP_BASE + self.TEMP_RATE * elapsed_h  # per waypoint, uniform over FL
        got = prof["air_temperature"].values
        expected = np.broadcast_to(expected[:, np.newaxis], got.shape)
        np.testing.assert_allclose(got, expected, rtol=1e-4)

    def test_single_time_step_uses_that_step(
        self, track: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """With one met time, every waypoint reads that step (no interpolation possible)."""
        lon, lat, time = track
        prof = flight_profile_from_met(self._met(track, n_times=1), lon, lat, time, self.ALT_FT)
        np.testing.assert_allclose(prof["air_temperature"].values, self.TEMP_BASE, rtol=1e-4)

    def test_output_shape_and_coords(
        self, track: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Aligned waypoint-for-waypoint with the flight, with winds renamed for the solver."""
        lon, lat, time = track
        prof = flight_profile_from_met(self._met(track), lon, lat, time, self.ALT_FT)
        assert prof["air_temperature"].dims == ("waypoint", "altitude_ft")
        assert prof.sizes["waypoint"] == len(lon)
        np.testing.assert_array_equal(prof["longitude"].values, lon)
        np.testing.assert_array_equal(prof["latitude"].values, lat)
        np.testing.assert_array_equal(prof["time"].values, time)
        np.testing.assert_allclose(prof["u_wind"].values, self.U_WIND, rtol=1e-5)
        np.testing.assert_allclose(prof["v_wind"].values, self.V_WIND, rtol=1e-5)

    @pytest.mark.parametrize("source", ["in_met", "separate_arg"])
    def test_eef_lands_in_profile(
        self, track: tuple[np.ndarray, np.ndarray, np.ndarray], source: str
    ) -> None:
        """EEF reaches the profile whether it rides in the met or is passed as a separate field."""
        lon, lat, time = track
        if source == "in_met":
            met = self._met(track, with_eef=True)
            eef_arg = None
            expected = self.EEF_IN_MET
        else:
            met = self._met(track)  # no eef in the met itself
            grid = met.data
            shape = tuple(grid.sizes[d] for d in ("longitude", "latitude", "level", "time"))
            expected = 2.0e9
            eef_arg = xr.DataArray(
                np.full(shape, expected),
                dims=["longitude", "latitude", "level", "time"],
                coords={d: grid[d] for d in ("longitude", "latitude", "level", "time")},
            )
        prof = flight_profile_from_met(met, lon, lat, time, self.ALT_FT, eef=eef_arg)
        assert "eef_per_m" in prof
        np.testing.assert_allclose(prof["eef_per_m"].values, expected, rtol=1e-4)

    def test_time_window_out_of_range_raises(
        self, track: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Waypoint times the met doesn't cover fail loudly rather than extrapolate silently."""
        lon, lat, time = track
        far = time - np.timedelta64(30, "D")
        with pytest.raises(ValueError, match="No met data covers"):
            flight_profile_from_met(self._met(track), lon, lat, far, self.ALT_FT)

    def test_from_flight_with_met_solves_end_to_end(
        self, flight: Flight, track: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """The gridded entry point builds the profile and solves over the full flight."""
        lon, _, _ = track
        met = self._met(track, with_eef=True)
        opt = Optimizer.from_flight(flight, met=met, aircraft_type="A320", altitude_ft=self.ALT_FT)
        opt.solve()
        out = opt.to_flight()
        assert len(out["longitude"]) == len(lon)
