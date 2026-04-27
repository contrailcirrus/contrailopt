"""Tests for the HorizontalDAG class in contrailopt.dag."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pycontrails import MetDataset

from contrailopt import AirportCoords, EdgeMetLookup, HorizontalDAG


@pytest.fixture
def diamond() -> HorizontalDAG:
    r"""A 4-node diamond DAG.

      1
     / \
    0   3
     \ /
      2
    """
    lon = np.array([-80.0, -78.0, -78.0, -76.0])
    lat = np.array([40.0, 40.1, 39.9, 40.0])
    return HorizontalDAG.from_points(lon, lat, max_angle_deg=60.0, max_dist_m=250_000.0)


@pytest.fixture
def lattice() -> HorizontalDAG:
    """A 21x21 regular lattice DAG from (lat=0, lon=40) to (lat=0, lon=60).

    441 nodes on integer coordinates, lat in [-10, 10], lon in [40, 60].
    Pruning removes nodes outside the azimuth corridor.
    """
    lats = np.arange(-10, 11, dtype=float)
    lons = np.arange(40, 61, dtype=float)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    lat = lat_grid.ravel()
    lon = lon_grid.ravel()
    origin_idx = int(np.flatnonzero((lat == 0) & (lon == 40))[0])
    dest_idx = int(np.flatnonzero((lat == 0) & (lon == 60))[0])
    return HorizontalDAG.from_points(lon, lat, origin_idx=origin_idx, dest_idx=dest_idx)


class TestFromPoints:
    def test_diamond_structure(self, diamond: HorizontalDAG) -> None:
        assert diamond.n_nodes == 4
        assert diamond.n_edges == 4
        assert diamond.h_origin == 0
        assert diamond.h_dest == 3

        assert set(diamond.neighbors(0)) == {1, 2}
        assert set(diamond.neighbors(1)) == {3}
        assert set(diamond.neighbors(2)) == {3}
        assert set(diamond.neighbors(3)) == set()

    def test_diamond_repr(self, diamond: HorizontalDAG) -> None:
        assert repr(diamond) == "HorizontalDAG(4 nodes, 4 edges)"

    def test_diamond_adjacency(self, diamond: HorizontalDAG) -> None:
        adj = diamond.adjacency_matrix()
        assert adj.shape == (diamond.n_nodes, diamond.n_nodes)
        assert adj.sum() == diamond.n_edges
        assert adj[0, 1]
        assert adj[0, 2]
        assert adj[1, 3]
        assert adj[2, 3]

    def test_edge_distances_positive(self, diamond: HorizontalDAG) -> None:
        assert np.all(diamond.edge_dist > 0.0)
        assert np.all(diamond.edge_dist < 250_000.0)  # used in diamond fixture

    def test_no_self_loops(self, diamond: HorizontalDAG) -> None:
        edges = diamond.edges
        assert not np.any(edges[:, 0] == edges[:, 1])

    def test_lattice_structure(self, lattice: HorizontalDAG) -> None:
        assert lattice.n_nodes == 441
        assert lattice.n_edges == 2240

    def test_all_edges_forward(self, lattice: HorizontalDAG) -> None:
        # Every edge points strictly eastward on the lattice
        edges = lattice.edges
        assert np.all(lattice.lon[edges[:, 1]] > lattice.lon[edges[:, 0]])


class TestPrune:
    def test_diamond_already_pruned(self, diamond: HorizontalDAG) -> None:
        pruned = diamond.prune()
        assert pruned.n_nodes == diamond.n_nodes
        assert pruned.n_edges == diamond.n_edges

    def test_lattice_prune_removes_nodes(self, lattice: HorizontalDAG) -> None:
        pruned = lattice.prune()
        assert pruned.n_nodes == 127
        assert pruned.n_edges == 1240
        assert pruned.n_nodes < lattice.n_nodes

    def test_prune_connectivity(self, lattice: HorizontalDAG) -> None:
        pruned = lattice.prune()
        adj = pruned.adjacency_matrix()

        reachable = np.zeros(pruned.n_nodes, dtype=bool)
        reachable[pruned.h_origin] = True
        for _ in range(pruned.n_nodes):
            reachable |= adj[reachable].any(axis=0)

        assert reachable.all()

    def test_prune_idempotent(self, lattice: HorizontalDAG) -> None:
        once = lattice.prune()
        twice = once.prune()
        assert once.n_nodes == twice.n_nodes
        assert once.n_edges == twice.n_edges


class TestTopoWavefronts:
    def test_diamond_two_wavefronts(self, diamond: HorizontalDAG) -> None:
        waves = list(diamond.topo_wavefronts())
        assert len(waves) == 3
        assert set(waves[0]) == {0}
        assert set(waves[1]) == {1, 2}
        assert set(waves[2]) == {3}

    def test_wavefronts_cover_all_non_origin(self, diamond: HorizontalDAG) -> None:
        waves = list(diamond.topo_wavefronts())
        all_wave_nodes = set(np.concatenate(waves).tolist())
        assert all_wave_nodes == set(range(diamond.n_nodes))

    def test_lattice_wavefront_count(self, lattice: HorizontalDAG) -> None:
        pruned = lattice.prune()
        waves = list(pruned.topo_wavefronts())
        assert len(waves) == 21
        assert waves[0].size == 1  # only origin in first wavefront
        assert waves[0].tolist() == [pruned.h_origin]

    def test_wavefronts_are_acyclic(self, lattice: HorizontalDAG) -> None:
        pruned = lattice.prune()
        waves = list(pruned.topo_wavefronts())
        adj = pruned.adjacency_matrix()
        for i in range(len(waves)):
            for j in range(i + 1, len(waves)):
                for src in waves[j]:
                    for dst in waves[i]:
                        assert not adj[src, dst]

    def test_dest_in_last_wavefront(self, lattice: HorizontalDAG) -> None:
        pruned = lattice.prune()
        waves = list(pruned.topo_wavefronts())
        assert pruned.h_dest in waves[-1]


class TestEdgesProperty:
    def test_edges_shape(self, diamond: HorizontalDAG) -> None:
        edges = diamond.edges
        assert edges.shape == (diamond.n_edges, 2)

    def test_edges_match_csr(self, lattice: HorizontalDAG) -> None:
        pruned = lattice.prune()
        edges = pruned.edges
        for i in range(pruned.n_nodes):
            nbrs = pruned.neighbors(i)
            edge_nbrs = edges[edges[:, 0] == i, 1]
            assert set(nbrs) == set(edge_nbrs)

    def test_adjacency_matrix_consistent_with_edges(self, lattice: HorizontalDAG) -> None:
        pruned = lattice.prune()
        adj = pruned.adjacency_matrix()
        edges = pruned.edges
        for src, dst in edges:
            assert adj[src, dst]
        assert adj.sum() == pruned.n_edges


class TestNeighborsBatch:
    @pytest.mark.parametrize("idx", [74, 116, 179])
    def test_batch_agrees_with_single(self, lattice: HorizontalDAG, idx: int) -> None:
        expected = lattice.neighbors(idx)
        batch = lattice.neighbors_batch(np.array([idx]))
        np.testing.assert_array_equal(batch, expected)

    def test_batch_multi(self, lattice: HorizontalDAG) -> None:
        idxs = np.array([74, 116, 179])
        batch = lattice.neighbors_batch(idxs)
        expected = np.concatenate([lattice.neighbors(i) for i in idxs])
        np.testing.assert_array_equal(batch, expected)


class TestSampleEdges:
    def test_ptr_shape_and_monotonic(self, diamond: HorizontalDAG) -> None:
        _, _, _, edge_ptr = diamond.sample_edges(20000.0)
        assert edge_ptr.shape == (diamond.n_edges + 1,)
        assert edge_ptr[0] == 0
        assert np.all(np.diff(edge_ptr) >= 2)

    def test_output_lengths_consistent(self, diamond: HorizontalDAG) -> None:
        sample_lon, sample_lat, edge_idx, edge_ptr = diamond.sample_edges(20000.0)
        total = edge_ptr[-1]
        assert sample_lon.shape == (total,)
        assert sample_lat.shape == (total,)
        assert edge_idx.shape == (total,)

    def test_endpoints_match_nodes(self, diamond: HorizontalDAG) -> None:
        sample_lon, sample_lat, _, edge_ptr = diamond.sample_edges(20000.0)
        edge_src = diamond.edge_src
        for i in range(diamond.n_edges):
            s, e = edge_ptr[i], edge_ptr[i + 1]
            assert sample_lon[s] == pytest.approx(diamond.lon[edge_src[i]], abs=1e-6)
            assert sample_lat[s] == pytest.approx(diamond.lat[edge_src[i]], abs=1e-6)
            assert sample_lon[e - 1] == pytest.approx(diamond.lon[diamond.adj[i]], abs=1e-6)
            assert sample_lat[e - 1] == pytest.approx(diamond.lat[diamond.adj[i]], abs=1e-6)

    def test_spacing_respected(self, lattice: HorizontalDAG) -> None:
        pruned = lattice.prune()
        _, _, _, edge_ptr = pruned.sample_edges(spacing_m=50_000.0)
        n_per_edge = np.diff(edge_ptr)
        assert np.all(n_per_edge >= 2)

        # Longer edges should have more samples
        long_edges = pruned.edge_dist > 200_000.0
        short_edges = pruned.edge_dist < 100_000.0
        if long_edges.any() and short_edges.any():
            assert n_per_edge[long_edges].mean() > n_per_edge[short_edges].mean()


class TestFromPoisson:
    def test_builds_connected_dag(self) -> None:
        dag = HorizontalDAG.from_poisson(-118.4, 33.9, -73.8, 40.6).prune()
        assert dag.n_nodes > 10
        assert dag.n_edges > dag.n_nodes

    def test_origin_and_dest_reachable(self) -> None:
        dag = HorizontalDAG.from_poisson(-118.4, 33.9, -73.8, 40.6).prune()
        waves = list(dag.topo_wavefronts())
        all_nodes = set(np.concatenate(waves).tolist())
        assert dag.h_dest in all_nodes

    def test_edge_distances_reasonable(self) -> None:
        dag = HorizontalDAG.from_poisson(-118.4, 33.9, -73.8, 40.6).prune()
        assert np.all(dag.edge_dist > 0.0)
        assert np.all(dag.edge_dist < 600_000.0)


class TestAirportCoords:
    def test_from_icao(self) -> None:
        jfk = AirportCoords.from_icao("KJFK")
        assert jfk.icao_code == "KJFK"
        assert jfk.longitude == pytest.approx(-73.78, abs=1e-3)
        assert jfk.latitude == pytest.approx(40.64, abs=1e-3)
        assert jfk.elevation_ft == pytest.approx(13.0, abs=1.0)

    def test_from_icao_invalid(self) -> None:
        with pytest.raises(ValueError, match="Could not find airport"):
            AirportCoords.from_icao("ZZZZ")

    def test_coords_property(self) -> None:
        ac = AirportCoords(icao_code="TEST", longitude=-73.78, latitude=40.64, elevation_ft=13.0)
        assert ac.coords == (-73.78, 40.64)


class TestExpandNeighbors:
    def test_expand_single_node(self, diamond: HorizontalDAG) -> None:
        nodes = np.array([0])
        flat_nbr, flat_dist, src_idx, _ = diamond.expand_neighbors(nodes)
        expected_nbrs = diamond.neighbors(0)
        np.testing.assert_array_equal(np.sort(flat_nbr), np.sort(expected_nbrs))
        assert len(flat_dist) == len(expected_nbrs)
        assert np.all(flat_dist > 0.0)
        np.testing.assert_array_equal(src_idx, np.zeros(len(expected_nbrs), dtype=np.int64))

    def test_expand_multi_node(self, diamond: HorizontalDAG) -> None:
        nodes = np.array([0, 1])
        flat_nbr, _, src_idx, _ = diamond.expand_neighbors(nodes)
        # node 0 has 2 neighbors, node 1 has 1
        assert len(flat_nbr) == 3
        assert (src_idx == 0).sum() == 2
        assert (src_idx == 1).sum() == 1


@pytest.fixture
def mock_met() -> MetDataset:
    """Build a small synthetic MetDataset covering the diamond fixture."""
    lons = np.arange(-81.0, -74.0, 1.0)
    lats = np.arange(38.0, 42.0, 0.5)
    levels = np.array([250.0, 300.0, 350.0])
    times = pd.date_range("2024-01-01", periods=6, freq="h")

    shape = (len(lons), len(lats), len(levels), len(times))
    rng = np.random.default_rng(0)

    ds = xr.Dataset(
        {
            "air_temperature": (
                ["longitude", "latitude", "level", "time"],
                220.0 + rng.standard_normal(shape) * 5.0,
            ),
            "eastward_wind": (
                ["longitude", "latitude", "level", "time"],
                rng.standard_normal(shape) * 10.0,
            ),
            "northward_wind": (
                ["longitude", "latitude", "level", "time"],
                rng.standard_normal(shape) * 10.0,
            ),
        },
        coords={
            "longitude": lons,
            "latitude": lats,
            "level": levels,
            "time": times,
        },
    )
    return MetDataset(ds)


class TestEdgeMetLookup:
    def test_from_met(self, diamond: HorizontalDAG, mock_met: MetDataset) -> None:
        altitude_ft = np.array([28000.0, 32000.0])
        lookup = EdgeMetLookup.from_met(
            mock_met,
            diamond,
            altitude_ft=altitude_ft,
            takeoff_time=pd.Timestamp("2024-01-01"),
            flight_hours=4,
            spacing_m=50_000.0,
        )
        assert lookup.sample_lon.shape == lookup.sample_lat.shape
        assert lookup.cum_dist.shape == lookup.sample_lon.shape
        assert lookup.delta_dist.shape == lookup.sample_lon.shape
        assert lookup.sample_azimuth.shape == lookup.sample_lon.shape
        assert "air_temperature" in lookup.ds.data_vars

        # All interpolated met values should be finite
        for da in lookup.ds.values():
            assert da.notnull().all()

    def test_cum_dist_starts_at_zero(self, diamond: HorizontalDAG, mock_met: MetDataset) -> None:
        lookup = EdgeMetLookup.from_met(
            mock_met,
            diamond,
            altitude_ft=np.array([30000.0]),
            takeoff_time=pd.Timestamp("2024-01-01"),
            flight_hours=4,
            spacing_m=50_000.0,
        )
        # First sample of each edge should have cum_dist == 0
        edge_starts = lookup.edge_ptr[:-1]
        np.testing.assert_array_equal(lookup.cum_dist[edge_starts], 0.0)

    def test_delta_dist_is_diff_of_cum_dist(
        self, diamond: HorizontalDAG, mock_met: MetDataset
    ) -> None:
        lookup = EdgeMetLookup.from_met(
            mock_met,
            diamond,
            altitude_ft=np.array([30000.0]),
            takeoff_time=pd.Timestamp("2024-01-01"),
            flight_hours=4,
            spacing_m=50_000.0,
        )
        n_edges = len(lookup.edge_ptr) - 1
        for i in range(n_edges):
            s, e = lookup.edge_ptr[i], lookup.edge_ptr[i + 1]
            cd = lookup.cum_dist[s:e]
            dd = lookup.delta_dist[s:e]
            np.testing.assert_allclose(dd[:-1], np.diff(cd), atol=1e-6)
            assert dd[-1] == 0.0

    def test_repr(self, diamond: HorizontalDAG, mock_met: MetDataset) -> None:
        lookup = EdgeMetLookup.from_met(
            mock_met,
            diamond,
            altitude_ft=np.array([30000.0]),
            takeoff_time=pd.Timestamp("2024-01-01"),
            flight_hours=4,
            spacing_m=50_000.0,
        )
        r = repr(lookup)
        assert "EdgeMetLookup" in r
        assert "edges" in r
        assert "samples" in r

    def test_post_init_missing_var(self, diamond: HorizontalDAG, mock_met: MetDataset) -> None:
        lookup = EdgeMetLookup.from_met(
            mock_met,
            diamond,
            altitude_ft=np.array([30000.0]),
            takeoff_time=pd.Timestamp("2024-01-01"),
            flight_hours=4,
            spacing_m=50_000.0,
        )
        # Manually construct with missing variable
        bad_ds = lookup.ds.drop_vars("eastward_wind")
        with pytest.raises(ValueError, match="missing required variables"):
            EdgeMetLookup(
                ds=bad_ds,
                edge_ptr=lookup.edge_ptr,
                edge_idx=lookup.edge_idx,
                sample_lon=lookup.sample_lon,
                sample_lat=lookup.sample_lat,
                cum_dist=lookup.cum_dist,
                delta_dist=lookup.delta_dist,
                sample_azimuth=lookup.sample_azimuth,
            )

    def test_call_interpolation(self, diamond: HorizontalDAG, mock_met: MetDataset) -> None:
        lookup = EdgeMetLookup.from_met(
            mock_met,
            diamond,
            altitude_ft=np.array([30000.0]),
            takeoff_time=pd.Timestamp("2024-01-01"),
            flight_hours=4,
            spacing_m=50_000.0,
        )
        # Query the first 3 samples at a time between two hourly steps
        idxs = np.array([0, 1, 2])
        times = np.full((3, 1), np.datetime64("2024-01-01T01:30", "ns"))
        result = lookup(idxs, times)
        assert result.air_temperature.shape == (3, 1)  # (n_sample, n_fl)
        assert result.eastward_wind.shape == (3, 1)
        assert result.northward_wind.shape == (3, 1)
        assert np.all(np.isfinite(result.air_temperature))

    def test_call_at_exact_time(self, diamond: HorizontalDAG, mock_met: MetDataset) -> None:
        lookup = EdgeMetLookup.from_met(
            mock_met,
            diamond,
            altitude_ft=np.array([30000.0]),
            takeoff_time=pd.Timestamp("2024-01-01"),
            flight_hours=4,
            spacing_m=50_000.0,
        )
        idxs = np.array([0])
        times = np.full((1, 1), np.datetime64("2024-01-01T00:00", "ns"))
        result = lookup(idxs, times)
        assert result.air_temperature.shape == (1, 1)
        assert np.all(np.isfinite(result.air_temperature))

    def test_call_all_samples(self, diamond: HorizontalDAG, mock_met: MetDataset) -> None:
        lookup = EdgeMetLookup.from_met(
            mock_met,
            diamond,
            altitude_ft=np.array([28000.0, 32000.0]),
            takeoff_time=pd.Timestamp("2024-01-01"),
            flight_hours=4,
            spacing_m=50_000.0,
        )
        n_samples = len(lookup.sample_lon)
        n_fl = lookup.ds.sizes["altitude_ft"]
        idxs = np.arange(n_samples)
        times = np.full((n_samples, n_fl), np.datetime64("2024-01-01T01:30", "ns"))
        result = lookup(idxs, times)
        assert result.air_temperature.shape == (n_samples, n_fl)
        assert np.all(np.isfinite(result.air_temperature))
        assert np.all(np.isfinite(result.eastward_wind))
        assert np.all(np.isfinite(result.northward_wind))

    def test_call_boundary_times(self, diamond: HorizontalDAG, mock_met: MetDataset) -> None:
        # Query at the first and last available met time steps
        lookup = EdgeMetLookup.from_met(
            mock_met,
            diamond,
            altitude_ft=np.array([30000.0]),
            takeoff_time=pd.Timestamp("2024-01-01"),
            flight_hours=4,
            spacing_m=50_000.0,
        )
        idxs = np.array([0, 0])
        first_time = lookup.ds["time"].values[0]
        last_time = lookup.ds["time"].values[-1]
        times = np.array([[first_time], [last_time]])
        result = lookup(idxs, times)
        assert result.air_temperature.shape == (2, 1)
        assert np.all(np.isfinite(result.air_temperature))


class TestReverse:
    def test_edge_count_preserved(self, diamond: HorizontalDAG) -> None:
        rev = diamond.reverse()
        assert rev.n_edges == diamond.n_edges
        assert rev.n_nodes == diamond.n_nodes

    def test_origin_dest_swapped(self, diamond: HorizontalDAG) -> None:
        rev = diamond.reverse()
        assert rev.h_origin == diamond.h_dest
        assert rev.h_dest == diamond.h_origin

    def test_edges_flipped(self, diamond: HorizontalDAG) -> None:
        orig_edges = set(map(tuple, diamond.edges.tolist()))
        rev_edges = set(map(tuple, diamond.reverse().edges.tolist()))
        flipped = {(v, u) for u, v in orig_edges}
        assert rev_edges == flipped

    def test_edge_distances_preserved(self, diamond: HorizontalDAG) -> None:
        orig_dists = sorted(diamond.edge_dist.tolist())
        rev_dists = sorted(diamond.reverse().edge_dist.tolist())
        np.testing.assert_allclose(orig_dists, rev_dists)

    def test_double_reverse_recovers_original(self, lattice: HorizontalDAG) -> None:
        pruned = lattice.prune()
        recovered = pruned.reverse().reverse()
        assert recovered.n_nodes == pruned.n_nodes
        assert recovered.n_edges == pruned.n_edges
        assert recovered.h_origin == pruned.h_origin
        assert recovered.h_dest == pruned.h_dest
        orig_edges = set(map(tuple, pruned.edges.tolist()))
        rec_edges = set(map(tuple, recovered.edges.tolist()))
        assert orig_edges == rec_edges

    def test_reverse_lattice_connectivity(self, lattice: HorizontalDAG) -> None:
        pruned = lattice.prune()
        rev = pruned.reverse()
        # In the reversed DAG, dest (now origin) should reach all nodes
        adj = rev.adjacency_matrix()
        reachable = np.zeros(rev.n_nodes, dtype=bool)
        reachable[rev.h_origin] = True
        for _ in range(rev.n_nodes):
            reachable |= adj[reachable].any(axis=0)
        assert reachable.all()


class TestOutDegree:
    def test_out_degree_shape(self, diamond: HorizontalDAG) -> None:
        assert diamond.out_degree.shape == (diamond.n_nodes,)

    def test_out_degree_sums_to_edges(self, diamond: HorizontalDAG) -> None:
        assert diamond.out_degree.sum() == diamond.n_edges

    def test_out_degree_values(self, diamond: HorizontalDAG) -> None:
        # Node 0 -> {1, 2}, node 1 -> {3}, node 2 -> {3}, node 3 -> {}
        np.testing.assert_array_equal(diamond.out_degree, [2, 1, 1, 0])

    def test_out_degree_lattice(self, lattice: HorizontalDAG) -> None:
        pruned = lattice.prune()
        assert pruned.out_degree.sum() == pruned.n_edges
        assert pruned.out_degree[pruned.h_dest] == 0
        assert pruned.out_degree[pruned.h_origin] > 0


class TestDisconnectedPrune:
    def test_prune_disconnected_graph(self) -> None:
        # Origin cannot reach dest — prune should return only shared reachable nodes
        lon = np.array([0.0, 1.0, 10.0, 11.0])
        lat = np.array([0.0, 0.0, 0.0, 0.0])

        # Manually build a DAG where 0->1 and 2->3 but no path from 0 to 3
        adj_ptr = np.array([0, 1, 1, 2, 2], dtype=np.int64)
        adj = np.array([1, 3], dtype=np.int64)
        edge_dist = np.array([100_000.0, 100_000.0])
        dag = HorizontalDAG(
            lon=lon,
            lat=lat,
            adj_ptr=adj_ptr,
            adj=adj,
            edge_dist=edge_dist,
            h_origin=0,
            h_dest=3,
        )

        pruned = dag.prune()
        # Output is completely empty
        assert pruned.n_nodes == 0
        assert pruned.n_edges == 0
