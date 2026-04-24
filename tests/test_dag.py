"""Tests for the HorizontalDAG class in contrailopt.dag."""

import numpy as np
import pytest

from contrailopt import HorizontalDAG


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
        """Every edge points strictly eastward on the lattice."""
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


class TestSampleEdges:
    def test_ptr_shape_and_monotonic(self, diamond: HorizontalDAG) -> None:
        _, _, _, edge_ptr = diamond.sample_edges()
        assert edge_ptr.shape == (diamond.n_edges + 1,)
        assert edge_ptr[0] == 0
        assert np.all(np.diff(edge_ptr) >= 2)

    def test_output_lengths_consistent(self, diamond: HorizontalDAG) -> None:
        sample_lon, sample_lat, edge_idx, edge_ptr = diamond.sample_edges()
        total = edge_ptr[-1]
        assert sample_lon.shape == (total,)
        assert sample_lat.shape == (total,)
        assert edge_idx.shape == (total,)

    def test_endpoints_match_nodes(self, diamond: HorizontalDAG) -> None:
        sample_lon, sample_lat, _, edge_ptr = diamond.sample_edges()
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
