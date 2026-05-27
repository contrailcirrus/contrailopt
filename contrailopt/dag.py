"""Utilities for horizontal DAG construction."""

import warnings
from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pycontrails import Flight, MetDataset
from pycontrails.core import airports
from pycontrails.physics import geo, units

from contrailopt.slerp import gc_interp, gc_npts, spherical_fwd

if TYPE_CHECKING:
    from cartopy.mpl.geoaxes import GeoAxes


@dataclass(kw_only=True, slots=True, frozen=True)
class AirportCoords:
    """Coordinates and elevation of an airport, identified by ICAO code."""

    icao_code: str
    longitude: float
    latitude: float
    elevation_ft: float

    @classmethod
    def from_icao(cls, icao_code: str) -> Self:
        """Look up airport coordinates by ICAO code."""
        airports_df = airports.global_airport_database()

        row = airports_df.query(f"icao_code == '{icao_code}'")
        if row.empty:
            raise ValueError(f"Could not find airport with ICAO code {icao_code}")

        return cls(
            icao_code=icao_code,
            longitude=row["longitude"].item(),
            latitude=row["latitude"].item(),
            elevation_ft=row["elevation_ft"].item(),
        )

    @property
    def coords(self) -> tuple[float, float]:
        """Return (longitude, latitude) coordinates as a tuple."""
        return self.longitude, self.latitude


def _csr_flat_pos(
    adj_ptr: npt.NDArray[np.int64],
    nodes: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return flat indices into CSR data arrays for a batch of row nodes.

    Returns ``(flat_pos, lengths)``, where flat_pos indexes into adj
    and lengths[i] is the number of entries for nodes[i].
    """
    starts = adj_ptr[nodes]
    lengths = adj_ptr[nodes + 1] - starts
    flat_starts = np.repeat(starts, lengths)
    offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    flat_pos = flat_starts + offsets
    return flat_pos, lengths


def _neighbors_batch(
    adj_ptr: npt.NDArray[np.int64],
    adj: npt.NDArray[np.int64],
    nodes: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Determine neighbors (duplicates included with multiplicity) for a batch of nodes."""
    flat_pos, _ = _csr_flat_pos(adj_ptr, nodes)
    return adj[flat_pos]


def _reachability(
    seed: int,
    n_nodes: int,
    adj_ptr: npt.NDArray[np.int64],
    adj: npt.NDArray[np.int64],
) -> npt.NDArray[np.bool_]:
    """Return boolean array of nodes reachable from seed via adjacency."""
    seen = np.zeros(n_nodes, dtype=bool)
    frontier = np.array([seed], dtype=np.int64)
    seen[frontier] = True

    while frontier.size:
        nxt = _neighbors_batch(adj_ptr, adj, frontier)  # all outgoing neighbors of current frontier
        nxt = np.unique(nxt)  # dedupe duplicates from shared parents
        frontier = nxt[~seen[nxt]]  # only newly discovered nodes
        seen[frontier] = True

    return seen


def _reverse_csr(
    adj: npt.NDArray[np.int64],
    n_nodes: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Compute reverse CSR pointer and edge permutation order."""
    order = np.argsort(adj)

    rev_ptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.add.at(rev_ptr[1:], adj, 1)
    np.cumsum(rev_ptr, out=rev_ptr)

    return rev_ptr, order


@dataclass(kw_only=True, slots=True, frozen=True)
class HorizontalDAG:
    """Directed graph on (lon, lat) nodes with CSR adjacency.

    All geometry (distances, azimuths, interpolation, polygon exclusion) is
    computed on a sphere, not on a planar lon/lat grid.
    """

    #: Longitude of each node in degrees ``(n,)``. Assumed to be in the range [-180, 180).
    #: (This assumption is used in :meth:`crosses_antimeridian`).
    lon: npt.NDArray[np.floating]

    #: Latitude of each node in degrees ``(n,)``. Assumed to be in the range [-90, 90].
    lat: npt.NDArray[np.floating]

    #: CSR row pointers ``(n + 1,)``.
    #: Neighbors of node ``i`` are ``adj[adj_ptr[i]: adj_ptr[i+1]]``.
    adj_ptr: npt.NDArray[np.int64]

    #: Neighbor (destination) indices for each directed edge ``(m,)``.
    adj: npt.NDArray[np.int64]

    #: Great-circle distance in meters for each directed edge ``(m,)``.
    edge_dist: npt.NDArray[np.floating]

    #: Index of the distinguished origin node.
    h_origin: int

    #: Index of the distinguished destination node.
    h_dest: int

    def __repr__(self) -> str:
        name = type(self).__name__
        return f"{name}({self.n_nodes} nodes, {self.n_edges} edges)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HorizontalDAG):
            # https://docs.python.org/3/reference/datamodel.html#object.__eq__
            return NotImplemented
        return (
            self.h_origin == other.h_origin
            and self.h_dest == other.h_dest
            and np.array_equal(self.lon, other.lon)
            and np.array_equal(self.lat, other.lat)
            and np.array_equal(self.adj_ptr, other.adj_ptr)
            and np.array_equal(self.adj, other.adj)
            and np.array_equal(self.edge_dist, other.edge_dist)
        )

    @property
    def n_nodes(self) -> int:
        """The number of nodes in the graph."""
        return len(self.lon)

    @property
    def n_edges(self) -> int:
        """The number of directed edges in the graph."""
        return len(self.adj)

    @property
    def out_degree(self) -> npt.NDArray[np.int64]:
        """The out-degree of each node."""
        return np.diff(self.adj_ptr)

    @property
    def edge_src(self) -> npt.NDArray[np.int64]:
        """The source endpoint index of each directed edge."""
        return np.repeat(np.arange(self.n_nodes, dtype=np.int64), self.out_degree)

    @property
    def edges(self) -> npt.NDArray[np.int64]:
        """The directed edges of the graph as (src, dest) index pairs in an ``(m, 2)`` array."""
        return np.column_stack([self.edge_src, self.adj])

    @property
    def crosses_antimeridian(self) -> bool:
        """Determine if the great circle between origin and destination crosses the antimeridian."""
        return abs(self.lon[self.h_origin].item() - self.lon[self.h_dest].item()) > 180.0

    def neighbors(self, i: int) -> npt.NDArray[np.int64]:
        """Return the neighbors of a specified node."""
        return self.adj[self.adj_ptr[i] : self.adj_ptr[i + 1]]

    def edge_index(self, src: int, dst: int) -> int:
        """Return the CSR index of the directed edge (src, dst).

        Performs a linear scan over the neighbors of ``src``. This could be
        replaced with ``np.searchsorted`` if needed, but that would require
        enforcing sorted neighbors during construction.

        Raises
        ------
        ValueError
            If no edge from ``src`` to ``dst`` exists.
        """
        start = self.adj_ptr[src].item()
        end = self.adj_ptr[src + 1]
        row = self.adj[start:end]  # row = self.neighbors(src), but we need start again

        pos = np.flatnonzero(row == dst)
        if len(pos) == 0:
            raise ValueError(f"No edge from {src} to {dst}")
        return start + pos.item()

    def neighbors_batch(self, nodes: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """Return neighbors (duplicates included with multiplicity) for a batch of nodes."""
        return _neighbors_batch(self.adj_ptr, self.adj, nodes)

    def expand_neighbors(
        self, nodes: npt.NDArray[np.int64]
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.floating],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ]:
        """Expand CSR adjacency for a batch of nodes into flat edge arrays.

        Returns
        -------
        flat_nbr : npt.NDArray[np.int64]
            ``(e,)`` neighbor indices for all edges leaving ``nodes``.
        flat_dist : npt.NDArray[np.floating]
            ``(e,)`` edge distances in meters.
        src_idx : npt.NDArray[np.int64]
            ``(e,)`` index into ``nodes`` for each flat entry, so
            ``nodes[src_idx[k]]`` is the source node of flat edge ``k``.
        flat_edge_idx : npt.NDArray[np.int64]
            ``(e,)`` index of each edge in the CSR arrays (``adj``, ``edge_dist``).

        Here ``e = out_degree[nodes].sum()``, the total number of outgoing
        edges from all ``nodes``.
        """
        flat_pos, lengths = _csr_flat_pos(self.adj_ptr, nodes)
        src_idx = np.repeat(np.arange(len(nodes)), lengths)
        return self.adj[flat_pos], self.edge_dist[flat_pos], src_idx, flat_pos

    def adjacency_matrix(self) -> npt.NDArray[np.bool]:
        """Return dense boolean adjacency matrix A where A[i, j] is True for i->j."""
        matrix = np.zeros((self.n_nodes, self.n_nodes), dtype=bool)
        matrix[self.edge_src, self.adj] = True
        return matrix

    def reverse(self) -> Self:
        """Return a new DAG with all edge directions flipped and origin/dest swapped."""
        src = self.edge_src
        rev_ptr, rev_order = _reverse_csr(self.adj, self.n_nodes)

        rev_adj = src[rev_order]
        rev_edge_dist = self.edge_dist[rev_order]

        return type(self)(
            lon=self.lon,
            lat=self.lat,
            adj_ptr=rev_ptr,
            adj=rev_adj,
            edge_dist=rev_edge_dist,
            h_origin=self.h_dest,
            h_dest=self.h_origin,
        )

    def prune_unreachable(self) -> Self:
        """Return a new DAG with only nodes reachable from origin that also reach dest."""
        src = self.edge_src
        fwd = _reachability(self.h_origin, self.n_nodes, self.adj_ptr, self.adj)

        rev_ptr, rev_order = _reverse_csr(self.adj, self.n_nodes)
        rev_adj = src[rev_order]
        bwd = _reachability(self.h_dest, self.n_nodes, rev_ptr, rev_adj)

        live = fwd & bwd

        if not live[self.h_origin] or not live[self.h_dest]:
            raise ValueError("Origin or destination became unreachable after pruning")

        # Remap node indices
        n = live.sum()
        new_idx = np.full(self.n_nodes, -1, dtype=np.int64)
        new_idx[live] = np.arange(n, dtype=np.int64)

        # Filter edges and distances via CSR ordering
        mask = live[src] & live[self.adj]
        new_src = new_idx[src[mask]]
        new_dst = new_idx[self.adj[mask]]
        new_edge_dist = self.edge_dist[mask]

        # Build new CSR (already sorted by source from parent CSR)
        adj_ptr = np.zeros(n + 1, dtype=np.int64)
        np.add.at(adj_ptr[1:], new_src, 1)
        np.cumsum(adj_ptr, out=adj_ptr)

        return type(self)(
            lon=self.lon[live],
            lat=self.lat[live],
            adj_ptr=adj_ptr,
            adj=new_dst,
            edge_dist=new_edge_dist,
            h_origin=new_idx[self.h_origin].item(),
            h_dest=new_idx[self.h_dest].item(),
        )

    def prune_edges(self, degree: int) -> Self:
        """Keep the best ``degree`` outgoing and incoming edges per node.

        Each edge is scored by the sum of its azimuth deviations: how far the
        edge direction deviates from the azimuth toward the destination (at
        the tail) plus how far the reverse deviates from the azimuth toward
        the origin (at the head). An edge is kept if it ranks among the best
        ``degree`` outgoing edges of its source or among the best ``degree``
        incoming edges of its destination.

        Parameters
        ----------
        degree : int
            Number of outgoing and incoming edges to keep per node.

        Returns
        -------
        Self
            A new DAG with at most ``degree`` outgoing and incoming edges per node.
        """
        # Compute azimuth deviation scores for each edge
        src = self.edge_src
        az_to_dest = geo.azimuth(self.lon, self.lat, self.lon[self.h_dest], self.lat[self.h_dest])
        az_to_origin = geo.azimuth(
            self.lon, self.lat, self.lon[self.h_origin], self.lat[self.h_origin]
        )
        az_edge = geo.azimuth(self.lon[src], self.lat[src], self.lon[self.adj], self.lat[self.adj])
        az_rev = geo.azimuth(self.lon[self.adj], self.lat[self.adj], self.lon[src], self.lat[src])

        delta_tail = np.abs((az_edge - az_to_dest[src] + 180.0) % 360.0 - 180.0)
        delta_head = np.abs((az_rev - az_to_origin[self.adj] + 180.0) % 360.0 - 180.0)
        score = delta_tail + delta_head  # many other variations also work: max, min, p-weighted

        # Rank each edge among its source node's outgoing edges by score
        out_order = np.lexsort((score, src))
        out_rank = np.empty(self.n_edges, dtype=np.int64)
        out_rank[out_order] = np.arange(self.n_edges) - self.adj_ptr[src[out_order]]

        # Rank each edge among its dest node's incoming edges by score
        dst = self.adj
        rev_ptr, _ = _reverse_csr(dst, self.n_nodes)
        in_order = np.lexsort((score, dst))
        in_rank = np.empty(self.n_edges, dtype=np.int64)
        in_rank[in_order] = np.arange(self.n_edges) - rev_ptr[dst[in_order]]

        keep = (out_rank < degree) | (in_rank < degree)

        # Construct new CSR to return
        kept_src = src[keep]
        kept_dst = self.adj[keep]
        kept_dist = self.edge_dist[keep]

        adj_ptr = np.zeros(self.n_nodes + 1, dtype=np.int64)
        np.add.at(adj_ptr[1:], kept_src, 1)
        np.cumsum(adj_ptr, out=adj_ptr)

        return type(self)(
            lon=self.lon,
            lat=self.lat,
            adj_ptr=adj_ptr,
            adj=kept_dst,
            edge_dist=kept_dist,
            h_origin=self.h_origin,
            h_dest=self.h_dest,
        )

    def exclude_polygons(self, polygons: list[list[tuple[float, float]]]) -> Self:
        """Return a new DAG with edges crossing any polygon removed.

        Uses spherely for geodesic intersection tests on the sphere.

        Parameters
        ----------
        polygons : list[list[tuple[float, float]]]
            List of polygons, where each polygon is a list of ``(lon, lat)`` vertices.

        Returns
        -------
        HorizontalDAG
            A new DAG with offending edges removed and then pruned.
        """
        import spherely

        src = self.edge_src

        # Build a linestring for every edge
        linestrings = [
            spherely.create_linestring([(self.lon[s], self.lat[s]), (self.lon[d], self.lat[d])])
            for s, d in zip(src, self.adj, strict=True)
        ]
        edge_geoms = np.array(linestrings)

        # Test each polygon against all edges. We could also take a union of all polygons
        # and test once if this becomes a bottleneck
        excluded = np.zeros(self.n_edges, dtype=bool)
        for coords in polygons:
            poly = spherely.create_polygon(coords)
            excluded |= spherely.intersects(poly, edge_geoms)

        # Build filtered edge list
        keep = ~excluded
        kept_src = src[keep]
        kept_dst = self.adj[keep]
        kept_dist = self.edge_dist[keep]

        # Rebuild CSR
        adj_ptr = np.zeros(self.n_nodes + 1, dtype=np.int64)
        np.add.at(adj_ptr[1:], kept_src, 1)
        np.cumsum(adj_ptr, out=adj_ptr)

        return type(self)(
            lon=self.lon,
            lat=self.lat,
            adj_ptr=adj_ptr,
            adj=kept_dst,
            edge_dist=kept_dist,
            h_origin=self.h_origin,
            h_dest=self.h_dest,
        ).prune_unreachable()

    def sample_edges(
        self, spacing_m: float
    ) -> tuple[
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ]:
        """Sample points along every edge at most ``spacing_m`` meters apart.

        Points are uniformly spaced along each edge, and both edge endpoints
        (source and destination nodes) are included as samples.

        Returns
        -------
        sample_lon : npt.NDArray[np.floating]
            ``(s,)`` longitude of each sample point.
        sample_lat : npt.NDArray[np.floating]
            ``(s,)`` latitude of each sample point.
        edge_idx : npt.NDArray[np.int64]
            ``(s,)`` edge index for each sample point.
        edge_ptr : npt.NDArray[np.int64]
            ``(m + 1,)`` CSR-style pointer so edge ``i``'s samples are at
            ``sample_lon[edge_ptr[i]: edge_ptr[i+1]]``.

        Here ``s = edge_ptr[-1]``, the total number of sample points across all edges.
        """
        src = self.edge_src
        src_lon = self.lon[src]
        src_lat = self.lat[src]
        dst_lon = self.lon[self.adj]
        dst_lat = self.lat[self.adj]

        dist = geo.haversine(src_lon, src_lat, dst_lon, dst_lat)

        n_samples = np.maximum(np.ceil(dist / spacing_m).astype(int) + 1, 2)
        edge_ptr = np.zeros(self.n_edges + 1, dtype=np.int64)
        np.cumsum(n_samples, out=edge_ptr[1:])
        total = edge_ptr[-1]

        local_idx = np.arange(total) - np.repeat(edge_ptr[:-1], n_samples)
        frac = local_idx / np.repeat(n_samples - 1, n_samples)

        sample_lon, sample_lat = gc_interp(
            np.repeat(src_lon, n_samples),
            np.repeat(src_lat, n_samples),
            np.repeat(dst_lon, n_samples),
            np.repeat(dst_lat, n_samples),
            frac.astype(src_lon.dtype),
        )

        edge_idx = np.repeat(np.arange(self.n_edges), n_samples)
        return sample_lon, sample_lat, edge_idx, edge_ptr

    def plot(self, ax: "GeoAxes | None" = None, linewidth: float = 2.0) -> "GeoAxes":
        """Plot the DAG on a cartopy map."""
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        if ax is None:
            central_lon = 180.0 if self.crosses_antimeridian else 0.0
            proj = ccrs.PlateCarree(central_longitude=central_lon)
            _, ax = plt.subplots(figsize=(20, 10), subplot_kw={"projection": proj})

            plot_lon = (self.lon - central_lon + 180.0) % 360.0 - 180.0
            ax.set_extent(
                [
                    plot_lon.min() - 2.0,
                    plot_lon.max() + 2.0,
                    self.lat.min() - 2.0,
                    self.lat.max() + 2.0,
                ],
                crs=proj,
            )
        else:
            proj = ax.projection
            central_lon = proj.proj4_params.get("lon_0", 0.0)
            plot_lon = (self.lon - central_lon + 180.0) % 360.0 - 180.0

        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray")
        ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor="gray")

        # Draw edges
        edge_src = self.edge_src
        segments = np.stack(
            [
                np.column_stack([plot_lon[edge_src], self.lat[edge_src]]),
                np.column_stack([plot_lon[self.adj], self.lat[self.adj]]),
            ],
            axis=1,
        )
        lc = LineCollection(
            segments,
            colors="steelblue",
            linewidths=linewidth,
            alpha=0.2,
            transform=proj,
        )
        ax.add_collection(lc)

        # Draw nodes
        ax.scatter(plot_lon, self.lat, s=2, color="black", transform=proj, zorder=5)
        ax.plot(
            plot_lon[self.h_origin],
            self.lat[self.h_origin],
            "ro",
            markersize=8,
            transform=proj,
            zorder=10,
        )
        ax.plot(
            plot_lon[self.h_dest],
            self.lat[self.h_dest],
            "go",
            markersize=8,
            transform=proj,
            zorder=10,
        )

        ax.set_title(f"{self.n_edges} edges, {self.n_nodes} nodes")
        return ax

    @classmethod
    def from_points(
        cls,
        lon: npt.NDArray[np.floating],
        lat: npt.NDArray[np.floating],
        origin_idx: int = 0,
        dest_idx: int = -1,
        max_angle_deg: float = 40.0,
        max_dist_m: float = 500_000.0,
    ) -> Self:
        """Build a DAG from lon/lat arrays using the dual azimuth constraint."""
        edges, dists = _dual_az_edges(lon, lat, origin_idx, dest_idx, max_angle_deg, max_dist_m)

        n = len(lon)
        order = np.argsort(edges[:, 0])
        adj_ptr = np.zeros(n + 1, dtype=np.int64)
        np.add.at(adj_ptr[1:], edges[order, 0], 1)
        np.cumsum(adj_ptr, out=adj_ptr)

        return cls(
            lon=lon,
            lat=lat,
            adj_ptr=adj_ptr,
            adj=edges[order, 1],
            edge_dist=dists[order],
            h_origin=origin_idx if origin_idx >= 0 else n + origin_idx,
            h_dest=dest_idx if dest_idx >= 0 else n + dest_idx,
        )

    @classmethod
    def from_poisson(
        cls,
        origin_lon: float,
        origin_lat: float,
        dest_lon: float,
        dest_lat: float,
        poisson_spacing_m: float = 80_000.0,
        max_cross_track: float | None = None,
        max_angle_deg: float = 40.0,
        max_dist_m: float = 500_000.0,
        dtype: type[np.floating] = np.float64,
        rng: np.random.Generator | None = None,
    ) -> Self:
        """Build a DAG from Poisson-disk sampled points along the OD great circle."""
        from scipy.stats.qmc import PoissonDisk

        gs_distance = geo.haversine(origin_lon, origin_lat, dest_lon, dest_lat).item()

        if max_cross_track is None:
            max_cross_track = min(1_000_000.0, gs_distance / 4.0)

        nx = int(gs_distance / 100_000.0) + 1

        # Great circle spine
        gc_lons, gc_lats = gc_npts(origin_lon, origin_lat, dest_lon, dest_lat, nx - 2)
        gc_lons = np.concatenate([[origin_lon], gc_lons, [dest_lon]])
        gc_lats = np.concatenate([[origin_lat], gc_lats, [dest_lat]])

        # Perpendicular azimuths along spine
        az_fwd = geo.azimuth(gc_lons[:-1], gc_lats[:-1], gc_lons[1:], gc_lats[1:])
        az_perp = np.empty(nx)
        az_perp[:-1] = az_fwd + 90.0
        az_perp[-1] = az_perp[-2]

        # Poisson disk sampling with physically uniform spacing.
        # The corridor is gs_distance x 2*max_cross_track in physical space.
        # We sample in [0, aspect] x [0, 1] so the radius is isotropic in
        # physical units, then normalize the along-track coordinate.
        corridor_width = 2.0 * max_cross_track
        aspect = gs_distance / corridor_width
        unit_radius = poisson_spacing_m / corridor_width

        poisson_disk = PoissonDisk(
            d=2,
            radius=unit_radius,
            l_bounds=[0, 0],
            u_bounds=[aspect, 1],
            rng=rng,
        )
        pts = poisson_disk.fill_space()
        t = pts[:, 0] / aspect
        cross = pts[:, 1] * corridor_width - max_cross_track

        # Prepend origin and append dest
        t = np.concatenate([[0.0], t, [1.0]])
        cross = np.concatenate([[0.0], cross, [0.0]], dtype=dtype)

        # Project to (lon, lat) dealing with potential antimeridian crossing via unwrapping.
        t_gc = np.linspace(0.0, 1.0, nx)
        gc_lons_unwrap = np.unwrap(gc_lons, period=360.0)
        az_perp_unwrap = np.unwrap(az_perp, period=360.0)
        lon_base = np.interp(t, t_gc, gc_lons_unwrap).astype(dtype)
        lon_base = (lon_base + 180.0) % 360.0 - 180.0
        lat_base = np.interp(t, t_gc, gc_lats).astype(dtype)
        az_base = np.interp(t, t_gc, az_perp_unwrap).astype(dtype)
        lon, lat = spherical_fwd(lon_base, lat_base, az_base, cross)
        lon = (lon + 180.0) % 360.0 - 180.0

        return cls.from_points(
            lon,
            lat,
            origin_idx=0,
            dest_idx=len(lon) - 1,
            max_angle_deg=max_angle_deg,
            max_dist_m=max_dist_m,
        )

    @classmethod
    def from_flight(
        cls,
        flight: Flight,
        max_dist_m: float = 500_000.0,
    ) -> Self:
        """Build a DAG from a ``pycontrails.Flight`` trajectory.

        Waypoint *i* is connected to waypoint *j* iff *j* is strictly forward
        in time and within ``max_dist_m`` great-circle distance of *i*.

        Parameters
        ----------
        flight : Flight
            A flight trajectory with longitude, latitude, and time columns.
        max_dist_m : float
            Maximum great-circle distance in meters for an edge.

        Returns
        -------
        HorizontalDAG
            A DAG whose nodes are the flight waypoints, with the first
            waypoint as origin and the last as destination.
        """
        lon = flight["longitude"]
        lat = flight["latitude"]
        time = flight["time"]
        n = len(lon)

        # Compute full 2D pairwise distance matrix, could be smarter here if needed
        dist = geo.haversine(
            lon[:, np.newaxis],
            lat[:, np.newaxis],
            lon[np.newaxis, :],
            lat[np.newaxis, :],
        )

        dist_filt = dist <= max_dist_m
        time_filt = time[np.newaxis, :] > time[:, np.newaxis]
        tail, head = np.nonzero(dist_filt & time_filt)

        edge_dist = dist[tail, head]
        order = np.argsort(tail)
        tail = tail[order]
        head = head[order]
        edge_dist = edge_dist[order]

        adj_ptr = np.zeros(n + 1, dtype=np.int64)
        np.add.at(adj_ptr[1:], tail, 1)
        np.cumsum(adj_ptr, out=adj_ptr)

        return cls(
            lon=lon,
            lat=lat,
            adj_ptr=adj_ptr,
            adj=head,
            edge_dist=edge_dist,
            h_origin=0,
            h_dest=n - 1,
        )

    def topo_wavefronts(self) -> Generator[npt.NDArray[np.int64], None, None]:
        """Yield topological wavefronts reachable from origin.

        Only nodes reachable from ``h_origin`` are emitted. Unreachable nodes
        (those with incoming edges from outside the reachable subgraph) are
        excluded.

        The first wavefront contains only the origin. Wavefront *k* contains
        nodes whose reachable in-degree drops to zero after removing wavefronts
        0 ... k-1.
        """
        reachable = _reachability(self.h_origin, self.n_nodes, self.adj_ptr, self.adj)
        live_edges = reachable[self.edge_src]

        in_degree = np.zeros(self.n_nodes, dtype=np.int64)
        np.add.at(in_degree, self.adj[live_edges], 1)

        wave_nodes = np.array([self.h_origin])
        while wave_nodes.size:
            yield wave_nodes

            neighbors = self.neighbors_batch(wave_nodes)
            np.subtract.at(in_degree, neighbors, 1)
            candidates = np.unique(neighbors)
            filt = in_degree[candidates] == 0
            wave_nodes = candidates[filt]


@dataclass(kw_only=True, slots=True, frozen=True)
class EdgeInterpolation:
    """Met fields interpolated at sample points."""

    air_temperature: npt.NDArray[np.floating]
    eastward_wind: npt.NDArray[np.floating]
    northward_wind: npt.NDArray[np.floating]
    eef_per_m: npt.NDArray[np.floating] | None


@dataclass(kw_only=True, slots=True, frozen=True)
class EdgeMetLookup:
    """Pre-interpolated met data on edge sample points."""

    #: ``xr.Dataset`` with dims ``(sample, altitude_ft, time)`` containing
    #: weather variables interpolated onto edge sample coordinates.
    ds: xr.Dataset

    #: CSR-style pointer array ``(n_edges + 1,)``. Samples for edge ``i``
    #: are at indices ``edge_ptr[i]:edge_ptr[i+1]``.
    edge_ptr: npt.NDArray[np.int64]

    #: Edge index for each sample point ``(n_samples,)``.
    edge_idx: npt.NDArray[np.int64]

    #: Longitude of each sample point ``(n_samples,)``.
    sample_lon: npt.NDArray[np.floating]

    #: Latitude of each sample point ``(n_samples,)``.
    sample_lat: npt.NDArray[np.floating]

    #: Cumulative distance from edge source to each sample point in meters ``(n_samples,)``.
    cum_dist: npt.NDArray[np.floating]

    #: Distance in meters from this sample to the next ``(n_samples,)``.
    #: The last sample of each edge has ``delta_dist = 0``.
    #: Equal to ``diff(cum_dist)`` within each edge.
    delta_dist: npt.NDArray[np.floating]

    #: Azimuth in radians from each sample to the next ``(n_samples,)``.
    #: The last sample of each edge copies the previous sample's azimuth.
    sample_azimuth: npt.NDArray[np.floating]

    def __post_init__(self) -> None:
        required = {"air_temperature", "eastward_wind", "northward_wind"}
        missing = required - set(self.ds)
        if missing:
            raise ValueError(f"Met dataset missing required variables: {missing}")

    def __repr__(self) -> str:
        n_samples = len(self.edge_idx)
        n_edges = len(self.edge_ptr) - 1
        n_fl = self.ds.sizes["altitude_ft"]
        n_time = self.ds.sizes["time"]
        name = type(self).__name__
        return f"{name}({n_edges} edges, {n_samples} samples, {n_fl} FLs, {n_time} time steps)"

    def __call__(
        self,
        sample_idxs: npt.NDArray[np.int64],
        times: npt.NDArray[np.datetime64],
        fl_idx: npt.NDArray[np.int64] | None = None,
    ) -> EdgeInterpolation:
        """Interpolate all variables at given sample indices and times.

        Parameters
        ----------
        sample_idxs : npt.NDArray[np.int64]
            1D array of sample indices to query.
        times : npt.NDArray[np.datetime64]
            2D array of time coordinates with shape ``(n_sample, n_fl)``, where
            ``n_sample = len(sample_idxs)``. Each FL gets its own query time
            (e.g. to account for FL-dependent climb duration).
        fl_idx : npt.NDArray[np.int64] | None
            Flight level indices into the ``altitude_ft`` dimension. If ``None``
            (default), all FLs are returned with shape ``(n_sample, n_fl)``.
            If an array, outputs are ``(n_sample, len(fl_idx))``.

        Returns
        -------
        EdgeInterpolation
            Interpolated met fields at the requested sample and time coordinates.
        """
        time_coords = self.ds["time"].values  # (n_time,) datetime64[ns]
        time_s = (time_coords - time_coords[0]) / np.timedelta64(1, "s")
        query_s = (times - time_coords[0]) / np.timedelta64(1, "s")

        n_time = len(time_coords)
        fp = np.arange(n_time, dtype=np.float64)
        t_frac = np.interp(query_s, time_s, fp).astype(np.float32)  # np.interp returns float64
        if np.any(~np.isfinite(t_frac)):  # idiot check
            raise RuntimeError("Non-finite t_frac values")

        t_lo = np.floor(t_frac).astype(np.int16)  # n_time << int16.max, and f32 + int16 = f32
        t_hi = np.minimum(t_lo + 1, n_time - 1)
        w = t_frac - t_lo

        fl_idx = np.arange(self.ds.sizes["altitude_ft"]) if fl_idx is None else fl_idx

        def _lerp(name: str) -> npt.NDArray[np.floating]:
            data = self.ds[name].values  # (n_total_samples, n_fl, n_time)
            lo = data[sample_idxs[:, np.newaxis], fl_idx[np.newaxis, :], t_lo]
            hi = data[sample_idxs[:, np.newaxis], fl_idx[np.newaxis, :], t_hi]
            return lo + w * (hi - lo)

        return EdgeInterpolation(
            air_temperature=_lerp("air_temperature"),
            eastward_wind=_lerp("eastward_wind"),
            northward_wind=_lerp("northward_wind"),
            eef_per_m=_lerp("eef_per_m") if "eef_per_m" in self.ds else None,
        )

    @classmethod
    def from_met(
        cls,
        met: MetDataset | xr.Dataset,
        dag: HorizontalDAG,
        altitude_ft: npt.NDArray[np.floating],
        takeoff_time: pd.Timestamp,
        flight_hours: int,
        spacing_m: float,
    ) -> Self:
        """Interpolate met data onto ``dag`` edge sample points.

        Parameters
        ----------
        met : MetDataset | xr.Dataset
            Gridded met dataset with "air_temperature", "eastward_wind", and "northward_wind".
            If "eef_per_m" is present, it will also be included in the output with
            NaN values filled to 0.0 (no EEF forecast is treated as zero forcing).
            NaN values in weather variables are not allowed and will raise an error.
            Either a pycontrails ``MetDataset`` or a raw ``xr.Dataset`` with similar
            structure can be passed.
        dag : HorizontalDAG
            Horizontal DAG whose edges will be sampled.
        altitude_ft : npt.NDArray[np.floating]
            An array of altitudes in feet to interpolate onto.
        takeoff_time : pd.Timestamp
            Departure time for the flight, used to select met time steps.
        flight_hours : int
            Number of hourly time steps to retain starting from takeoff_time.
        spacing_m : float
            Spacing in meters between sample points along edges. Passed to ``dag.sample_edges``.

        Returns
        -------
        EdgeMetLookup
            EdgeMetLookup with weather interpolated onto ``(sample, altitude_ft, time)`` dims.

        """
        sample_lon, sample_lat, edge_idx, edge_ptr = dag.sample_edges(spacing_m=spacing_m)

        # Compute distance from edge source to each sample
        edge_src = dag.edge_src[edge_idx]
        src_lon = dag.lon[edge_src]
        src_lat = dag.lat[edge_src]
        cum_dist = geo.haversine(src_lon, src_lat, sample_lon, sample_lat)
        cum_dist[edge_ptr[:-1]] = 0.0  # defensive, not strictly needed

        # Compute distance and azimuth from one sample to the next (used for wind calcs)
        last = edge_ptr[1:] - 1
        delta_dist = np.empty_like(cum_dist)
        delta_dist[:-1] = np.diff(cum_dist)
        delta_dist[last] = 0.0
        sample_azimuth = np.empty_like(cum_dist)
        sample_azimuth[:-1] = np.deg2rad(
            geo.azimuth(sample_lon[:-1], sample_lat[:-1], sample_lon[1:], sample_lat[1:])
        )
        sample_azimuth[last] = sample_azimuth[last - 1]  # copy previous azimuth for last sample

        # Ensure variables
        ds = met.data if isinstance(met, MetDataset) else met

        variables = ["air_temperature", "eastward_wind", "northward_wind"]
        if "eef_per_m" in ds:
            variables.append("eef_per_m")
        ds = ds[variables]

        # Downselect met in time
        # We're assuming that met has hourly spacing, which could be relaxed
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
                stacklevel=2,
            )

        ds = ds.sel(time=usable)

        # Convert to altitude_ft coordinates
        ds_altitude_ft = units.pl_to_ft(ds["level"])
        ds = ds.assign_coords(altitude_ft=ds_altitude_ft).swap_dims(level="altitude_ft")

        # Select or interpolate vertically on FL choices
        try:
            ds = ds.sel(altitude_ft=altitude_ft, method="nearest", tolerance=50.0)
        except KeyError:
            ds = ds.interp(altitude_ft=altitude_ft)
            # Ensure float32 — interp promotes to float64
            for var in ds:
                ds[var] = ds[var].astype(np.float32)

        # Interpolate horizontally onto sample points
        # Calling ds.interp chews up too much memory and the pycontrails RGI isn't
        # exactly designed for this, so just call custom numpy-based _bilinear_interp
        ds = _bilinear_interp(ds, sample_lon, sample_lat)

        # Raise on NaN in core weather - downstream computations would be poisoned.
        for var in ("air_temperature", "eastward_wind", "northward_wind"):
            if ds[var].isnull().any():
                raise ValueError(f"NaN values found in '{var}' after interpolation onto samples")

        # NaN-fill eef_per_m with 0.0. If NaNs are kept, downstream computations would be poisoned.
        if "eef_per_m" in ds:
            ds["eef_per_m"] = ds["eef_per_m"].fillna(0.0)

        return cls(
            ds=ds,
            edge_ptr=edge_ptr,
            edge_idx=edge_idx,
            sample_lon=sample_lon,
            sample_lat=sample_lat,
            cum_dist=cum_dist,
            delta_dist=delta_dist,
            sample_azimuth=sample_azimuth,
        )


def _dual_az_edges(
    lon: npt.NDArray[np.floating],
    lat: npt.NDArray[np.floating],
    origin_idx: int,
    dest_idx: int,
    max_angle_deg: float = 40.0,
    max_dist_m: float = 500_000.0,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.floating]]:
    """Build directed edges using a dual azimuth constraint.

    For each pair of nodes within ``max_dist_m``, the directed edge tail -> head
    is included iff:

    1. The azimuth from tail to head is within ``max_angle_deg`` of the azimuth
       from tail to the destination (node ``dest_idx``).
    2. The azimuth from head to tail is within ``max_angle_deg`` of the azimuth
       from head to the origin (node ``origin_idx``).

    These two conditions ensure that each edge roughly points toward the destination
    and away from the origin. Together, they constrain edges to lie within a
    football-shaped corridor between origin and destination and guarantee that each
    edge is forward-pointing.

    Returns
    -------
    edges : npt.NDArray[np.int64]
        ``(m, 2)`` array of ``[tail, head]`` index pairs.
    edge_dist : npt.NDArray[np.floating]
        ``(m,)`` haversine distances in meters. The dtype matches the input lon/lat dtype.
    """
    # Build 2d array of all candidate pairs within distance threshold
    # If this gets expensive, we could use a KDTree approach instead
    dist = geo.haversine(
        lon[:, np.newaxis],
        lat[:, np.newaxis],
        lon[np.newaxis, :],
        lat[np.newaxis, :],
    )
    tail, head = np.nonzero((dist > 0.0) & (dist <= max_dist_m))

    # Precompute per-node azimuths from each node to dest and origin
    az_to_dest = geo.azimuth(lon, lat, lon[dest_idx], lat[dest_idx])
    az_to_origin = geo.azimuth(lon, lat, lon[origin_idx], lat[origin_idx])

    # Compute azimuths for all candidate edges
    az_at_tail = geo.azimuth(lon[tail], lat[tail], lon[head], lat[head])
    at_at_head = geo.azimuth(lon[head], lat[head], lon[tail], lat[tail])

    # Compute delta angles for azimuth(tail -> head) vs azimuth(tail -> dest)
    # and for azimuth(head -> tail) vs azimuth(head -> origin)
    delta_tail = np.abs((az_at_tail - az_to_dest[tail] + 180.0) % 360.0 - 180.0)
    delta_head = np.abs((at_at_head - az_to_origin[head] + 180.0) % 360.0 - 180.0)

    keep = (delta_tail <= max_angle_deg) & (delta_head <= max_angle_deg)
    edges = np.column_stack([tail[keep], head[keep]])
    edge_dist = dist[tail[keep], head[keep]]
    return edges, edge_dist


def _bilinear_interp(
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

    result_vars = {}
    for name, da in ds.items():
        # data shape: (longitude, latitude, altitude_ft, time)
        if da.dims != ("longitude", "latitude", "altitude_ft", "time"):
            raise ValueError(f"Unexpected dimensions for variable {name}: {da.dims}")

        v = da.values  # this materializes data into memory if not already loaded
        f00 = v[i, j]
        f01 = v[i + 1, j]
        f10 = v[i, j + 1]
        f11 = v[i + 1, j + 1]

        val = (
            (1.0 - wx) * (1.0 - wy) * f00
            + wx * (1.0 - wy) * f01
            + (1.0 - wx) * wy * f10
            + wx * wy * f11
        )
        result_vars[name] = (("sample", "altitude_ft", "time"), val)

    return xr.Dataset(
        result_vars,
        coords={
            "sample": np.arange(len(sample_lon), dtype=np.int64),
            "altitude_ft": ds["altitude_ft"],
            "time": ds["time"],
        },
    )
