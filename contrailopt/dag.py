"""Utilities for horizontal DAG construction."""

from collections.abc import Generator
from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pycontrails.core import airports
from pycontrails.physics import geo, units

from contrailopt.slerp import gc_interp, gc_npts, spherical_azimuth, spherical_fwd


@dataclass(kw_only=True, slots=True, frozen=True)
class AirportCoords:
    icao_code: str
    longitude: float
    latitude: float
    elevation_ft: float

    @classmethod
    def from_icao(cls, icao_code: str) -> Self:
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
        return self.longitude, self.latitude


def _csr_flat_pos(
    adj_ptr: npt.NDArray[np.integer],
    nodes: npt.NDArray[np.integer],
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    """Return flat indices into CSR data arrays for a batch of row nodes.

    Returns ``(flat_pos, lengths)`` where flat_pos indexes into adj/edge_dist
    and lengths[i] is the number of entries for nodes[i].
    """
    starts = adj_ptr[nodes]
    lengths = adj_ptr[nodes + 1] - starts
    flat_starts = np.repeat(starts, lengths)
    offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    flat_pos = flat_starts + offsets
    return flat_pos, lengths


def _neighbors_batch(
    adj_ptr: npt.NDArray[np.integer],
    adj: npt.NDArray[np.integer],
    nodes: npt.NDArray[np.integer],
) -> npt.NDArray[np.integer]:
    """Determine neighbors (duplicates included with multiplicity) for a batch of nodes."""
    flat_pos, _ = _csr_flat_pos(adj_ptr, nodes)
    return adj[flat_pos]


def _reachability(
    seed: int,
    n_nodes: int,
    adj_ptr: npt.NDArray[np.integer],
    adj: npt.NDArray[np.integer],
) -> npt.NDArray[np.bool_]:
    """Return boolean array of nodes reachable from seed via batch_neighbors."""
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
    adj: npt.NDArray[np.integer],
    n_nodes: int,
    src: npt.NDArray[np.integer],
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    """Compute reverse (dst -> src) CSR representation."""
    order = np.argsort(adj)

    rev_ptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.add.at(rev_ptr[1:], adj, 1)
    np.cumsum(rev_ptr, out=rev_ptr)

    return rev_ptr, src[order]


@dataclass(kw_only=True, slots=True)
class HorizontalDAG:
    """Directed graph on (lon, lat) nodes with CSR adjacency."""

    lon: npt.NDArray[np.floating]  # (N,)
    lat: npt.NDArray[np.floating]  # (N,)
    adj_ptr: npt.NDArray[np.integer]  # (N + 1,) CSR row pointers
    adj: npt.NDArray[np.integer]  # (M,) neighbor indices, the destinations of directed edges
    edge_dist: npt.NDArray[np.floating]  # (M,) great-circle distance per edge in meters
    h_origin: int  # index of the distinguished origin node
    h_dest: int  # index of the distinguished destination node

    def __repr__(self) -> str:
        return f"HorizontalDAG({self.n_nodes} nodes, {self.n_edges} edges)"

    @property
    def n_nodes(self) -> int:
        """The number of nodes in the graph."""
        return len(self.lon)

    @property
    def n_edges(self) -> int:
        """The number of directed edges in the graph."""
        return len(self.adj)

    @property
    def out_degree(self) -> npt.NDArray[np.integer]:
        """The out-degree of each node."""
        return np.diff(self.adj_ptr)

    @property
    def edge_src(self) -> npt.NDArray[np.integer]:
        """The source endpoint index of each directed edge."""
        return np.repeat(np.arange(self.n_nodes, dtype=np.int64), self.out_degree)

    @property
    def edges(self) -> npt.NDArray[np.integer]:
        """The directed edges of the graph as (src, dest) index pairs in an ``(M, 2)`` array."""
        return np.column_stack([self.edge_src, self.adj])

    def neighbors(self, i: int) -> npt.NDArray[np.integer]:
        """Return the neighbors of a specified node."""
        return self.adj[self.adj_ptr[i] : self.adj_ptr[i + 1]]

    def neighbors_batch(self, nodes: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
        """Return neighbors (duplicates included with multiplicity) for a batch of nodes."""
        return _neighbors_batch(self.adj_ptr, self.adj, nodes)

    def edge_distances(self, i: int) -> npt.NDArray[np.floating]:
        return self.edge_dist[self.adj_ptr[i] : self.adj_ptr[i + 1]]

    def expand_neighbors(
        self, nodes: npt.NDArray[np.integer]
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.floating], npt.NDArray[np.integer]]:
        """Expand CSR adjacency for a batch of nodes into flat edge arrays.

        Generalizes :meth:`neighbors_batch` by also returning edge distances
        and a source-index mapping.

        Returns ``(flat_nbr, flat_dist, src_idx)`` where:
        - flat_nbr: (F,) neighbor indices for all edges leaving nodes.
        - flat_dist: (F,) edge distances in meters for those edges.
        - src_idx: (F,) index into nodes for each flat entry, so
          ``nodes[src_idx[k]]`` is the source node of flat edge k.

        Here F is the total number of outgoing edges from all ``nodes`` (with multiplicity,
        since the same neighbor can appear via different source nodes).
        """
        flat_pos, lengths = _csr_flat_pos(self.adj_ptr, nodes)
        src_idx = np.repeat(np.arange(len(nodes)), lengths)
        return self.adj[flat_pos], self.edge_dist[flat_pos], src_idx

    def adjacency_matrix(self) -> npt.NDArray[np.bool]:
        """Return dense boolean adjacency matrix A where A[i, j] is True for i->j."""
        matrix = np.zeros((self.n_nodes, self.n_nodes), dtype=bool)
        matrix[self.edge_src, self.adj] = True
        return matrix

    def prune(self) -> Self:
        """Return a new DAG with only nodes reachable from origin that also reach dest."""
        src = self.edge_src
        fwd = _reachability(self.h_origin, self.n_nodes, self.adj_ptr, self.adj)
        rev_ptr, rev_adj = _reverse_csr(self.adj, self.n_nodes, src)
        bwd = _reachability(self.h_dest, self.n_nodes, rev_ptr, rev_adj)
        live = fwd & bwd

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

    def sample_edges(
        self, spacing_m: float = 20_000.0
    ) -> tuple[
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.integer],
        npt.NDArray[np.integer],
    ]:
        """Sample points along every edge at roughly ``spacing_m`` meter intervals.

        Returns (sample_lon, sample_lat, edge_idx, edge_ptr) where:
        - sample_lon, sample_lat: flat arrays of all sample coordinates.
        - edge_idx: edge index for each sample point.
        - edge_ptr: CSR-style pointer so edge i's samples are at
          sample_lon[edge_ptr[i]:edge_ptr[i+1]].
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
            frac,
        )

        edge_idx = np.repeat(np.arange(self.n_edges), n_samples)
        return sample_lon, sample_lat, edge_idx, edge_ptr

    def plot(self, ax=None) -> "matplotlib.axes.Axes":
        """Plot the DAG on a cartopy map."""
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        pc = ccrs.PlateCarree()

        if ax is None:
            _, ax = plt.subplots(figsize=(20, 10), subplot_kw={"projection": pc})

            lon_min = self.lon.min() - 2.0
            lon_max = self.lon.max() + 2.0
            lat_min = self.lat.min() - 2.0
            lat_max = self.lat.max() + 2.0
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=pc)

        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor="gray")

        # Draw edges
        edge_src = self.edge_src
        segments = np.stack(
            [
                np.column_stack([self.lon[edge_src], self.lat[edge_src]]),
                np.column_stack([self.lon[self.adj], self.lat[self.adj]]),
            ],
            axis=1,
        )
        lc = LineCollection(segments, colors="steelblue", linewidths=0.3, alpha=0.4, transform=pc)
        ax.add_collection(lc)

        # Draw nodes
        ax.scatter(self.lon, self.lat, s=2, color="black", transform=pc, zorder=5)
        ax.plot(
            self.lon[self.h_origin],
            self.lat[self.h_origin],
            "ro",
            markersize=8,
            transform=pc,
            zorder=10,
        )
        ax.plot(
            self.lon[self.h_dest],
            self.lat[self.h_dest],
            "go",
            markersize=8,
            transform=pc,
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
        poisson_radius: float = 0.025,
        max_cross_track: float | None = None,
        max_angle_deg: float = 40.0,
        max_dist_m: float = 500_000.0,
    ) -> Self:
        """Build a DAG from Poisson-disk sampled points along the OD great circle."""
        from scipy.stats.qmc import PoissonDisk

        gs_distance = geo.haversine(origin_lon, origin_lat, dest_lon, dest_lat)

        if max_cross_track is None:
            max_cross_track = min(1_000_000.0, gs_distance / 4.0)

        nx = int(gs_distance / 100_000.0) + 1

        # Great circle spine
        gc_lons, gc_lats = gc_npts(origin_lon, origin_lat, dest_lon, dest_lat, nx - 2)
        gc_lons = np.concatenate([[origin_lon], gc_lons, [dest_lon]])
        gc_lats = np.concatenate([[origin_lat], gc_lats, [dest_lat]])

        # Perpendicular azimuths along spine
        az_fwd = spherical_azimuth(gc_lons[:-1], gc_lats[:-1], gc_lons[1:], gc_lats[1:])
        az_perp = np.empty(nx)
        az_perp[:-1] = az_fwd + 90.0
        az_perp[-1] = az_perp[-2]

        # Poisson disk sampling in [0, 1]^2
        sampler = PoissonDisk(d=2, radius=poisson_radius)
        pts = sampler.fill_space()
        t = pts[:, 0]
        cross = pts[:, 1] * 2.0 * max_cross_track - max_cross_track

        # Prepend origin and append dest
        t = np.concatenate([[0.0], t, [1.0]])
        cross = np.concatenate([[0.0], cross, [0.0]])

        # Project to (lon, lat)
        t_gc = np.linspace(0.0, 1.0, nx)
        lon_base = np.interp(t, t_gc, gc_lons)
        lat_base = np.interp(t, t_gc, gc_lats)
        az_base = np.interp(t, t_gc, az_perp)
        lon, lat = spherical_fwd(lon_base, lat_base, az_base, cross)

        return cls.from_points(
            lon,
            lat,
            origin_idx=0,
            dest_idx=len(lon) - 1,
            max_angle_deg=max_angle_deg,
            max_dist_m=max_dist_m,
        )

    def topo_wavefronts(self) -> Generator[npt.NDArray[np.integer], None, None]:
        """Return topological wavefronts reachable from origin.

        Exclude the first wavefront containing only the origin.

        Wavefront k contains nodes whose remaining in-degree is zero after removing
        wavefronts 0..k-1. Therefore edges and paths only go from earlier wavefronts
        to later wavefronts, so no later wavefront can reach an earlier one.
        """
        in_degree = np.zeros(self.n_nodes, dtype=np.int64)
        np.add.at(in_degree, self.adj, 1)

        wave_nodes = np.array([self.h_origin])
        while True:
            neighbors = self.neighbors_batch(wave_nodes)
            np.subtract.at(in_degree, neighbors, 1)
            candidates = np.unique(neighbors)
            filt = in_degree[candidates] == 0
            wave_nodes = candidates[filt]

            if wave_nodes.size == 0:
                break
            yield wave_nodes


@dataclass(kw_only=True, slots=True)
class EdgeMetLookup:
    """Pre-interpolated met data on edge sample points.

    Attributes:
        ds: xr.Dataset with dims (sample, altitude_ft, time) containing
            weather variables interpolated onto edge sample coordinates.
        edge_ptr: CSR-style pointer array (n_edges + 1,). Samples for edge i
            are at indices edge_ptr[i]:edge_ptr[i+1].
        edge_idx: Edge index for each sample point (n_samples,).
        sample_lon: Longitude of each sample point (n_samples,).
        sample_lat: Latitude of each sample point (n_samples,).
        sample_dist: Distance from edge source to each sample point in meters (n_samples,).
    """

    ds: xr.Dataset
    edge_ptr: npt.NDArray[np.integer]
    edge_idx: npt.NDArray[np.integer]
    sample_lon: npt.NDArray[np.floating]
    sample_lat: npt.NDArray[np.floating]
    sample_dist: npt.NDArray[np.floating]

    def sel_edge(self, edge_i: int) -> xr.Dataset:
        """Return met data for all samples along a single edge."""
        s = self.edge_ptr[edge_i]
        e = self.edge_ptr[edge_i + 1]
        return self.ds.isel(sample=slice(s, e))

    def sel_edges(self, edge_indices: npt.NDArray[np.integer]) -> list[xr.Dataset]:
        """Return met data slices for a batch of edges."""
        return [self.sel_edge(i) for i in edge_indices]


def preinterp_met(
    ds: xr.Dataset,
    dag: HorizontalDAG,
    fl_choices: npt.NDArray[np.floating],
    takeoff_time: pd.Timestamp,
    flight_hours: int,
    spacing_m: float,
) -> EdgeMetLookup:
    """Interpolate met data onto edge sample points.

    Parameters:
        met: Gridded meteorological dataset with pycontrails conventions.
        dag: Horizontal DAG whose edges will be sampled.
        fl_choices: Flight level altitudes in feet to interpolate onto.
        takeoff_time: Departure time.
        flight_hours: Number of hourly time steps to retain.
        spacing_m: Approximate spacing in meters between sample points along edges.

    Returns:
        EdgeMetLookup with weather interpolated onto (sample, altitude_ft, time).
    """
    sample_lon, sample_lat, edge_idx, edge_ptr = dag.sample_edges(spacing_m=spacing_m)

    # Compute cumulative distance from edge source per sample
    edge_src = dag.edge_src[edge_idx]
    src_lon = dag.lon[edge_src]
    src_lat = dag.lat[edge_src]
    sample_dist = geo.haversine(src_lon, src_lat, sample_lon, sample_lat)
    sample_dist[edge_ptr[:-1]] = 0.0

    # Downselect met in time, this will error if not all times are available
    times = pd.date_range(takeoff_time, periods=flight_hours, freq="h")
    ds = ds.sel(time=times)

    # Convert to altitude_ft coordinates
    altitude_ft = units.pl_to_ft(ds["level"])
    ds = ds.assign_coords(altitude_ft=altitude_ft).swap_dims(level="altitude_ft")

    # Interpolate horizontally onto sample points and vertically onto FL choices
    ds = ds.interp(
        altitude_ft=fl_choices,
        longitude=xr.DataArray(sample_lon, dims="sample"),
        latitude=xr.DataArray(sample_lat, dims="sample"),
    )

    return EdgeMetLookup(
        ds=ds,
        edge_ptr=edge_ptr,
        edge_idx=edge_idx,
        sample_lon=sample_lon,
        sample_lat=sample_lat,
        sample_dist=sample_dist,
    )


def _dual_az_edges(
    lon: npt.NDArray[np.floating],
    lat: npt.NDArray[np.floating],
    origin_idx: int,
    dest_idx: int,
    max_angle_deg: float = 40.0,
    max_dist_m: float = 500_000.0,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]]:
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

    Returns a tuple of:
    - edges: (M, 2) int array of [tail, head] index pairs.
    - edge_dist: (M,) float array of haversine distances in meters.
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
    az_to_dest = spherical_azimuth(lon, lat, lon[dest_idx], lat[dest_idx])
    az_to_origin = spherical_azimuth(lon, lat, lon[origin_idx], lat[origin_idx])

    # Compute azimuths for all candidate edges
    az_at_tail = spherical_azimuth(lon[tail], lat[tail], lon[head], lat[head])
    at_at_head = spherical_azimuth(lon[head], lat[head], lon[tail], lat[tail])

    # Compute delta angles for azimuth(tail -> head) vs azimuth(tail -> dest)
    # and for azimuth(head -> tail) vs azimuth(head -> origin)
    delta_tail = np.abs((az_at_tail - az_to_dest[tail] + 180.0) % 360.0 - 180.0)
    delta_head = np.abs((at_at_head - az_to_origin[head] + 180.0) % 360.0 - 180.0)

    keep = (delta_tail <= max_angle_deg) & (delta_head <= max_angle_deg)
    edges = np.column_stack([tail[keep], head[keep]])
    edge_dist = dist[tail[keep], head[keep]]
    return edges, edge_dist
