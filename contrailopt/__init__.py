"""``contrailopt`` is a library for contrail-aware flight optimization."""

from importlib import metadata

from contrailopt.dag import AirportCoords, EdgeMetLookup, HorizontalDAG
from contrailopt.optimize import DAGResult, DAGState, Optimizer
from contrailopt.waypoints import load_faa_waypoints

__version__ = metadata.version("contrailopt")

__all__ = [
    "AirportCoords",
    "DAGResult",
    "DAGState",
    "EdgeMetLookup",
    "HorizontalDAG",
    "Optimizer",
    "load_faa_waypoints",
]
