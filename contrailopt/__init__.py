"""``contrailopt`` is a library for contrail-aware flight optimization."""

from importlib import metadata

from contrailopt.dag import AirportCoords, EdgeMetLookup, HorizontalDAG
from contrailopt.optimize import DAGResult, DAGState, Optimizer, cruise_flight_levels
from contrailopt.waypoints import load_faa_waypoints

__version__ = metadata.version("contrailopt")

__all__ = [
    "AirportCoords",
    "DAGResult",
    "DAGState",
    "EdgeMetLookup",
    "HorizontalDAG",
    "Optimizer",
    "cruise_flight_levels",
    "load_faa_waypoints",
]
