"""``contrailopt`` is a library for contrail-aware flight optimization."""

from importlib import metadata

from contrailopt.dag import AirportCoords, EdgeMetLookup, HorizontalDAG
from contrailopt.metrics import flight_metrics
from contrailopt.optimize import (
    DAGResult,
    DAGState,
    Optimizer,
    cruise_flight_levels,
    estimate_flight_hours,
    solve_dag,
)
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
    "estimate_flight_hours",
    "flight_metrics",
    "load_faa_waypoints",
    "solve_dag",
]
