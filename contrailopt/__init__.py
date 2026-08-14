"""``contrailopt`` is a library for contrail-aware flight optimization."""

from importlib import metadata

from contrailopt import slerp
from contrailopt.dag import (
    AirportCoords,
    EdgeInterpolation,
    EdgeMetLookup,
    HorizontalDAG,
    Track,
    validate_flight_profile,
)
from contrailopt.metrics import flight_metrics
from contrailopt.optimize import (
    DAGResult,
    DAGState,
    Optimizer,
    cruise_flight_levels,
    estimate_flight_hours,
    solve_dag,
    solve_track,
)
from contrailopt.waypoints import load_faa_waypoints

__version__ = metadata.version("contrailopt")

__all__ = [
    "AirportCoords",
    "DAGResult",
    "DAGState",
    "EdgeInterpolation",
    "EdgeMetLookup",
    "HorizontalDAG",
    "Optimizer",
    "Track",
    "cruise_flight_levels",
    "estimate_flight_hours",
    "flight_metrics",
    "load_faa_waypoints",
    "slerp",
    "solve_dag",
    "solve_track",
    "validate_flight_profile",
]
