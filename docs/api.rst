API Reference
=============

.. currentmodule:: contrailopt


Optimization
------------

.. autosummary::
   :toctree: api/

   Optimizer
   DAGResult
   DAGState
   solve_dag
   solve_track
   cruise_flight_levels
   estimate_flight_hours


Geometry
--------

.. autosummary::
   :toctree: api/

   AirportCoords
   HorizontalDAG
   EdgeInterpolation
   EdgeMetLookup
   Track
   slerp


Utilities
---------

.. autosummary::
   :toctree: api/

   fill_nan_spatial
   load_faa_waypoints
   validate_flight_profile
