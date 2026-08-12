# contrailopt

`contrailopt` is a library for contrail-aware flight optimization. It builds a graph of candidate flight trajectories over vertical flight levels, Mach numbers, and horizontal routes, then solves for the path that minimizes a total-cost objective. The objective combines a fuel term, a time term (via a configurable cost index), and a climate term accounting for the warming caused by CO2 and contrails.

Aircraft performance is modeled with the [Poll–Schumann (PS) model](https://doi.org/10.1017/aer.2020.62). The library optionally takes gridded weather data, used for performance and true-airspeed calculations, and a contrail forecast (in units of J/m) that drives the climate term. It solves the resulting continuous optimization problem and additionally performs discrete avoidance around polygonal regions. `contrailopt` is designed to be interoperable with [pycontrails](https://py.contrails.org).

## Installation

`contrailopt` is not yet published on PyPI. Clone the repository and install from source with `pip`:

```bash
pip install -e .
```

 or another compatible tool such as [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Development

Run the linter and test suite with the `dev` dependency group, as the GitHub CI does:

```bash
uv run --group dev pre-commit run --all-files
uv run --group dev pytest
```

## Documentation

Build the docs with the `docs` dependency group:

```bash
uv run --group docs sphinx-build docs docs/_build/html
```
