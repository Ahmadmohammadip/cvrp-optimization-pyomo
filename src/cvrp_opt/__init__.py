"""Exact MILP solution to the Capacitated Vehicle Routing Problem.

Two-index vehicle-flow formulation with Miller-Tucker-Zemlin subtour
elimination. Exact by design, which means small by consequence: see
docs/formulation.md for why, and the benchmark table in README.md for what
that costs in practice.
"""

__version__ = "0.1.0"
