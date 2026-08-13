"""Solver interface and result extraction.

Unlike the pure-LP sibling repos, this is a MILP over an NP-hard problem, so
"the solver stopped" and "the solver proved an optimum" are different events
and the result type has to say which happened. A run that hits its time limit
with a good incumbent is a useful answer — it just is not a proven one, and
`CVRPResult.is_optimal` is the flag that keeps those apart.
"""

import time
from dataclasses import dataclass, field

from pyomo.environ import ConcreteModel, SolverFactory, value
from pyomo.opt import TerminationCondition

DEFAULT_SOLVER = "appsi_highs"

# x_ij is binary, but solvers return 0.9999999. Anything above this counts as 1.
ARC_SELECTION_THRESHOLD = 0.5


@dataclass
class CVRPResult:
    """A solved (or time-limited) CVRP instance.

    `is_optimal` is the load-bearing field: False means the solver returned the
    best route set it found before running out of time, and `gap` bounds how
    far from optimal that could be.
    """

    arcs: list[tuple]
    total_distance: float
    lower_bound: float
    gap: float
    is_optimal: bool
    termination: str
    solve_time: float
    routes: list[list] = field(default_factory=list)
    route_loads: list[float] = field(default_factory=list)
    route_distances: list[float] = field(default_factory=list)

    @property
    def vehicles_used(self) -> int:
        return len(self.routes)

    def summary(self) -> str:
        status = (
            "proven optimal"
            if self.is_optimal
            else f"feasible, not proven optimal (gap {self.gap:.2%})"
        )
        return (
            f"{self.total_distance:.2f} total distance over {self.vehicles_used} "
            f"vehicle(s) — {status}, {self.solve_time:.1f}s"
        )


def solve_cvrp(
    model: ConcreteModel,
    solver_name: str = DEFAULT_SOLVER,
    time_limit: float | None = None,
) -> CVRPResult:
    """Solve the model and extract the selected arcs.

    `time_limit` is in seconds. Hitting it is not an error: if the solver found
    any feasible route set, it is returned with `is_optimal=False` and a gap.
    Raises only when there is genuinely nothing to return — an infeasible
    instance, or a time limit reached before any incumbent was found.
    """
    solver = SolverFactory(solver_name)
    if time_limit is not None:
        solver.options["time_limit"] = time_limit

    start = time.perf_counter()
    # load_solutions=False so a suboptimal incumbent is loaded deliberately
    # rather than triggering Pyomo's warning about doing it by accident.
    results = solver.solve(model, load_solutions=False)
    elapsed = time.perf_counter() - start

    condition = results.solver.termination_condition
    lower_bound = _as_float(results.problem.lower_bound)
    upper_bound = _as_float(results.problem.upper_bound)
    has_incumbent = upper_bound is not None and abs(upper_bound) != float("inf")

    if condition == TerminationCondition.infeasible:
        raise RuntimeError(
            "Instance is infeasible. With a validated System this should have been "
            "caught at construction — check fleet size and capacity against demand."
        )
    if condition not in (TerminationCondition.optimal, TerminationCondition.maxTimeLimit):
        raise RuntimeError(f"Solve failed with {solver_name}: termination = {condition}")
    if not has_incumbent:
        raise RuntimeError(
            f"Solver stopped ({condition}) without finding any feasible route set"
            + (f" within {time_limit}s" if time_limit else "")
            + ". Try a longer time limit or a smaller instance."
        )

    model.solutions.load_from(results)

    is_optimal = condition == TerminationCondition.optimal
    total_distance = value(model.total_distance)
    gap = _relative_gap(lower_bound, total_distance)

    arcs = [
        (i, j)
        for (i, j) in model.A
        if value(model.x[i, j]) > ARC_SELECTION_THRESHOLD
    ]

    return CVRPResult(
        arcs=arcs,
        total_distance=total_distance,
        lower_bound=lower_bound if lower_bound is not None else float("nan"),
        gap=gap,
        is_optimal=is_optimal,
        termination=str(condition),
        solve_time=elapsed,
    )


def _as_float(bound) -> float | None:
    if bound is None:
        return None
    try:
        return float(bound)
    except (TypeError, ValueError):
        return None


def _relative_gap(lower_bound: float | None, incumbent: float) -> float:
    """Relative MIP gap, the same quantity solvers report.

    Note that a "proven optimal" result usually still shows a tiny non-zero gap:
    HiGHS stops once the gap falls under its default relative tolerance (1e-4),
    so optimality is proven to within that tolerance, not to the last decimal.
    """
    if lower_bound is None or incumbent == 0:
        return 0.0
    return abs(incumbent - lower_bound) / abs(incumbent)
