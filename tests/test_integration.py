"""End-to-end runs on the committed sample instances.

Only the 5- and 10-customer instances are solved here. The larger ones exist
for the benchmark, and putting them in the test suite would make CI take
minutes for no extra confidence — the point they demonstrate is measured in
benchmark.py, not asserted here.
"""

from pathlib import Path

import pytest

from cvrp_opt.data.loaders import load_instance_json
from cvrp_opt.model.builder import build_from_system
from cvrp_opt.solve import solve_cvrp

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_instances"
CI_SAFE_INSTANCES = ["cvrp_05.json", "cvrp_10.json"]
ALL_INSTANCES = [*CI_SAFE_INSTANCES, "cvrp_15.json", "cvrp_20.json", "cvrp_25.json"]


@pytest.mark.parametrize("filename", ALL_INSTANCES)
def test_every_sample_instance_loads_and_validates(filename):
    # Loading is cheap for all five; only solving is expensive.
    system = load_instance_json(SAMPLE_DIR / filename)

    expected_n = int(filename.split("_")[1].split(".")[0])
    assert system.n_customers == expected_n
    assert system.num_vehicles == system.k_min
    assert all(c.demand <= system.fleet.capacity for c in system.customers)


@pytest.mark.parametrize("filename", CI_SAFE_INSTANCES)
def test_sample_instance_solves_to_a_valid_solution(filename):
    system = load_instance_json(SAMPLE_DIR / filename)

    result = solve_cvrp(build_from_system(system))

    assert result.is_optimal

    visited = sorted(c for route in result.routes for c in route)
    assert visited == sorted(c.id for c in system.customers)

    for load in result.route_loads:
        assert load <= system.fleet.capacity + 1e-6

    assert result.vehicles_used <= system.num_vehicles
    assert result.vehicles_used >= system.k_min


@pytest.mark.parametrize("filename", CI_SAFE_INSTANCES)
def test_reconstructed_routes_agree_with_the_objective(filename):
    # The check that catches reconstruction drifting away from the model: if
    # the routes were traced wrongly, their distances would stop summing to the
    # objective the solver reported.
    system = load_instance_json(SAMPLE_DIR / filename)

    result = solve_cvrp(build_from_system(system))

    assert sum(result.route_distances) == pytest.approx(result.total_distance, abs=1e-6)
    assert sum(result.route_loads) == pytest.approx(system.total_demand, abs=1e-6)


def test_lower_bound_brackets_the_objective():
    system = load_instance_json(SAMPLE_DIR / "cvrp_10.json")

    result = solve_cvrp(build_from_system(system))

    assert result.lower_bound <= result.total_distance + 1e-6
    assert result.gap == pytest.approx(0.0, abs=1e-3)


def test_time_limit_is_reported_honestly():
    # Two seconds is enough for HiGHS to find a feasible route set on the
    # 15-customer instance but nowhere near enough to prove it optimal. The
    # result must come back flagged as unproven with a real gap, never silently
    # presented as an optimum.
    system = load_instance_json(SAMPLE_DIR / "cvrp_15.json")

    result = solve_cvrp(build_from_system(system), time_limit=2)

    assert not result.is_optimal
    assert result.gap > 0
    assert result.lower_bound < result.total_distance
    assert "not proven optimal" in result.summary()
    # Even an unproven answer must be a structurally valid set of routes.
    visited = sorted(c for route in result.routes for c in route)
    assert visited == sorted(c.id for c in system.customers)
    for load in result.route_loads:
        assert load <= system.fleet.capacity + 1e-6


def test_no_feasible_solution_in_time_raises_rather_than_returning_nothing():
    # One second on the largest instance is not enough for HiGHS to find even a
    # first incumbent. There is genuinely nothing to return, so this must fail
    # loudly instead of handing back an empty route list that would read as
    # "zero distance".
    system = load_instance_json(SAMPLE_DIR / "cvrp_25.json")

    with pytest.raises(RuntimeError, match="without finding any feasible route set"):
        solve_cvrp(build_from_system(system), time_limit=1)


def test_summary_reads_clearly_when_proven():
    system = load_instance_json(SAMPLE_DIR / "cvrp_05.json")

    result = solve_cvrp(build_from_system(system))

    assert "proven optimal" in result.summary()
    assert "not proven" not in result.summary()
