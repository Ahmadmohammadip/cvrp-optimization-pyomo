"""Phase 1: the degree and fleet-size constraints, and the objective on an
instance whose optimum is known by hand."""

import pytest
from pyomo.environ import value

from cvrp_opt.model.builder import build_cvrp_model
from cvrp_opt.solve import solve_cvrp


def _solve(cross, num_vehicles=None):
    model = build_cvrp_model(
        cross.distance,
        cross.demand,
        cross.capacity,
        num_vehicles or cross.k_min,
        depot=cross.depot,
    )
    return model, solve_cvrp(model)


def _departures(result, depot):
    return sum(1 for (i, _) in result.arcs if i == depot)


def test_matches_the_hand_computed_optimum(cross):
    _, result = _solve(cross)

    assert result.is_optimal
    assert result.total_distance == pytest.approx(cross.optimal_distance, abs=1e-6)


def test_every_customer_is_entered_and_left_exactly_once(cross):
    _, result = _solve(cross)

    for c in cross.customers:
        assert sum(1 for (i, _) in result.arcs if i == c) == 1
        assert sum(1 for (_, j) in result.arcs if j == c) == 1


def test_depot_departures_are_balanced_and_within_the_fleet(cross):
    _, result = _solve(cross)

    returns = sum(1 for (_, j) in result.arcs if j == cross.depot)

    assert _departures(result, cross.depot) == returns
    assert _departures(result, cross.depot) <= cross.k_min


def test_capacity_forces_the_minimum_vehicle_count(cross):
    # Four customers at demand 5 against capacity 10 cannot be served by one
    # vehicle, and nothing in the model says so directly — MTZ derives it.
    _, result = _solve(cross)

    assert _departures(result, cross.depot) == 2


def test_surplus_vehicles_are_left_unused(cross):
    # The `<= K` formulation (PROJECT_BRIEF.md section 7.1 supersedes 1.5):
    # offering more vehicles than needed must not make the answer worse.
    _, at_minimum = _solve(cross, num_vehicles=2)
    _, with_surplus = _solve(cross, num_vehicles=5)

    assert with_surplus.total_distance == pytest.approx(
        at_minimum.total_distance, abs=1e-6
    )
    assert _departures(with_surplus, cross.depot) == 2  # three vehicles stay parked


def test_no_customer_only_subtour_exists(cross):
    # Every selected arc must be reachable from the depot. MTZ is what rules
    # out a cycle that never touches it; this walks the arcs to confirm.
    _, result = _solve(cross)

    successors = {i: j for (i, j) in result.arcs if i != cross.depot}
    reached = set()
    for start in [j for (i, j) in result.arcs if i == cross.depot]:
        node = start
        while node != cross.depot:
            assert node not in reached, f"node {node} visited twice — subtour"
            reached.add(node)
            node = successors[node]

    assert reached == set(cross.customers)


def test_builder_rejects_a_depot_with_demand(cross):
    demand = dict(cross.demand)
    demand[cross.depot] = 3.0

    with pytest.raises(ValueError, match="must have demand 0"):
        build_cvrp_model(cross.distance, demand, cross.capacity, 2, depot=cross.depot)


def test_builder_rejects_missing_arcs(cross):
    distance = cross.distance
    del distance[(1, 2)]

    with pytest.raises(ValueError, match="missing 1 arc"):
        build_cvrp_model(distance, cross.demand, cross.capacity, 2, depot=cross.depot)


def test_objective_equals_the_summed_arc_costs(cross):
    model, result = _solve(cross)

    assert value(model.total_distance) == pytest.approx(
        sum(cross.distance[a] for a in result.arcs), abs=1e-6
    )
