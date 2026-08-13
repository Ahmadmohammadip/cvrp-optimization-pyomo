"""Phase 2: turning selected arcs into ordered per-vehicle routes.

PROJECT_BRIEF.md section 1.7 flags this as a common source of silent bugs, so
these tests work on hand-built arc sets rather than solver output — including
malformed ones that must raise rather than return something plausible.
"""

import pytest

from cvrp_opt.model.builder import build_cvrp_model
from cvrp_opt.solve import reconstruct_routes, solve_cvrp

DEPOT = 0


def test_single_vehicle_round_trip():
    arcs = [(0, 1), (1, 2), (2, 3), (3, 0)]

    assert reconstruct_routes(arcs, DEPOT) == [[1, 2, 3]]


def test_two_vehicles_are_split_correctly():
    arcs = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)]

    routes = reconstruct_routes(arcs, DEPOT)

    assert sorted(routes) == [[1, 2], [3, 4]]


def test_single_customer_routes():
    # The depot is left and re-entered three times over; each stop stands alone.
    arcs = [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0)]

    routes = reconstruct_routes(arcs, DEPOT)

    assert sorted(routes) == [[1], [2], [3]]


def test_route_order_is_preserved_not_sorted():
    arcs = [(0, 3), (3, 1), (1, 2), (2, 0)]

    assert reconstruct_routes(arcs, DEPOT) == [[3, 1, 2]]


def test_no_arcs_gives_no_routes():
    assert reconstruct_routes([], DEPOT) == []


def test_string_node_ids_work():
    arcs = [("depot", "a"), ("a", "b"), ("b", "depot")]

    assert reconstruct_routes(arcs, "depot") == [["a", "b"]]


def test_subtour_is_rejected():
    # A valid route 0->1->0, plus a 2->3->2 cycle that never touches the depot.
    # This is precisely what MTZ exists to prevent; if it ever leaks through,
    # reconstruction must not quietly report just the depot route.
    arcs = [(0, 1), (1, 0), (2, 3), (3, 2)]

    with pytest.raises(ValueError, match="not reachable from the depot"):
        reconstruct_routes(arcs, DEPOT)


def test_dead_end_route_is_rejected():
    arcs = [(0, 1), (1, 2)]  # node 2 never leads anywhere

    with pytest.raises(ValueError, match="no outgoing arc"):
        reconstruct_routes(arcs, DEPOT)


def test_duplicate_outgoing_arc_is_rejected():
    arcs = [(0, 1), (1, 2), (1, 3), (2, 0), (3, 0)]

    with pytest.raises(ValueError, match="more than one outgoing arc"):
        reconstruct_routes(arcs, DEPOT)


def test_shared_customer_between_routes_is_rejected():
    # Two depot departures that converge on the same customer. Node 2 has a
    # single successor so the successor map is fine, but the routes overlap.
    arcs = [(0, 1), (1, 3), (3, 0), (0, 2), (2, 3)]

    with pytest.raises(ValueError, match="more than one route"):
        reconstruct_routes(arcs, DEPOT)


# --- against real solver output ---


def _solve(cross, num_vehicles=None):
    model = build_cvrp_model(
        cross.distance,
        cross.demand,
        cross.capacity,
        num_vehicles or cross.k_min,
        depot=cross.depot,
    )
    return solve_cvrp(model)


def test_solved_instance_yields_valid_routes(cross):
    result = _solve(cross)

    assert result.vehicles_used == 2
    # The known optimum pairs the customers by axis.
    assert sorted(sorted(r) for r in result.routes) == [[1, 2], [3, 4]]


def test_every_customer_appears_exactly_once(cross):
    result = _solve(cross)

    visited = [c for route in result.routes for c in route]

    assert sorted(visited) == sorted(cross.customers)
    assert len(visited) == len(set(visited))


def test_route_distances_sum_to_the_objective(cross):
    # If reconstruction ever drifts from the model, these stop agreeing.
    result = _solve(cross)

    assert sum(result.route_distances) == pytest.approx(result.total_distance, abs=1e-6)


def test_route_loads_are_reported_per_vehicle(cross):
    result = _solve(cross)

    assert sorted(result.route_loads) == [10.0, 10.0]
    assert sum(result.route_loads) == pytest.approx(
        sum(cross.demand[c] for c in cross.customers), abs=1e-6
    )


def test_surplus_vehicles_do_not_produce_empty_routes(cross):
    # `<= K` means unused vehicles simply never leave, so they must not show up
    # as zero-length routes.
    result = _solve(cross, num_vehicles=5)

    assert result.vehicles_used == 2
    assert all(route for route in result.routes)
