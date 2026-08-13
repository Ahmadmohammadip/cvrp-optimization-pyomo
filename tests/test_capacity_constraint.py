"""Phase 3: capacity enforcement on solved instances.

The MTZ family carries capacity as well as subtour elimination, so there is no
separate capacity constraint to point at — these tests check the property it is
supposed to produce.
"""

import pytest

from cvrp_opt.data.schema import Customer, Depot, Fleet, System
from cvrp_opt.model.builder import build_from_system
from cvrp_opt.solve import solve_cvrp


def _ring_system(demands: list[float], capacity: float, num_vehicles=None) -> System:
    """Customers evenly spaced on a circle of radius 10 around the depot, so
    geography is neutral and capacity is what drives the vehicle count."""
    import math

    n = len(demands)
    customers = [
        Customer(
            id=i + 1,
            x=10 * math.cos(2 * math.pi * i / n),
            y=10 * math.sin(2 * math.pi * i / n),
            demand=d,
        )
        for i, d in enumerate(demands)
    ]
    return System(
        depot=Depot(id=0, x=0.0, y=0.0),
        customers=customers,
        fleet=Fleet(capacity=capacity, num_vehicles=num_vehicles),
    )


def test_no_route_exceeds_capacity():
    system = _ring_system([10.0] * 6, capacity=30.0)

    result = solve_cvrp(build_from_system(system))

    assert result.is_optimal
    for load in result.route_loads:
        assert load <= system.fleet.capacity + 1e-6


def test_capacity_dictates_the_number_of_vehicles():
    # Six customers at 10 each against capacity 30: two vehicles, both full.
    system = _ring_system([10.0] * 6, capacity=30.0)

    result = solve_cvrp(build_from_system(system))

    assert result.vehicles_used == 2
    assert sorted(result.route_loads) == [30.0, 30.0]


def test_tighter_capacity_forces_more_vehicles():
    loose = _ring_system([10.0] * 6, capacity=30.0)
    tight = _ring_system([10.0] * 6, capacity=20.0)

    loose_result = solve_cvrp(build_from_system(loose))
    tight_result = solve_cvrp(build_from_system(tight))

    assert tight.k_min == 3
    assert tight_result.vehicles_used == 3
    # More vehicles means more depot round trips, so a longer total tour.
    assert tight_result.total_distance > loose_result.total_distance


def test_uneven_demands_still_respect_capacity():
    system = _ring_system([17.0, 3.0, 12.0, 8.0, 19.0, 1.0], capacity=25.0)

    result = solve_cvrp(build_from_system(system))

    assert result.is_optimal
    for route, load in zip(result.routes, result.route_loads, strict=True):
        expected = sum(system.demands[c] for c in route)
        assert load == pytest.approx(expected, abs=1e-6)
        assert load <= 25.0 + 1e-6


def test_every_customer_is_served_exactly_once():
    system = _ring_system([17.0, 3.0, 12.0, 8.0, 19.0, 1.0], capacity=25.0)

    result = solve_cvrp(build_from_system(system))

    visited = sorted(c for route in result.routes for c in route)
    assert visited == [c.id for c in system.customers]


def test_total_load_equals_total_demand():
    system = _ring_system([17.0, 3.0, 12.0, 8.0, 19.0, 1.0], capacity=25.0)

    result = solve_cvrp(build_from_system(system))

    assert sum(result.route_loads) == pytest.approx(system.total_demand, abs=1e-6)


def test_surplus_vehicles_do_not_change_the_optimum():
    # The `<= K` formulation: extra capacity in the fleet must never hurt.
    at_minimum = _ring_system([10.0] * 6, capacity=30.0, num_vehicles=2)
    surplus = _ring_system([10.0] * 6, capacity=30.0, num_vehicles=6)

    minimum_result = solve_cvrp(build_from_system(at_minimum))
    surplus_result = solve_cvrp(build_from_system(surplus))

    assert surplus_result.total_distance == pytest.approx(
        minimum_result.total_distance, abs=1e-6
    )
    assert surplus_result.vehicles_used == 2
