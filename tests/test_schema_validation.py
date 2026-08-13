"""Phase 3: instance validation.

For a MILP that can run for minutes, catching a malformed instance at
construction matters more than usual — the alternative is waiting out a long
solve to be told "infeasible" with no explanation.
"""

import re

import pytest

from cvrp_opt.data.schema import Customer, Depot, Fleet, System


def _customers(n=3, demand=10.0):
    return [Customer(id=i, x=float(i), y=0.0, demand=demand) for i in range(1, n + 1)]


def _system(**overrides) -> System:
    defaults = dict(
        depot=Depot(id=0, x=0.0, y=0.0),
        customers=_customers(),
        fleet=Fleet(capacity=30.0),
    )
    defaults.update(overrides)
    return System(**defaults)


def test_valid_instance_reports_its_derived_values():
    system = _system()

    assert system.n_customers == 3
    assert system.total_demand == pytest.approx(30.0)
    assert system.k_min == 1
    assert system.num_vehicles == 1  # unset fleet size falls back to k_min


def test_negative_demand_is_rejected():
    with pytest.raises(ValueError, match="demand must be >= 0"):
        Customer(id=1, x=0.0, y=0.0, demand=-5.0)


def test_non_positive_capacity_is_rejected():
    with pytest.raises(ValueError, match="capacity must be > 0"):
        Fleet(capacity=0.0)


def test_zero_vehicles_is_rejected():
    with pytest.raises(ValueError, match="num_vehicles must be >= 1"):
        Fleet(capacity=10.0, num_vehicles=0)


def test_empty_customer_list_is_rejected():
    with pytest.raises(ValueError, match="at least one customer"):
        _system(customers=[])


def test_duplicate_customer_ids_are_rejected():
    duplicated = [*_customers(2), Customer(id=1, x=9.0, y=9.0, demand=1.0)]

    with pytest.raises(ValueError, match=r"ids must be unique, repeated: \[1\]"):
        _system(customers=duplicated)


def test_customer_colliding_with_the_depot_id_is_rejected():
    clashing = [Customer(id=0, x=1.0, y=1.0, demand=5.0)]

    with pytest.raises(ValueError, match="collides with the depot id"):
        _system(customers=clashing)


def test_demand_above_vehicle_capacity_is_rejected():
    # The clearest infeasibility there is: no vehicle could ever serve this
    # customer, and this model does not split deliveries.
    too_big = [Customer(id=1, x=1.0, y=1.0, demand=50.0)]

    with pytest.raises(ValueError, match=re.escape("above vehicle capacity 30.0")):
        _system(customers=too_big)


def test_fleet_too_small_for_total_demand_is_rejected():
    # Six customers at 10 each against capacity 30 needs two vehicles.
    with pytest.raises(ValueError, match="at least 2 needed"):
        _system(customers=_customers(6), fleet=Fleet(capacity=30.0, num_vehicles=1))


def test_k_min_rounds_up_to_cover_a_partial_load():
    system = _system(customers=_customers(4), fleet=Fleet(capacity=30.0))

    assert system.total_demand == pytest.approx(40.0)
    assert system.k_min == 2  # 40/30 = 1.33 -> 2


def test_k_min_is_not_inflated_by_float_noise():
    # 3 x 10.0 / 30.0 is exactly 1, and must not ceil to 2.
    system = _system(customers=_customers(3), fleet=Fleet(capacity=30.0))

    assert system.k_min == 1


def test_surplus_fleet_size_is_reported_as_configured():
    # K above k_min is allowed; the model leaves the extras parked.
    system = _system(fleet=Fleet(capacity=30.0, num_vehicles=4))

    assert system.k_min == 1
    assert system.num_vehicles == 4


def test_coords_and_demands_include_the_depot():
    system = _system()

    assert system.coords[0] == (0.0, 0.0)
    assert system.demands[0] == 0.0
    assert len(system.coords) == 4
    assert len(system.demands) == 4


def test_distance_matrix_covers_every_ordered_pair_and_is_symmetric():
    system = _system()
    distance = system.distance_matrix()

    nodes = list(system.coords)
    assert len(distance) == len(nodes) * (len(nodes) - 1)
    for i in nodes:
        for j in nodes:
            if i != j:
                assert distance[(i, j)] == pytest.approx(distance[(j, i)])
    assert (0, 0) not in distance  # no self-loops


def test_distance_matrix_is_exact_euclidean_not_rounded():
    system = _system(
        customers=[Customer(id=1, x=3.0, y=4.0, demand=1.0)],
        fleet=Fleet(capacity=10.0),
    )

    assert system.distance_matrix()[(0, 1)] == pytest.approx(5.0)


def test_string_node_ids_are_supported():
    system = System(
        depot=Depot(id="warehouse", x=0.0, y=0.0),
        customers=[Customer(id="shop-a", x=1.0, y=0.0, demand=2.0)],
        fleet=Fleet(capacity=10.0),
    )

    assert system.demands["warehouse"] == 0.0
    assert system.distance_matrix()[("warehouse", "shop-a")] == pytest.approx(1.0)
