"""Phase 3: JSON and CSV loading, including the failure messages."""

import json

import pytest

from cvrp_opt.data.loaders import (
    load_instance_csv,
    load_instance_csv_text,
    load_instance_json,
)

INSTANCE = {
    "depot": {"id": 0, "x": 50.0, "y": 50.0},
    "customers": [
        {"id": 1, "x": 10.0, "y": 20.0, "demand": 12.0},
        {"id": 2, "x": 70.0, "y": 30.0, "demand": 18.0},
        {"id": 3, "x": 40.0, "y": 90.0, "demand": 9.0},
    ],
    "fleet": {"capacity": 25.0, "num_vehicles": 2},
}

NODE_CSV = "id,x,y,demand\n0,50,50,0\n1,10,20,12\n2,70,30,18\n3,40,90,9\n"


def _write_json(tmp_path, data, name="instance.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_loads_a_complete_json_instance(tmp_path):
    system = load_instance_json(_write_json(tmp_path, INSTANCE))

    assert system.n_customers == 3
    assert system.total_demand == pytest.approx(39.0)
    assert system.fleet.capacity == 25.0
    assert system.num_vehicles == 2
    assert system.k_min == 2


def test_json_fleet_size_may_be_omitted(tmp_path):
    data = json.loads(json.dumps(INSTANCE))
    del data["fleet"]["num_vehicles"]

    system = load_instance_json(_write_json(tmp_path, data))

    assert system.fleet.num_vehicles is None
    assert system.num_vehicles == system.k_min == 2


@pytest.mark.parametrize("key", ["depot", "customers", "fleet"])
def test_json_missing_top_level_key_is_rejected(tmp_path, key):
    data = json.loads(json.dumps(INSTANCE))
    del data[key]

    with pytest.raises(ValueError, match=f"missing required key '{key}'"):
        load_instance_json(_write_json(tmp_path, data))


def test_invalid_instance_in_json_fails_at_load(tmp_path):
    # Validation lives in the schema, so a bad file fails on load rather than
    # after a long solve.
    data = json.loads(json.dumps(INSTANCE))
    data["fleet"]["num_vehicles"] = 1

    with pytest.raises(ValueError, match="at least 2 needed"):
        load_instance_json(_write_json(tmp_path, data))


def test_loads_a_node_csv(tmp_path):
    path = tmp_path / "nodes.csv"
    path.write_text(NODE_CSV, encoding="utf-8")

    system = load_instance_csv(path, capacity=25.0, num_vehicles=2)

    assert system.depot.id == 0
    assert system.depot.x == 50.0
    assert system.n_customers == 3
    assert system.total_demand == pytest.approx(39.0)


def test_csv_and_json_agree(tmp_path):
    path = tmp_path / "nodes.csv"
    path.write_text(NODE_CSV, encoding="utf-8")

    from_csv = load_instance_csv(path, capacity=25.0, num_vehicles=2)
    from_json = load_instance_json(_write_json(tmp_path, INSTANCE))

    assert from_csv.coords == from_json.coords
    assert from_csv.demands == from_json.demands
    assert from_csv.num_vehicles == from_json.num_vehicles


def test_csv_text_loader_matches_the_file_loader(tmp_path):
    path = tmp_path / "nodes.csv"
    path.write_text(NODE_CSV, encoding="utf-8")

    from_file = load_instance_csv(path, capacity=25.0)
    from_text = load_instance_csv_text(NODE_CSV, capacity=25.0)

    assert from_text.coords == from_file.coords
    assert from_text.demands == from_file.demands


def test_csv_text_loader_tolerates_a_byte_order_mark():
    system = load_instance_csv_text("﻿" + NODE_CSV, capacity=25.0)

    assert system.n_customers == 3


def test_csv_missing_columns_are_named(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("id,x,y\n0,1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"missing required column\(s\) \['demand'\]"):
        load_instance_csv(path, capacity=10.0)


def test_csv_non_numeric_value_names_the_line(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("id,x,y,demand\n0,50,50,0\n1,10,nope,12\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 3: column 'y' is not a number"):
        load_instance_csv(path, capacity=25.0)


def test_csv_without_a_depot_row_is_rejected(tmp_path):
    path = tmp_path / "no_depot.csv"
    path.write_text("id,x,y,demand\n1,10,20,12\n2,70,30,18\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no row with id 0"):
        load_instance_csv(path, capacity=25.0)


def test_csv_depot_with_demand_is_rejected(tmp_path):
    path = tmp_path / "bad_depot.csv"
    path.write_text("id,x,y,demand\n0,50,50,7\n1,10,20,12\n", encoding="utf-8")

    with pytest.raises(ValueError, match="depot 0 must have demand 0"):
        load_instance_csv(path, capacity=25.0)


def test_csv_duplicate_depot_row_is_rejected(tmp_path):
    path = tmp_path / "two_depots.csv"
    path.write_text("id,x,y,demand\n0,50,50,0\n0,10,10,0\n1,10,20,12\n", encoding="utf-8")

    with pytest.raises(ValueError, match="depot 0 appears twice"):
        load_instance_csv(path, capacity=25.0)


def test_csv_header_only_is_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("id,x,y,demand\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no data rows"):
        load_instance_csv(path, capacity=10.0)


def test_csv_supports_string_node_ids():
    text = "id,x,y,demand\nhub,0,0,0\nshop-a,3,4,5\n"

    system = load_instance_csv_text(text, capacity=10.0, depot_id="hub")

    assert system.depot.id == "hub"
    assert system.customers[0].id == "shop-a"
    assert system.distance_matrix()[("hub", "shop-a")] == pytest.approx(5.0)
