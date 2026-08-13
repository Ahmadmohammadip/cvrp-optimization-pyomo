"""Load validated CVRP instances from JSON and CSV.

JSON carries a whole instance — depot, customers, and fleet are nested, which
maps onto the file format directly. CSV carries only the node table, since
capacity and fleet size are not per-node facts; they are passed as arguments.

Everything returned here has already passed the schema's validation, so a
malformed file fails at load rather than after a long solve.
"""

import csv
import json
from pathlib import Path

from cvrp_opt.data.schema import Customer, Depot, Fleet, System

ID_COLUMN = "id"
X_COLUMN = "x"
Y_COLUMN = "y"
DEMAND_COLUMN = "demand"
REQUIRED_COLUMNS = (ID_COLUMN, X_COLUMN, Y_COLUMN, DEMAND_COLUMN)


def load_instance_json(path: str | Path) -> System:
    """Read a complete instance from JSON.

    Expected shape:

        {
          "depot": {"id": 0, "x": 50.0, "y": 50.0},
          "customers": [{"id": 1, "x": 12.3, "y": 45.6, "demand": 14.0}, ...],
          "fleet": {"capacity": 60.0, "num_vehicles": 4}
        }

    `fleet.num_vehicles` may be omitted, leaving the System to use k_min.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    for key in ("depot", "customers", "fleet"):
        if key not in data:
            raise ValueError(f"{path.name}: missing required key {key!r}")

    depot = Depot(**data["depot"])
    customers = [Customer(**c) for c in data["customers"]]
    fleet = Fleet(**data["fleet"])

    return System(depot=depot, customers=customers, fleet=fleet)


def load_instance_csv(
    path: str | Path,
    capacity: float,
    num_vehicles: int | None = None,
    depot_id: int | str = 0,
) -> System:
    """Read a node table from CSV and combine it with fleet parameters.

    Requires columns `id`, `x`, `y`, `demand`. The row whose id matches
    `depot_id` becomes the depot; every other row becomes a customer.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return _instance_from_rows(rows, path.name, capacity, num_vehicles, depot_id)


def load_instance_csv_text(
    text: str,
    capacity: float,
    num_vehicles: int | None = None,
    depot_id: int | str = 0,
    label: str = "uploaded.csv",
) -> System:
    """Same as `load_instance_csv` for CSV already in memory — an uploaded file
    in the Streamlit app. `label` only appears in error messages."""
    rows = list(csv.DictReader(text.lstrip("﻿").splitlines()))
    return _instance_from_rows(rows, label, capacity, num_vehicles, depot_id)


def _instance_from_rows(
    rows: list[dict],
    label: str,
    capacity: float,
    num_vehicles: int | None,
    depot_id: int | str,
) -> System:
    if not rows:
        raise ValueError(f"{label}: file has a header but no data rows")

    columns = rows[0].keys()
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ValueError(
            f"{label}: missing required column(s) {missing} "
            f"(found: {sorted(c for c in columns if c)})"
        )

    depot = None
    customers = []
    for line_number, row in enumerate(rows, start=2):  # row 1 is the header
        node_id = _parse_id(row[ID_COLUMN], label, line_number)
        x = _parse_float(row[X_COLUMN], X_COLUMN, label, line_number)
        y = _parse_float(row[Y_COLUMN], Y_COLUMN, label, line_number)
        demand = _parse_float(row[DEMAND_COLUMN], DEMAND_COLUMN, label, line_number)

        if node_id == depot_id:
            if depot is not None:
                raise ValueError(f"{label} line {line_number}: depot {depot_id!r} appears twice")
            if demand != 0:
                raise ValueError(
                    f"{label} line {line_number}: depot {depot_id!r} must have demand 0, "
                    f"got {demand}"
                )
            depot = Depot(id=node_id, x=x, y=y)
        else:
            customers.append(Customer(id=node_id, x=x, y=y, demand=demand))

    if depot is None:
        raise ValueError(
            f"{label}: no row with id {depot_id!r} — one row must be the depot, "
            f"or pass a different depot_id"
        )

    return System(
        depot=depot,
        customers=customers,
        fleet=Fleet(capacity=capacity, num_vehicles=num_vehicles),
    )


def _parse_id(raw: str, label: str, line_number: int) -> int | str:
    """Ids are ints when they look like ints, strings otherwise — so a CSV
    written with 0/1/2 matches a depot_id of 0 rather than of '0'."""
    value = (raw or "").strip()
    if not value:
        raise ValueError(f"{label} line {line_number}: empty value in column 'id'")
    try:
        return int(value)
    except ValueError:
        return value


def _parse_float(raw: str, column: str, label: str, line_number: int) -> float:
    value = (raw or "").strip()
    if not value:
        raise ValueError(f"{label} line {line_number}: empty value in column {column!r}")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"{label} line {line_number}: column {column!r} is not a number ({raw!r})"
        ) from exc
