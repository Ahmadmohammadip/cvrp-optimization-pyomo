"""Regenerates the sample instances in this directory.

THESE INSTANCES ARE SYNTHETIC. Customers are drawn uniformly at random from a
100x100 square with a depot at the centre, and demands uniformly from 5 to 20.
They are not TSPLIB, not Christofides, not Uchoa — none of the standard CVRP
benchmark sets — and objective values here are not comparable to published
results for those. They exist to exercise the model and to show how solve time
grows with instance size.

The generator is committed alongside the JSON so the data is inspectable
rather than magic. It is seeded per instance size, so re-running reproduces
the same files byte for byte.

Run with:  python data/sample_instances/generate_instances.py
"""

import json
import random
from pathlib import Path

HERE = Path(__file__).parent

SIZES = (5, 10, 15, 20, 25)
CAPACITY = 60.0
GRID = 100.0
DEPOT_AT = (50.0, 50.0)
DEMAND_RANGE = (5, 20)


def make_instance(n_customers: int) -> dict:
    # Seeded by size, so each file is reproducible and independent of the order
    # the sizes are generated in.
    rng = random.Random(n_customers)

    customers = []
    for i in range(1, n_customers + 1):
        customers.append(
            {
                "id": i,
                "x": round(rng.uniform(0, GRID), 2),
                "y": round(rng.uniform(0, GRID), 2),
                "demand": float(rng.randint(*DEMAND_RANGE)),
            }
        )

    return {
        "name": f"synthetic-{n_customers:02d}",
        "description": (
            f"Synthetic CVRP instance: {n_customers} customers uniform on a "
            f"{GRID:.0f}x{GRID:.0f} grid, depot at centre, demands {DEMAND_RANGE[0]}-"
            f"{DEMAND_RANGE[1]}. Not a standard benchmark instance."
        ),
        "depot": {"id": 0, "x": DEPOT_AT[0], "y": DEPOT_AT[1]},
        "customers": customers,
        # num_vehicles is deliberately omitted: the loader falls back to k_min,
        # which is what the benchmark measures.
        "fleet": {"capacity": CAPACITY},
    }


def main() -> None:
    for n in SIZES:
        instance = make_instance(n)
        path = HERE / f"cvrp_{n:02d}.json"
        path.write_text(json.dumps(instance, indent=2) + "\n", encoding="utf-8")

        total = sum(c["demand"] for c in instance["customers"])
        k_min = -(-int(total) // int(CAPACITY))  # ceil for ints
        print(f"wrote {path.name}: {n} customers, total demand {total:.0f}, k_min {k_min}")


if __name__ == "__main__":
    main()
