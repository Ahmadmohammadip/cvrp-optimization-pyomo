"""Phase 6: the benchmark harness.

These test the harness, not the published numbers — the committed results in
docs/benchmark_results.json come from a real run at the full time limit, which
is far too slow for CI. Here the 5-customer instance stands in.
"""

import json
from pathlib import Path

import pytest

from cvrp_opt.benchmark import environment, render_markdown_table, run_benchmark

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample_instances"
RESULTS_PATH = REPO_ROOT / "docs" / "benchmark_results.json"


def test_run_benchmark_records_a_solved_instance():
    rows = run_benchmark([SAMPLE_DIR / "cvrp_05.json"], time_limit=60, verbose=False)

    assert len(rows) == 1
    row = rows[0]
    assert row["instance"] == "cvrp_05"
    assert row["n_customers"] == 5
    assert row["proven"] is True
    assert row["solve_time"] > 0
    assert row["lower_bound"] <= row["total_distance"] + 1e-6
    assert row["vehicles_used"] >= row["k_min"]


def test_environment_block_names_what_produced_the_numbers():
    env = environment(time_limit=300)

    # A solve time without a machine attached to it is not a measurement.
    for key in ("solver", "python_version", "platform", "time_limit_seconds"):
        assert env[key]
    assert env["time_limit_seconds"] == 300
    assert "lower bound" in env["note"]


def test_markdown_table_marks_time_limited_rows():
    rows = [
        {
            "n_customers": 5,
            "vehicles_used": 1,
            "total_distance": 138.9,
            "lower_bound": 138.9,
            "gap": 0.0,
            "solve_time": 0.3,
            "proven": True,
        },
        {
            "n_customers": 25,
            "vehicles_used": 6,
            "total_distance": 900.0,
            "lower_bound": 600.0,
            "gap": 0.3333,
            "solve_time": 300.0,
            "proven": False,
        },
    ]

    table = render_markdown_table(rows)

    assert "| 5 |" in table
    assert "hit time limit" in table
    # A stopped run bounds the solve time from below; it does not measure it.
    assert "&gt;300s" in table
    assert "0.3s" in table


def test_markdown_table_sorts_by_instance_size():
    rows = [
        {
            "n_customers": n,
            "vehicles_used": 1,
            "total_distance": 1.0,
            "lower_bound": 1.0,
            "gap": 0.0,
            "solve_time": 1.0,
            "proven": True,
        }
        for n in (20, 5, 10)
    ]

    lines = render_markdown_table(rows).splitlines()[2:]
    sizes = [int(line.split("|")[1].strip()) for line in lines]

    assert sizes == [5, 10, 20]


@pytest.mark.skipif(
    not RESULTS_PATH.exists(), reason="benchmark has not been run in this checkout"
)
def test_committed_results_are_internally_consistent():
    # Guards against the published table drifting from the data behind it.
    payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    assert payload["environment"]["solver"]
    assert payload["results"]

    for row in payload["results"]:
        assert row["lower_bound"] <= row["total_distance"] + 1e-6
        assert row["vehicles_used"] >= row["k_min"]
        assert row["proven"] == (row["termination"] == "optimal")
        if not row["proven"]:
            assert row["gap"] > 0
