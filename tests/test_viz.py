"""Phase 5: plotting helpers. These check that each figure builds and carries
the right caveat — not how it looks.

The Agg backend is selected in conftest.py so these run headless.
"""

from pathlib import Path

import pytest

from cvrp_opt.data.loaders import load_instance_json
from cvrp_opt.model.builder import build_from_system
from cvrp_opt.solve import solve_cvrp
from cvrp_opt.viz import plot_benchmark, plot_capacity_utilization, plot_routes

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_instances"


@pytest.fixture(scope="module")
def solved():
    system = load_instance_json(SAMPLE_DIR / "cvrp_10.json")
    return system, solve_cvrp(build_from_system(system))


@pytest.mark.parametrize("plot_fn", [plot_routes, plot_capacity_utilization])
def test_plots_build(solved, plot_fn):
    system, result = solved
    fig = plot_fn(system, result)
    assert fig.axes


def test_route_map_states_optimality_in_the_title(solved):
    system, result = solved
    fig = plot_routes(system, result)

    title = fig.axes[0].get_title()

    assert "proven optimal" in title
    assert "not proven" not in title


def test_route_map_flags_an_unproven_solution():
    # A route map looks equally convincing whether or not optimality was
    # proven, so the caveat has to be on the figure itself.
    system = load_instance_json(SAMPLE_DIR / "cvrp_15.json")
    result = solve_cvrp(build_from_system(system), time_limit=2)

    title = plot_routes(system, result).axes[0].get_title()

    assert "not proven optimal" in title
    assert "gap" in title


def test_route_map_draws_one_line_per_vehicle(solved):
    system, result = solved
    ax = plot_routes(system, result).axes[0]

    route_lines = [line for line in ax.get_lines() if line.get_label().startswith("Vehicle")]

    assert len(route_lines) == result.vehicles_used


def test_capacity_chart_marks_the_limit(solved):
    system, result = solved
    ax = plot_capacity_utilization(system, result).axes[0]

    capacity_lines = [
        line for line in ax.get_lines() if line.get_label() == "Capacity"
    ]

    assert len(capacity_lines) == 1
    assert capacity_lines[0].get_ydata()[0] == pytest.approx(system.fleet.capacity)


def test_capacity_chart_has_one_bar_per_vehicle(solved):
    system, result = solved
    ax = plot_capacity_utilization(system, result).axes[0]

    assert len(ax.containers[0]) == result.vehicles_used


def _benchmark_rows():
    return [
        {"n_customers": 5, "solve_time": 0.05, "proven": True},
        {"n_customers": 10, "solve_time": 1.9, "proven": True},
        {"n_customers": 15, "solve_time": 42.0, "proven": True},
        {"n_customers": 20, "solve_time": 300.0, "proven": False},
    ]


def test_benchmark_chart_builds_and_uses_a_log_scale():
    ax = plot_benchmark(_benchmark_rows()).axes[0]

    assert ax.get_yscale() == "log"
    assert list(ax.get_xticks()) == [5, 10, 15, 20]


def test_benchmark_chart_separates_proven_from_time_limited():
    ax = plot_benchmark(_benchmark_rows()).axes[0]

    labels = [c.get_label() for c in ax.collections]

    assert "Proven optimal" in labels
    assert any("time limit" in label for label in labels)


def test_benchmark_chart_accepts_an_all_proven_run():
    rows = [r for r in _benchmark_rows() if r["proven"]]

    labels = [c.get_label() for c in plot_benchmark(rows).axes[0].collections]

    assert "Proven optimal" in labels
    assert not any("time limit" in label for label in labels)


def test_benchmark_chart_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one benchmark row"):
        plot_benchmark([])
