"""
Plotting helpers for a solved CVRP instance.

matplotlib only, kept dependency-light so these work headless in CI and inside
the Streamlit app. Each function returns a Figure — callers decide whether to
show(), save(), or hand it to st.pyplot().
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from cvrp_opt.data.schema import System
from cvrp_opt.solve import CVRPResult

# Qualitative palette: distinguishable at a glance and stable across figures,
# so vehicle 1 is the same colour on the route map and the capacity chart.
ROUTE_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#8c564b", "#17becf", "#e377c2", "#7f7f7f", "#bcbd22",
]


def _color(index: int) -> str:
    return ROUTE_COLORS[index % len(ROUTE_COLORS)]


def _status_note(result: CVRPResult) -> str:
    if result.is_optimal:
        return "proven optimal"
    return f"not proven optimal — gap {result.gap:.1%}"


def plot_routes(system: System, result: CVRPResult):
    """Map of the depot, the customers, and the route each vehicle drives.

    The title always states whether the solution was proven optimal. A route
    map looks equally convincing either way, which is exactly why the caveat
    belongs on the picture rather than only in the console.
    """
    coords = system.coords
    depot_id = system.depot.id
    fig, ax = plt.subplots(figsize=(8, 8))

    for index, route in enumerate(result.routes):
        path = [depot_id, *route, depot_id]
        xs = [coords[node][0] for node in path]
        ys = [coords[node][1] for node in path]
        load = result.route_loads[index] if index < len(result.route_loads) else 0.0
        ax.plot(
            xs,
            ys,
            color=_color(index),
            linewidth=1.8,
            alpha=0.9,
            zorder=1,
            label=f"Vehicle {index + 1} — load {load:g}/{system.fleet.capacity:g}",
        )
        # Direction markers, so a route is readable as a sequence rather than
        # an undirected shape.
        for k in range(len(path) - 1):
            ax.annotate(
                "",
                xy=(xs[k + 1], ys[k + 1]),
                xytext=(xs[k], ys[k]),
                arrowprops=dict(arrowstyle="-|>", color=_color(index), alpha=0.7, lw=1.2),
                zorder=1,
            )

    customer_x = [coords[c.id][0] for c in system.customers]
    customer_y = [coords[c.id][1] for c in system.customers]
    ax.scatter(customer_x, customer_y, s=60, color="white", edgecolor="black", zorder=2)
    for c in system.customers:
        ax.annotate(
            f"{c.id}",
            coords[c.id],
            fontsize=7,
            ha="center",
            va="center",
            zorder=3,
        )

    ax.scatter(
        [coords[depot_id][0]],
        [coords[depot_id][1]],
        s=220,
        marker="s",
        color="black",
        zorder=4,
        label="Depot",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Routes — {result.total_distance:.1f} total distance, "
        f"{result.vehicles_used} vehicle(s)\n{_status_note(result)}"
    )
    ax.legend(loc="best", fontsize="small")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return fig


def plot_capacity_utilization(system: System, result: CVRPResult):
    """Load carried by each vehicle against the capacity limit.

    Bars are coloured to match the route map. A bar crossing the capacity line
    would mean the model is broken, so the line is drawn rather than assumed.
    """
    capacity = system.fleet.capacity
    loads = result.route_loads
    labels = [f"V{i + 1}" for i in range(len(loads))]

    # Minimum width is set by the title, not the bars: a two-vehicle chart is
    # narrow enough that a longer title would be clipped.
    fig, ax = plt.subplots(figsize=(max(7, len(loads) * 0.9), 4.5))
    ax.bar(labels, loads, color=[_color(i) for i in range(len(loads))])
    ax.axhline(capacity, color="black", linestyle="--", linewidth=1.2, label="Capacity")

    for index, load in enumerate(loads):
        ax.annotate(
            f"{load / capacity:.0%}",
            (index, load),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylim(0, capacity * 1.18)
    ax.set_ylabel("Load carried")
    ax.set_title(
        f"Capacity utilization — {sum(loads):g}/{capacity * len(loads):g} "
        f"across {len(loads)} vehicle(s)"
    )
    ax.legend(loc="lower right", fontsize="small")
    fig.tight_layout()
    return fig


def plot_benchmark(rows: list[dict]):
    """Solve time against instance size, on a log scale.

    Expects the rows produced by `benchmark.run_benchmark`: each needs
    `n_customers`, `solve_time`, and `proven`. Points where optimality was
    proven are drawn filled; points that hit the time limit are hollow, because
    those are lower bounds on the true solve time, not measurements of it.

    Log scale because the growth is the point — on a linear axis the small
    instances collapse onto the x-axis and the shape is lost.
    """
    if not rows:
        raise ValueError("plot_benchmark needs at least one benchmark row")

    rows = sorted(rows, key=lambda r: r["n_customers"])
    sizes = [r["n_customers"] for r in rows]
    times = [r["solve_time"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, times, color="#1f77b4", linewidth=1.5, zorder=1)

    proven = [(n, t) for n, t, r in zip(sizes, times, rows, strict=True) if r["proven"]]
    unproven = [
        (n, t) for n, t, r in zip(sizes, times, rows, strict=True) if not r["proven"]
    ]

    if proven:
        ax.scatter(
            *zip(*proven, strict=True),
            s=70,
            color="#1f77b4",
            zorder=2,
            label="Proven optimal",
        )
    if unproven:
        ax.scatter(
            *zip(*unproven, strict=True),
            s=90,
            facecolor="none",
            edgecolor="#d62728",
            linewidth=2,
            zorder=2,
            label="Hit time limit (lower bound on solve time)",
        )

    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xlabel("Customers")
    ax.set_ylabel("Solve time (s, log scale)")
    ax.set_title("Exact CVRP solve time grows sharply with instance size")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize="small")
    fig.tight_layout()
    return fig
