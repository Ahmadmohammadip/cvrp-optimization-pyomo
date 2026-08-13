"""
Solve-time benchmark across instance sizes.

This is the empirical half of the repo's central claim: CVRP is NP-hard, this
solves it exactly, therefore it does not scale. Asserting that is cheap;
measuring it is the point.

Every number this module produces comes from an actual solver run. Nothing is
estimated, interpolated, or carried over from a previous machine. The
environment block written alongside the results records what produced them,
because a solve time without a machine attached to it is not a measurement.

Rows that hit the time limit are reported as such, with their gap. Those are
lower bounds on the true solve time, not solve times — the run was stopped, so
we know the answer took *at least* that long, not how long it would have
taken.

Run with:  python -m cvrp_opt.benchmark --time-limit 300
"""

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from cvrp_opt.data.loaders import load_instance_json
from cvrp_opt.model.builder import build_from_system
from cvrp_opt.solve import DEFAULT_SOLVER, solve_cvrp

DEFAULT_TIME_LIMIT = 300.0
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSTANCE_DIR = REPO_ROOT / "data" / "sample_instances"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "benchmark_results.json"


@dataclass
class BenchmarkRow:
    instance: str
    n_customers: int
    k_min: int
    vehicles_used: int
    total_distance: float
    lower_bound: float
    gap: float
    solve_time: float
    proven: bool
    termination: str


def run_benchmark(
    paths: list[Path],
    time_limit: float = DEFAULT_TIME_LIMIT,
    solver_name: str = DEFAULT_SOLVER,
    verbose: bool = True,
) -> list[dict]:
    """Solve each instance once and record what happened.

    One run per instance, not a best-of-N: this measures the shape of the
    growth, and repeated runs of a deterministic solver on the same input add
    little beyond timing noise.
    """
    rows = []
    for path in paths:
        system = load_instance_json(path)
        if verbose:
            print(f"solving {path.name} ({system.n_customers} customers)...", flush=True)

        result = solve_cvrp(
            build_from_system(system), solver_name=solver_name, time_limit=time_limit
        )
        row = BenchmarkRow(
            instance=path.stem,
            n_customers=system.n_customers,
            k_min=system.k_min,
            vehicles_used=result.vehicles_used,
            total_distance=round(result.total_distance, 3),
            lower_bound=round(result.lower_bound, 3),
            gap=round(result.gap, 6),
            solve_time=round(result.solve_time, 2),
            proven=result.is_optimal,
            termination=result.termination,
        )
        rows.append(asdict(row))
        if verbose:
            print(f"  {result.summary()}", flush=True)

    return rows


def environment(solver_name: str = DEFAULT_SOLVER, time_limit: float = DEFAULT_TIME_LIMIT):
    """What produced the numbers. Without this the table is uninterpretable."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as package_version

    import pyomo.version

    # highspy does not expose __version__, so ask the package metadata.
    try:
        solver_version = package_version("highspy")
    except PackageNotFoundError:  # pragma: no cover - solvers extra not installed
        solver_version = "unknown"

    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "solver": solver_name,
        "highspy_version": solver_version,
        "pyomo_version": pyomo.version.__version__,
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "processor": platform.processor() or "unknown",
        "time_limit_seconds": time_limit,
        "note": (
            "Single run per instance. Rows with proven=false hit the time limit: "
            "their solve_time is a lower bound on the true time to optimality, not "
            "a measurement of it."
        ),
    }


def render_markdown_table(rows: list[dict]) -> str:
    """Render the results as the markdown table used in README and docs."""
    header = (
        "| Customers | Vehicles | Best distance | Lower bound | Gap | "
        "Solve time | Proven optimal |\n"
        "|---:|---:|---:|---:|---:|---:|:--|"
    )
    lines = [header]
    for row in sorted(rows, key=lambda r: r["n_customers"]):
        proven = "yes" if row["proven"] else "**no — hit time limit**"
        time_cell = (
            f"{row['solve_time']:.1f}s" if row["proven"] else f"&gt;{row['solve_time']:.0f}s"
        )
        lines.append(
            f"| {row['n_customers']} | {row['vehicles_used']} | "
            f"{row['total_distance']:.1f} | {row['lower_bound']:.1f} | "
            f"{row['gap']:.1%} | {time_cell} | {proven} |"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--time-limit",
        type=float,
        default=DEFAULT_TIME_LIMIT,
        help=f"seconds per instance (default: {DEFAULT_TIME_LIMIT:.0f})",
    )
    parser.add_argument(
        "--instances",
        type=Path,
        default=DEFAULT_INSTANCE_DIR,
        help="directory of instance JSON files",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="where to write results JSON"
    )
    args = parser.parse_args(argv)

    paths = sorted(args.instances.glob("cvrp_*.json"))
    if not paths:
        print(f"no instances found in {args.instances}", file=sys.stderr)
        return 1

    rows = run_benchmark(paths, time_limit=args.time_limit)
    payload = {"environment": environment(time_limit=args.time_limit), "results": rows}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {args.output}\n")
    print(render_markdown_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
