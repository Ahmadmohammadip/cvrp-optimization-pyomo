"""
Streamlit demo for the exact CVRP solver.

Pick a sample instance (or upload a node CSV), set capacity, fleet size, and a
time limit, then solve. The result is always labelled with whether optimality
was proven — an unproven answer is still useful, but it is not the same thing.

Solving is expensive enough that it happens on an explicit submit rather than
on every widget change, and results are cached so redrawing the page does not
re-solve.

Run with:  streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path

import streamlit as st

from cvrp_opt.data.loaders import load_instance_csv_text, load_instance_json
from cvrp_opt.data.schema import Fleet, System
from cvrp_opt.model.builder import build_from_system
from cvrp_opt.solve import solve_cvrp
from cvrp_opt.viz import plot_capacity_utilization, plot_routes

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_instances"

# Above roughly this size, an exact solve stops finishing in a demo-friendly
# time. The app still allows it — that limitation is the repo's whole point —
# but it says so first.
SLOW_INSTANCE_THRESHOLD = 12

st.set_page_config(page_title="Exact CVRP Solver", layout="wide")

st.title("Capacitated Vehicle Routing — exact MILP")
st.caption(
    "Two-index vehicle-flow formulation with Miller-Tucker-Zemlin subtour elimination, "
    "solved exactly with HiGHS. Exact means no heuristics — and therefore small: "
    "solve time grows sharply with the number of customers."
)


@st.cache_data(show_spinner=False)
def solve_instance(payload: str, capacity: float, num_vehicles, time_limit: float):
    """Cached solve. Keyed on the instance JSON plus the fleet settings, so
    moving an unrelated widget does not trigger a fresh branch-and-bound run."""
    data = json.loads(payload)
    base = load_instance_json_from_dict(data)
    system = System(
        depot=base.depot,
        customers=base.customers,
        fleet=Fleet(capacity=capacity, num_vehicles=num_vehicles),
    )
    result = solve_cvrp(build_from_system(system), time_limit=time_limit)
    return system, result


def load_instance_json_from_dict(data: dict) -> System:
    """Round-trip a dict through the JSON loader so uploads and samples share
    one validation path."""
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as handle:
        json.dump(data, handle)
        temp_path = handle.name
    try:
        return load_instance_json(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)


sample_files = sorted(SAMPLE_DIR.glob("cvrp_*.json"))

with st.sidebar:
    st.header("Instance")
    source = st.radio(
        "Source", ["Sample instance", "Upload node CSV"], label_visibility="collapsed"
    )

    payload = None
    if source == "Sample instance":
        choice = st.selectbox(
            "Instance",
            sample_files,
            # int() strips the zero padding: "5 customers", not "05 customers".
            format_func=lambda p: f"{p.stem} ({int(p.stem.split('_')[1])} customers)",
        )
        payload = choice.read_text(encoding="utf-8")
    else:
        uploaded = st.file_uploader(
            "CSV with columns `id`, `x`, `y`, `demand` — the row with id 0 is the depot",
            type="csv",
        )
        if uploaded is not None:
            payload = uploaded

    st.header("Fleet")
    capacity = st.slider("Vehicle capacity", 10.0, 200.0, 60.0, step=5.0)
    override_fleet = st.checkbox(
        "Set fleet size manually",
        value=False,
        help="Unchecked uses k_min, the fewest vehicles that could cover total demand.",
    )
    num_vehicles = (
        st.slider("Vehicles available (K)", 1, 12, 4) if override_fleet else None
    )

    st.header("Solver")
    time_limit = st.slider(
        "Time limit (seconds)",
        5,
        120,
        30,
        step=5,
        help="Hitting it is not an error — you get the best routes found so far, plus a gap.",
    )

    submitted = st.button("Solve", type="primary", use_container_width=True)


if payload is None:
    st.info("Upload a node CSV to solve, or switch back to a sample instance.")
    st.stop()

# Normalize an uploaded CSV into the same JSON payload the samples use, so the
# cache key and the solve path are identical for both.
if not isinstance(payload, str):
    try:
        preview = load_instance_csv_text(
            payload.getvalue().decode("utf-8"),
            capacity=capacity,
            num_vehicles=num_vehicles,
            label=payload.name,
        )
    except ValueError as exc:
        st.error(f"Could not read {payload.name}: {exc}")
        st.stop()
    payload = json.dumps(
        {
            "depot": {"id": preview.depot.id, "x": preview.depot.x, "y": preview.depot.y},
            "customers": [
                {"id": c.id, "x": c.x, "y": c.y, "demand": c.demand}
                for c in preview.customers
            ],
            "fleet": {"capacity": capacity},
        }
    )

n_customers = len(json.loads(payload)["customers"])
if n_customers > SLOW_INSTANCE_THRESHOLD:
    st.warning(
        f"This instance has {n_customers} customers. Exact CVRP is NP-hard, and past "
        f"roughly {SLOW_INSTANCE_THRESHOLD} customers a proof of optimality usually "
        f"outlasts the time limit. Expect to wait the full {time_limit}s and get a "
        f"good-but-unproven answer."
    )

if not submitted:
    st.info("Set the fleet and time limit, then press **Solve**.")
    st.stop()

try:
    with st.spinner(f"Solving — up to {time_limit}s..."):
        system, result = solve_instance(payload, capacity, num_vehicles, float(time_limit))
except ValueError as exc:
    st.error(f"Invalid instance: {exc}")
    st.stop()
except RuntimeError as exc:
    st.error(f"{exc}")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total distance", f"{result.total_distance:,.1f}")
col2.metric("Vehicles used", f"{result.vehicles_used} of {system.num_vehicles}")
col3.metric("Solve time", f"{result.solve_time:.1f}s")
col4.metric("Optimality gap", f"{result.gap:.2%}")

if result.is_optimal:
    st.success(
        "Proven optimal: no shorter set of routes exists for this instance. "
        "(HiGHS proves optimality to within its default relative gap tolerance.)"
    )
else:
    st.warning(
        f"**Not proven optimal.** The solver hit the {time_limit}s limit. These are the "
        f"best routes it found; the true optimum is at least "
        f"{result.lower_bound:,.1f}, so this is within {result.gap:.1%} of it. "
        f"A longer limit narrows the gap."
    )

left, right = st.columns([3, 2])
with left:
    st.pyplot(plot_routes(system, result))
with right:
    st.pyplot(plot_capacity_utilization(system, result))
    st.subheader("Routes")
    for index, route in enumerate(result.routes):
        st.markdown(
            f"**Vehicle {index + 1}** — load {result.route_loads[index]:g}/"
            f"{system.fleet.capacity:g}, distance {result.route_distances[index]:.1f}  \n"
            f"depot → {' → '.join(str(c) for c in route)} → depot"
        )

with st.expander("Why this does not scale"):
    st.markdown(
        """
CVRP generalizes the Travelling Salesman Problem, so it is NP-hard. This repo
solves it **exactly** with branch-and-bound — no heuristics, no approximation —
which means worst-case exponential work as customers are added.

The MTZ subtour formulation compounds it: it is compact and readable, but its
linear relaxation is weak, so the lower bound climbs slowly and the search tree
stays large. Stronger formulations exist and need solver callbacks that free
solvers do not expose, which is out of scope here by design.

See `docs/formulation.md` and the measured benchmark table in the README.
"""
    )
