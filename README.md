# Capacitated Vehicle Routing with Pyomo — exact MILP

An **exact** solution to the Capacitated Vehicle Routing Problem (CVRP), built
with [Pyomo](https://www.pyomo.org/): single depot, homogeneous fleet, every
customer visited once, minimizing total distance. Two-index vehicle-flow
formulation with Miller–Tucker–Zemlin subtour elimination.

Exact means no heuristics, no metaheuristics, no approximation. When the solver
reports optimal, no shorter set of routes exists.

**It also means this does not scale, and that is the point.** CVRP is NP-hard —
it generalizes the Travelling Salesman Problem — so exact methods hit a wall
fast. This repo measures that wall rather than talking around it.

> Built incrementally, phase by phase — see `PROJECT_BRIEF.md` for the scope
> agreed before any code was written, and `docs/formulation.md` for the full
> formulation, citations, and assumptions.

## The wall, measured

Real runs of `python -m cvrp_opt.benchmark --time-limit 300` on the synthetic
sample instances. Raw data and the machine that produced it:
[`docs/benchmark_results.json`](docs/benchmark_results.json).

| Customers | Vehicles | Best distance | Lower bound | Gap | Solve time | Proven optimal |
|---:|---:|---:|---:|---:|---:|:--|
| 5 | 1 | 138.9 | 138.9 | 0.0% | 0.3s | yes |
| 10 | 3 | 326.9 | 326.9 | 0.0% | 1.9s | yes |
| 15 | 3 | 460.4 | 460.4 | 0.0% | 43.8s | yes |
| 20 | 5 | 598.8 | 427.2 | 28.7% | &gt;300s | **no — hit time limit** |
| 25 | 6 | 779.5 | 457.5 | 41.3% | &gt;300s | **no — hit time limit** |

Windows 11 (AMD64), Python 3.12.10, Pyomo 6.10.1, HiGHS via highspy 1.15.1,
one run per instance.

`>300s` is not a solve time. Those runs were stopped, so all the number says is
that a proof would have taken *at least* that long. The distance column for
those rows is the best tour found — possibly optimal; what is missing is the
proof, not necessarily the answer.

Ten to fifteen customers multiplies solve time by about 23. Fifteen to twenty
does not multiply it by anything measurable, because the run stops finishing.

Worth noticing *where* the time goes: on the 20-customer instance the best tour
of 598.8 is found within about 10 seconds and never improves over the remaining
290. Nearly all of that time is spent trying to raise a lower bound that MTZ's
weak linear relaxation keeps far below the answer. The hard part is the proof,
not the search.

## Features

- **Exact MTZ formulation** — degree constraints, fleet size, and one
  constraint family that does subtour elimination and capacity at once. Cited
  to Toth & Vigo rather than presented as original work.
- **Honest solver reporting** — a run that hits its time limit returns the best
  routes found, flagged `is_optimal=False` with the gap and lower bound. It is
  never dressed up as an optimum, and the route map says so in its title.
- **Fleet size that behaves** — `≤ K` rather than `= K`, so offering more
  vehicles than needed never makes the answer worse (see below).
- **Defensive route reconstruction** — turning arcs back into routes is easy to
  get quietly wrong, so it raises on dead ends, shared customers, and subtours
  rather than returning a plausible partial answer.
- **Validated instances** — a `System` fails at construction on duplicate ids,
  a demand no vehicle could carry, or a fleet too small for total demand.
  Getting that news up front matters when a solve can run for minutes.

## One departure from the brief

`PROJECT_BRIEF.md` §1.5 writes the depot degree as exactly `K` vehicles. This
model uses `≤ K` plus an explicit balance constraint (`PROJECT_BRIEF.md` §7.1).

Under equality, every surplus vehicle is *forced* to make a trip, so giving a
planner a larger fleet lengthens the total tour. That contradicts §1.6 of the
brief, which calls running above the minimum "a valid, common scenario". With
`≤ K`, unneeded vehicles stay parked — on the 10-customer instance, offering 3,
4, 5 or 6 vehicles all return the same 326.9 using 3.

The trade-off is real: freeing the depot count weakens the relaxation, so this
form solves more slowly than the equality one. Details in
`docs/formulation.md` §5.2.

## Install

```bash
pip install -e ".[dev,solvers,viz]"
```

[HiGHS](https://highs.dev/) arrives with the `solvers` extra (`highspy`) and is
the only solver needed — the model is a pure MILP with no nonlinear terms.
[CBC](https://github.com/coin-or/Cbc) works as a fallback if you have the
binary on your PATH: pass `solver_name="cbc"` to `solve_cvrp`. CBC is not
installed in CI.

## Quickstart

```python
from cvrp_opt.data.loaders import load_instance_json
from cvrp_opt.model.builder import build_from_system
from cvrp_opt.solve import solve_cvrp

system = load_instance_json("data/sample_instances/cvrp_10.json")
result = solve_cvrp(build_from_system(system), time_limit=60)

print(result.summary())
# 326.90 total distance over 3 vehicle(s) — proven optimal, 1.9s

for i, route in enumerate(result.routes):
    print(f"Vehicle {i + 1}: depot -> {' -> '.join(map(str, route))} -> depot "
          f"(load {result.route_loads[i]:g})")

if not result.is_optimal:
    print(f"Time limit hit — within {result.gap:.1%} of optimal")
```

## Repo layout

```
cvrp-optimization-pyomo/
├── src/cvrp_opt/
│   ├── data/schema.py      # Customer, Depot, Fleet, System (validated)
│   ├── data/loaders.py     # JSON instances, CSV node tables
│   ├── model/builder.py    # Pyomo ConcreteModel (MTZ formulation)
│   ├── solve.py            # solver interface, route reconstruction, CVRPResult
│   ├── viz.py              # route map, capacity chart, benchmark chart
│   └── benchmark.py        # solve-time harness
├── data/sample_instances/  # synthetic 5/10/15/20/25-customer instances
├── notebooks/01_walkthrough.ipynb
├── app/streamlit_app.py
├── tests/
├── docs/formulation.md, benchmark_results.json
└── .github/workflows/ci.yml
```

## Interactive demo

```bash
streamlit run app/streamlit_app.py
```

Pick an instance or upload a node CSV, set capacity, fleet size, and a time
limit. The result always states whether optimality was proven.

## Tests

```bash
pytest -v
```

The suite solves only the 5- and 10-customer instances. The larger ones exist
for the benchmark — running them in CI would cost minutes and prove nothing the
benchmark does not already measure.

## Reproducing the benchmark

```bash
python -m cvrp_opt.benchmark --time-limit 300
```

Writes `docs/benchmark_results.json` with an environment block recording the
machine and versions. Expect roughly 11 minutes: the two largest instances will
use their full allowance.

## A note on the sample data

The instances in `data/sample_instances/` are **synthetic** — uniform random
points on a 100×100 square, demands 5–20, depot at the centre. They are not
TSPLIB, Christofides, or Uchoa instances, and objective values here are not
comparable to published results for those. The seeded generator is committed
alongside them.

Distances are exact Euclidean floats. TSPLIB's `EUC_2D` convention rounds to
integers, which is another reason numbers here will not line up with the
literature.

## Known limitations

Exact-MILP-only by design, so no heuristics or metaheuristics. MTZ only — lazy
or branch-and-cut subtour elimination needs solver callbacks free solvers do
not expose. Single depot, homogeneous fleet, no time windows, no
pickup-and-delivery, no split deliveries. Full list with rationale in
`docs/formulation.md`.

## Companion repos

Three standalone optimization models built to the same conventions — validated
dataclasses that fail loudly at construction, a Pyomo builder that never touches
raw files, and a result dataclass rather than a live model — but sharing no code.

- [economic-dispatch-pyomo](https://github.com/Ahmadmohammadip/economic-dispatch-pyomo)
  — multi-period, multi-bus DC-OPF economic dispatch with generator ramping,
  curtailable renewables, storage, and locational marginal prices.
- [battery-storage-optimization-pyomo](https://github.com/Ahmadmohammadip/battery-storage-optimization-pyomo)
  — battery energy arbitrage co-optimized with frequency regulation capacity
  (revenue stacking) as a single LP.

## License

MIT — see `LICENSE`.
