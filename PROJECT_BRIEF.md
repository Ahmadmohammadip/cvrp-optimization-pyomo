# Capacitated Vehicle Routing Optimization with Pyomo — Handoff Brief

## Purpose of this document
Handoff brief for Claude Code (or any engineer) to build this project from
scratch. It captures the locked scope, full mathematical formulation, repo
architecture, and phased build plan agreed on before any code was written.
No code exists yet — this is the starting specification. Companion piece
to two existing repos, `economic-dispatch-pyomo` and (in progress)
`battery-storage-optimization-pyomo` — same conventions and level of
polish, but a fully independent, standalone repo (no shared code).

## Goal
Public GitHub repo: an exact MILP solution to the Capacitated Vehicle
Routing Problem (CVRP) in Pyomo, built and committed phase by phase, each
phase tested and working before the next. Public from commit 1.

## Scope (locked decisions) — read this before writing any code
- **Variant**: Capacitated VRP (CVRP) only. Single depot, homogeneous
  vehicle fleet (all vehicles share the same capacity $Q$), each customer
  visited exactly once, minimize total route distance/cost. **No time
  windows, no multiple depots, no pickup-and-delivery** — those are
  explicitly out of scope (see Section 5).
- **Exactness vs. scale — this is the most important framing decision in
  the whole project**: CVRP is NP-hard (it generalizes the Traveling
  Salesman Problem). This repo solves it **exactly** via MILP — no
  heuristics, no metaheuristics, no approximation. That means it is only
  practical for small instances, realistically on the order of 15–25
  customers with a free solver (HiGHS or CBC), before branch-and-bound
  solve time becomes impractical. **This is not a bug to fix or hide —
  it is the point.** The README and `docs/formulation.md` must state this
  plainly and explain *why* (worst-case exponential branch-and-bound over
  an NP-hard problem), and the repo should include an actual benchmark
  table (solve time vs. instance size) that demonstrates the blow-up
  empirically rather than just asserting it. A portfolio piece that is
  honest about complexity-theoretic limits reads as more credible than
  one that pretends to have solved VRP at scale.
- **Repo structure**: installable package (`src/cvrp_opt`) + notebook +
  Streamlit demo app, same shape as the other two repos.
- **Solver**: pure MILP (binary + continuous variables, linear
  objective/constraints) — HiGHS as the default free solver; CBC as a
  documented fallback. No Ipopt needed (no nonlinear terms).

## 1. Mathematical Formulation

This is the classic **two-index vehicle-flow formulation with
Miller–Tucker–Zemlin (MTZ) subtour elimination**, as presented in
standard VRP references (e.g. Toth & Vigo, *The Vehicle Routing Problem*).
It is a well-known textbook formulation, not something invented for this
project — cite it as such in `docs/formulation.md`.

### 1.1 Sets and indices

| Symbol | Description |
|---|---|
| $V = \{0, 1, \dots, n\}$ | Nodes: $0$ is the depot, $1, \dots, n$ are customers |
| $V_C = \{1, \dots, n\}$ | Customer nodes only (depot excluded) |
| $A$ | Arcs $(i, j)$ for $i \ne j$, $i, j \in V$ |

### 1.2 Parameters

| Symbol | Description |
|---|---|
| $c_{ij}$ | Cost (distance) of arc $(i,j)$ |
| $d_i$ | Demand of customer $i$ ($d_0 = 0$ for the depot) |
| $Q$ | Vehicle capacity (homogeneous fleet) |
| $K$ | Number of vehicles available |

### 1.3 Decision variables

| Symbol | Description |
|---|---|
| $x_{ij} \in \{0, 1\}$ | 1 if some vehicle traverses arc $(i,j)$ directly, else 0 |
| $u_i \in \mathbb{R}$ | MTZ auxiliary variable — cumulative demand delivered on the route up to and including customer $i$, for $i \in V_C$ |

### 1.4 Objective

$$
\min \sum_{(i,j) \in A} c_{ij} \, x_{ij}
$$

### 1.5 Constraints

**Degree constraints** — every customer has exactly one incoming and one
outgoing arc:

$$
\sum_{j \in V, j \ne i} x_{ij} = 1 \quad \forall i \in V_C, \qquad \sum_{i \in V, i \ne j} x_{ij} = 1 \quad \forall j \in V_C
$$

**Depot degree** — exactly $K$ vehicles leave and return:

$$
\sum_{j \in V_C} x_{0j} = K, \qquad \sum_{i \in V_C} x_{i0} = K
$$

**MTZ subtour elimination + capacity** (this single family of
constraints does double duty: it eliminates subtours *and* enforces the
capacity limit):

$$
u_i - u_j + Q \, x_{ij} \le Q - d_j \quad \forall i, j \in V_C, \, i \ne j
$$

$$
d_i \le u_i \le Q \quad \forall i \in V_C
$$

### 1.6 Fleet size parameter $K$

$K$ should be a configurable input, defaulting to the theoretical minimum
$K_{min} = \lceil \sum_i d_i / Q \rceil$ (the fewest vehicles that could
possibly cover total demand, ignoring routing). Allow $K > K_{min}$ as a
user-supplied option — more vehicles than strictly necessary is a valid,
common scenario. **Do not silently compute $K$ without exposing it** — an
infeasible instance because $K$ is too small should fail with a clear
message, not an opaque solver infeasibility (same "fail loud at
construction" philosophy as the other two repos).

### 1.7 Route reconstruction (needed post-solve, not part of the MILP itself)

The solved $x_{ij}$ values are a set of arcs, not directly a list of
routes. A post-processing step must trace each vehicle's path starting
from a depot-departure arc ($x_{0j} = 1$) until it returns to the depot
($x_{i0} = 1$), producing an ordered list of customer visits per vehicle.
This reconstruction function belongs in `solve.py`, and needs its own
test (`test_route_reconstruction.py`) — it's a common source of silent
bugs (e.g. mishandling a depot node that's visited more than twice if
$K > $ the number of distinct subtours found).

## 2. Repo architecture

```
cvrp-optimization-pyomo/
├── README.md
├── LICENSE (MIT)
├── pyproject.toml
├── PROJECT_BRIEF.md
├── src/
│   └── cvrp_opt/
│       ├── __init__.py
│       ├── data/
│       │   ├── schema.py         # Customer, Depot, Fleet, System (validated dataclasses)
│       │   └── loaders.py        # CSV/JSON -> validated data objects
│       ├── model/
│       │   ├── __init__.py
│       │   └── builder.py        # Pyomo ConcreteModel construction (MTZ formulation)
│       ├── solve.py               # solver interface, route reconstruction, result dataclass
│       └── viz.py                 # route map plot, capacity utilization chart, solve-time benchmark chart
├── data/
│   └── sample_instances/          # small synthetic instances (5, 10, 15, 20, 25 customers)
├── notebooks/
│   └── 01_walkthrough.ipynb
├── app/
│   └── streamlit_app.py           # place depot/customers on a map (or upload), see optimal routes
├── tests/
│   ├── test_degree_constraints.py
│   ├── test_capacity_constraint.py
│   ├── test_route_reconstruction.py
│   └── test_integration.py
├── .github/workflows/ci.yml       # ruff + pytest, HiGHS only
└── docs/
    └── formulation.md             # Section 1 above, rendered, MTZ formulation cited properly
```

**Design rationale** (carried over from the other two repos): `data/schema.py`
validated dataclasses, never raw dicts touching the model layer. A
`System` should fail loudly at construction (negative demand, demand
exceeding $K \times Q$ even under perfect packing, duplicate customer
IDs, $K < K_{min}$) rather than surfacing as a solver infeasibility.

## 3. Build plan (phased)

| Phase | Scope | Output |
|---|---|---|
| 1 | Core MILP: degree constraints + MTZ subtour/capacity, small hardcoded instance (5–6 customers) | Working MILP, solved and sanity-checked by hand/visually |
| 2 | Route reconstruction (arcs → ordered per-vehicle routes) + validation tests | `solve.py` route extraction, capacity respected per route |
| 3 | Data schema + loaders (JSON/CSV instance format), `System` validation | `data/schema.py`, `data/loaders.py` |
| 4 | Sample instance library (5/10/15/20/25 synthetic customer sets) | `data/sample_instances/` |
| 5 | `viz.py` — route map (matplotlib), capacity utilization per vehicle | Visual output |
| 6 | Solve-time benchmark across instance sizes — the empirical proof of the "exact but doesn't scale" framing | A results table/plot in the README, generated from actual runs, not guessed numbers |
| 7 | Notebook walkthrough + Streamlit app | Interactive demo |
| 8 | Tests, CI, README, `docs/formulation.md` polish | GitHub-ready |

Each phase should leave `main` green (tests passing) before moving to the
next, and correspond to its own commit(s).

**Important for whoever runs Phase 6**: the benchmark numbers must come
from actually running the solver on real instances of increasing size,
not from estimation or claimed-from-memory figures — this is exactly the
kind of "never fabricate data" situation worth being careful about, since
the whole point of that phase is to present real evidence.

## 4. Explicitly out of scope (do not build unless asked)
- Time windows (VRPTW)
- Multiple depots
- Pickup-and-delivery
- Heterogeneous fleet (different vehicle capacities/costs)
- Any heuristic or metaheuristic solver (nearest-neighbor, 2-opt, genetic
  algorithms, etc.) — this repo is exact-MILP-only by design; a
  heuristic companion could be a good *separate* future repo, not a
  feature of this one
- Lazy/branch-and-cut subtour elimination (requires solver callback
  support, e.g. Gurobi) — MTZ only, since the goal is a free-solver
  (HiGHS/CBC), fully-reproducible repo

## 5. Git conventions
- One phase per commit (or a few if a phase is large), each commit
  leaves `main` green
- Commit message prefixes: `feat` / `test` / `docs` / `ci` / `chore` / `fix`
- Public repo from commit 1
- Suggested repo name: `cvrp-optimization-pyomo`

## 6. Provenance note
This brief was authored directly in this conversation as a planning
document, before any code was written. The MTZ formulation in Section 1
is a standard, well-established textbook formulation (see Toth & Vigo,
*The Vehicle Routing Problem*, SIAM, 2002) — cited as such, not claimed as
original. Nothing in this document should be treated as already
implemented or as verified benchmark data; the solve-time benchmarks in
Phase 6 must be generated from real runs when that phase is built.
