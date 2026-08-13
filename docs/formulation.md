# Formulation

An exact mixed-integer linear program for the Capacitated Vehicle Routing
Problem: single depot, homogeneous fleet, every customer visited exactly once,
minimizing total distance.

## Provenance

This is the **two-index vehicle-flow formulation with Miller–Tucker–Zemlin
(MTZ) subtour elimination** — a standard, well-established textbook
formulation, not one devised for this project.

The MTZ constraints originate in Miller, Tucker & Zemlin, "Integer Programming
Formulation of Traveling Salesman Problems", *Journal of the ACM* 7(4), 1960.
Their adaptation to the capacitated vehicle routing problem, in the form used
here, is presented in Toth & Vigo, *The Vehicle Routing Problem* (SIAM
Monographs on Discrete Mathematics and Applications, 2002), the standard
reference for the field.

## 1. Sets and indices

| Symbol | Description | Code |
|---|---|---|
| $V = \{0, 1, \dots, n\}$ | Nodes; $0$ is the depot | `m.V` |
| $V_C = \{1, \dots, n\}$ | Customers only | `m.V_C` |
| $A$ | Arcs $(i,j)$, $i \neq j$ | `m.A` |

## 2. Parameters

| Symbol | Description | Code |
|---|---|---|
| $c_{ij}$ | Cost (distance) of arc $(i,j)$ | `m.cost` |
| $d_i$ | Demand of customer $i$, with $d_0 = 0$ | `m.demand` |
| $Q$ | Vehicle capacity (homogeneous fleet) | `m.capacity` |
| $K$ | Vehicles available | `m.num_vehicles` |

## 3. Decision variables

| Symbol | Description | Code |
|---|---|---|
| $x_{ij} \in \{0,1\}$ | 1 if a vehicle traverses arc $(i,j)$ | `m.x` |
| $u_i \in \mathbb{R}$ | Cumulative load delivered up to and including customer $i$ | `m.u` |

## 4. Objective

$$
\min \sum_{(i,j) \in A} c_{ij} \, x_{ij}
$$

## 5. Constraints

### 5.1 Degree constraints

Every customer is entered once and left once:

$$
\sum_{j \in V, j \neq i} x_{ij} = 1 \quad \forall i \in V_C, \qquad
\sum_{i \in V, i \neq j} x_{ij} = 1 \quad \forall j \in V_C
$$

### 5.2 Depot degree — `≤ K`, not `= K`

$$
\sum_{j \in V_C} x_{0j} \le K, \qquad \sum_{j \in V_C} x_{0j} = \sum_{i \in V_C} x_{i0}
$$

**This departs from `PROJECT_BRIEF.md` §1.5**, which writes both depot degrees
as exactly $K$. See §7.1 of the brief for the amendment.

The reason: §1.6 of the brief calls running more vehicles than the minimum "a
valid, common scenario". Under equality, every surplus vehicle is *forced* to
make a trip, so offering a larger fleet can only lengthen the total tour —
handing a planner more resources makes their answer worse, which is not what
anyone means by having spare vehicles. Under `≤ K` the solver parks what it
does not need.

Measured on the 10-customer sample instance, offering vehicles beyond $k_{min}$
leaves both the routes and the objective unchanged; under equality the same
instance gets steadily more expensive. The notebook reproduces this directly.

Equality also delivered depot balance for free — $K$ out, $K$ back. Inequality
does not, so the balance constraint above is stated explicitly. Without it the
arcs could dangle: departures without matching returns.

The cost is real. Freeing the depot count weakens the linear relaxation, so
this formulation solves more slowly than the equality form. That is a
deliberate trade of solve time for a model that behaves sensibly.

A second consequence: with `≤ K` there is no need to require $K \le n$.
Surplus vehicles beyond the number of customers simply go unused instead of
making the instance infeasible.

### 5.3 MTZ subtour elimination and capacity

$$
u_i - u_j + Q \, x_{ij} \le Q - d_j \quad \forall i, j \in V_C, \, i \neq j
$$

$$
d_i \le u_i \le Q \quad \forall i \in V_C
$$

One constraint family doing two jobs. Read it twice:

**As capacity.** When $x_{ij} = 1$ it collapses to $u_j \ge u_i + d_j$: load
accumulates along a route. Since $u_i \le Q$ everywhere, no route can carry
more than $Q$. When $x_{ij} = 0$ it reduces to $u_i - u_j \le Q - d_j$, which
holds automatically given the bounds — so the constraint is inactive and does
not distort anything.

**As subtour elimination.** Sum the inequalities around any cycle that avoids
the depot. The $u$ terms telescope to zero, leaving $0 \ge \sum d_j$ over the
cycle — false whenever demands are positive. So customer-only cycles cannot
exist. Two-customer cycles are covered by the same argument: $i \to j$ and
$j \to i$ together give $0 \ge d_i + d_j$.

The depot has no $u$ variable, which is precisely what lets routes pass
through it while forbidding cycles that avoid it.

### 5.4 What is *not* constrained

There is no "at least one vehicle must leave the depot" constraint, and none is
needed. Every customer requires an incoming arc; the only way to supply one
without forming a depot-free cycle is to connect back to the depot. Capacity
then forces the vehicle count up to $k_{min}$ on its own. It looks like an
omission and is not — hence this note, and the test that pins the behavior.

## 6. Fleet size $K$

$K$ defaults to the theoretical minimum

$$
k_{min} = \left\lceil \frac{\sum_i d_i}{Q} \right\rceil
$$

which is a lower bound only: it assumes perfect packing and ignores geography,
so a real instance may need more. $K$ is never silently computed and hidden —
`System.num_vehicles` always reports the value in force, and a fleet too small
for total demand is rejected at construction with a message naming the
shortfall, rather than surfacing as an opaque solver infeasibility.

## 7. Distances

Arc costs are **exact Euclidean distances kept as floats**. TSPLIB's `EUC_2D`
convention rounds distances to integers, and much of the published CVRP
literature reports objective values under that convention. Values produced here
are therefore not directly comparable to those results — the difference is the
rounding, not the routing.

## 8. Why this does not scale

CVRP generalizes the Travelling Salesman Problem, so it is **NP-hard**. Solving
it exactly means branch-and-bound, whose worst case is exponential in the
instance size. No amount of implementation care changes that; it is a property
of the problem, not of the code.

MTZ compounds it. The formulation is compact — $O(n^2)$ constraints, no
exponential family to separate — and readable, which is why it is the standard
teaching formulation. But its linear relaxation is **weak**: the bound it gives
is far below the true optimum, so branch-and-bound must explore a large tree
before it can prove anything. In the measurements below, the 20-customer
instance had found its final incumbent long before the time limit, yet the
lower bound was still nearly 29% away — almost all of that time was spent
trying to *prove* optimality, not to find the answer.

Stronger alternatives exist — two-commodity flow formulations, and set
partitioning with column generation — as does branch-and-cut with lazy subtour
constraints. The latter needs solver callbacks that free solvers do not expose,
which is out of scope here by design: the goal is a fully reproducible repo on
a free solver.

### Measured results

Produced by `python -m cvrp_opt.benchmark --time-limit 300` on the synthetic
sample instances. Raw data, including the machine and versions that produced
it, is in [`benchmark_results.json`](benchmark_results.json). Every number
below is from an actual solver run.

| Customers | Vehicles | Best distance | Lower bound | Gap | Solve time | Proven optimal |
|---:|---:|---:|---:|---:|---:|:--|
| 5 | 1 | 138.9 | 138.9 | 0.0% | 0.3s | yes |
| 10 | 3 | 326.9 | 326.9 | 0.0% | 1.9s | yes |
| 15 | 3 | 460.4 | 460.4 | 0.0% | 43.8s | yes |
| 20 | 5 | 598.8 | 427.2 | 28.7% | &gt;300s | **no — hit time limit** |
| 25 | 6 | 779.5 | 457.5 | 41.3% | &gt;300s | **no — hit time limit** |

Measured on Windows 11 (AMD64), Python 3.12.10, Pyomo 6.10.1, HiGHS via
highspy 1.15.1, one run per instance.

Read the last two rows carefully. `>300s` is not a solve time — the run was
stopped, so all it says is that proving optimality would have taken *at least*
that long. The distance column for those rows is the best tour found, which may
well be optimal; what is missing is the proof.

The shape is the point. Going from 10 to 15 customers multiplies solve time by
roughly 23. Going from 15 to 20 does not multiply it by anything measurable
here, because the run no longer finishes. Five customers is the difference
between "instant" and "gave up".

Note also how much of the difficulty is proof rather than search. On the
20-customer instance the incumbent of 598.8 is already found within 10 seconds
and never improves for the remaining 290; what the solver spends that time on
is trying to raise a lower bound that MTZ's weak relaxation keeps far below it.

## 9. Out of scope

Carried from `PROJECT_BRIEF.md` §4:

- Time windows (VRPTW)
- Multiple depots
- Pickup-and-delivery
- Heterogeneous fleet
- Any heuristic or metaheuristic — this repo is exact-MILP-only by design
- Lazy/branch-and-cut subtour elimination (needs solver callbacks)
