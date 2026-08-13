"""
Builds a Pyomo ConcreteModel for the Capacitated Vehicle Routing Problem.

Two-index vehicle-flow formulation with Miller-Tucker-Zemlin (MTZ) subtour
elimination. This is a standard textbook formulation — see Toth & Vigo,
*The Vehicle Routing Problem* (SIAM, 2002) — not something devised here.

    min   sum_(i,j) c_ij x_ij
    s.t.  sum_j x_ij = 1                        for each customer i   (out-degree)
          sum_i x_ij = 1                        for each customer j   (in-degree)
          sum_j x_0j <= K                                             (fleet size)
          sum_j x_0j  = sum_i x_i0                                    (depot balance)
          u_i - u_j + Q x_ij <= Q - d_j         for customers i != j  (MTZ)
          d_i <= u_i <= Q                       for each customer i

`x_ij` is 1 when a vehicle traverses arc (i, j); `u_i` is the cumulative load
delivered on the route up to and including customer i.

The MTZ family does double duty. Read it two ways:

* When `x_ij = 1` it collapses to `u_j >= u_i + d_j` — load accumulates along
  a route, and since `u_i <= Q`, no route can exceed capacity.
* Around any cycle that avoids the depot, those inequalities sum to
  `0 >= sum of demands`, which is false for positive demand. So customer-only
  cycles — subtours — cannot exist.

That second reading is also why there is no explicit "at least one vehicle
must leave the depot" constraint: every customer needs an incoming arc, and
the only way to supply one without forming a depot-free cycle is to route back
to the depot. Capacity then forces the vehicle count up to `k_min` on its own.

## Fleet size: `<= K`, not `= K`

PROJECT_BRIEF.md section 1.5 writes the depot degree as `= K`. This model uses
`<= K` plus an explicit balance constraint, because the brief's own section 1.6
calls running more vehicles than the minimum "a valid, common scenario" — and
under equality, every surplus vehicle is forced into service, so adding one can
only make the tour longer. With `<= K` the solver parks what it does not need.

Equality gave balance for free (K out, K back). Inequality does not, hence
`sum_j x_0j = sum_i x_i0`: departures must equal returns, or arcs would dangle.

The trade-off is real: freeing the depot count weakens the LP relaxation, so
this solves more slowly than the equality form. See docs/formulation.md.
"""

from pyomo.environ import (
    Any,
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Param,
    Set,
    Var,
    minimize,
)


def build_cvrp_model(
    distance: dict[tuple, float],
    demand: dict,
    capacity: float,
    num_vehicles: int,
    depot=0,
) -> ConcreteModel:
    """Build the CVRP MILP from primitive inputs.

    `distance` maps every arc (i, j), i != j, to its cost. `demand` maps every
    node to its demand, including the depot (which must be 0). Node ids may be
    any hashable — ints or strings both work.

    Callers holding a validated `System` should use `build_from_system`, which
    unpacks it and calls this.
    """
    nodes = sorted(demand.keys(), key=lambda k: (k != depot, k))
    if depot not in demand:
        raise ValueError(f"depot {depot!r} is not present in demand")
    if demand[depot] != 0:
        raise ValueError(f"depot {depot!r} must have demand 0, got {demand[depot]}")

    customers = [i for i in nodes if i != depot]
    if not customers:
        raise ValueError("model needs at least one customer")

    arcs = [(i, j) for i in nodes for j in nodes if i != j]
    missing = [a for a in arcs if a not in distance]
    if missing:
        raise ValueError(
            f"distance is missing {len(missing)} arc(s), e.g. {missing[:3]} — "
            f"every ordered pair of distinct nodes needs a cost"
        )

    m = ConcreteModel(name="CVRP_MTZ")

    # --- Sets ---
    m.V = Set(initialize=nodes, ordered=True)
    m.V_C = Set(initialize=customers, ordered=True)
    m.A = Set(initialize=arcs, dimen=2)
    # MTZ applies between customers only: the depot has no u variable, which is
    # exactly what lets routes through it while forbidding cycles that avoid it.
    m.A_C = Set(initialize=[(i, j) for i in customers for j in customers if i != j], dimen=2)

    # --- Parameters ---
    m.cost = Param(m.A, initialize={a: distance[a] for a in arcs}, within=NonNegativeReals)
    m.demand = Param(m.V_C, initialize={i: demand[i] for i in customers})
    m.capacity = Param(initialize=capacity)
    m.num_vehicles = Param(initialize=num_vehicles)
    # Carried on the model so route reconstruction can find the depot without
    # the caller having to pass it a second time. `within=Any` because node ids
    # are not required to be numeric.
    m.depot = Param(initialize=depot, within=Any)

    # --- Variables ---
    m.x = Var(m.A, domain=Binary)

    def _u_bounds(m, i):
        return (demand[i], capacity)

    m.u = Var(m.V_C, domain=NonNegativeReals, bounds=_u_bounds)

    # --- Objective ---
    def _total_distance_rule(m):
        return sum(m.cost[i, j] * m.x[i, j] for (i, j) in m.A)

    m.total_distance = Objective(rule=_total_distance_rule, sense=minimize)

    # --- Constraints ---

    def _out_degree_rule(m, i):
        return sum(m.x[i, j] for j in m.V if j != i) == 1

    m.out_degree_con = Constraint(m.V_C, rule=_out_degree_rule)

    def _in_degree_rule(m, j):
        return sum(m.x[i, j] for i in m.V if i != j) == 1

    m.in_degree_con = Constraint(m.V_C, rule=_in_degree_rule)

    def _fleet_size_rule(m):
        return sum(m.x[depot, j] for j in m.V_C) <= m.num_vehicles

    m.fleet_size_con = Constraint(rule=_fleet_size_rule)

    def _depot_balance_rule(m):
        return sum(m.x[depot, j] for j in m.V_C) == sum(m.x[i, depot] for i in m.V_C)

    m.depot_balance_con = Constraint(rule=_depot_balance_rule)

    def _mtz_rule(m, i, j):
        return m.u[i] - m.u[j] + m.capacity * m.x[i, j] <= m.capacity - m.demand[j]

    m.mtz_con = Constraint(m.A_C, rule=_mtz_rule)

    return m
