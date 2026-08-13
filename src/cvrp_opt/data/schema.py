"""
Typed, validated data structures for the CVRP.

Design intent, same as the sibling repos: an instance fails loudly at
construction — a customer whose demand no vehicle could ever carry, a fleet
too small to cover total demand, duplicate ids — rather than surfacing later
as an opaque solver infeasibility. For a MILP that can run for minutes, the
difference matters more than usual: waiting five minutes to learn the instance
was malformed is a bad way to find out.
"""

import math
from dataclasses import dataclass, field

# Guards ceil() against float noise, so 20.0 / 10.0 gives 2 rather than 3.
_CEIL_EPSILON = 1e-9


@dataclass(frozen=True)
class Depot:
    """The single depot every route starts and ends at.

    Demand is not a field: the depot has none by definition, and making it
    settable would only create a way to express something meaningless.
    """

    id: int | str = 0
    x: float = 0.0
    y: float = 0.0


@dataclass(frozen=True)
class Customer:
    id: int | str
    x: float
    y: float
    demand: float

    def __post_init__(self):
        if self.demand < 0:
            raise ValueError(f"customer {self.id!r}: demand must be >= 0, got {self.demand}")


@dataclass(frozen=True)
class Fleet:
    """A homogeneous fleet: every vehicle shares one capacity.

    `num_vehicles` (K) may be left as None, in which case the System resolves
    it to the theoretical minimum k_min. It is never silently computed and
    hidden — `System.num_vehicles` always reports the value in force.
    """

    capacity: float
    num_vehicles: int | None = None

    def __post_init__(self):
        if self.capacity <= 0:
            raise ValueError(f"fleet capacity must be > 0, got {self.capacity}")
        if self.num_vehicles is not None and self.num_vehicles < 1:
            raise ValueError(f"fleet num_vehicles must be >= 1, got {self.num_vehicles}")


@dataclass(frozen=True)
class System:
    """A complete CVRP instance: one depot, some customers, one fleet."""

    depot: Depot
    customers: list[Customer] = field(default_factory=list)
    fleet: Fleet = field(default_factory=lambda: Fleet(capacity=1.0))

    def __post_init__(self):
        if not self.customers:
            raise ValueError("System must contain at least one customer")

        ids = [c.id for c in self.customers]
        if len(ids) != len(set(ids)):
            duplicates = sorted({i for i in ids if ids.count(i) > 1}, key=str)
            raise ValueError(f"customer ids must be unique, repeated: {duplicates}")
        if self.depot.id in ids:
            raise ValueError(
                f"customer id {self.depot.id!r} collides with the depot id — "
                f"node ids must be distinct across the whole instance"
            )

        # A single demand above capacity is unservable by any vehicle. Caught
        # first because it explains the failure far better than the aggregate
        # check below would.
        oversized = [c for c in self.customers if c.demand > self.fleet.capacity]
        if oversized:
            worst = max(oversized, key=lambda c: c.demand)
            raise ValueError(
                f"customer {worst.id!r} has demand {worst.demand} above vehicle "
                f"capacity {self.fleet.capacity} — no single vehicle could serve it, "
                f"and this model does not split deliveries"
            )

        if self.fleet.num_vehicles is not None and self.fleet.num_vehicles < self.k_min:
            raise ValueError(
                f"fleet of {self.fleet.num_vehicles} vehicle(s) cannot carry total "
                f"demand {self.total_demand} at capacity {self.fleet.capacity} — "
                f"at least {self.k_min} needed"
            )

    @property
    def n_customers(self) -> int:
        return len(self.customers)

    @property
    def total_demand(self) -> float:
        return sum(c.demand for c in self.customers)

    @property
    def k_min(self) -> int:
        """Fewest vehicles that could cover total demand, ignoring geography.

        A lower bound only: it assumes perfect packing, so a real instance may
        need more once routing is taken into account.
        """
        return max(1, math.ceil(self.total_demand / self.fleet.capacity - _CEIL_EPSILON))

    @property
    def num_vehicles(self) -> int:
        """K in force — the configured fleet size, or k_min when unset."""
        return self.fleet.num_vehicles if self.fleet.num_vehicles is not None else self.k_min

    @property
    def coords(self) -> dict:
        return {self.depot.id: (self.depot.x, self.depot.y)} | {
            c.id: (c.x, c.y) for c in self.customers
        }

    @property
    def demands(self) -> dict:
        """Demand by node id, including the depot at 0."""
        return {self.depot.id: 0.0} | {c.id: c.demand for c in self.customers}

    def distance_matrix(self) -> dict:
        """Exact Euclidean cost for every ordered pair of distinct nodes.

        Kept as floats rather than rounded to integers. TSPLIB's EUC_2D
        convention rounds, and benchmark objective values from that literature
        are not directly comparable to these — noted in docs/formulation.md.
        """
        coords = self.coords
        return {
            (i, j): math.dist(coords[i], coords[j])
            for i in coords
            for j in coords
            if i != j
        }
