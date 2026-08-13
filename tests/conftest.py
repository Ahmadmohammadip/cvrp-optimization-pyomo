"""Shared test setup and hand-checkable instances.

The Agg backend is selected here, before anything imports pyplot, so the
plotting tests run headless in CI the same way they do locally.

The `cross` instance is deliberately tiny and symmetric so its optimum can be
worked out on paper rather than taken on trust from the solver — the
arithmetic is in `CrossInstance` below.

Everything shared lives on fixtures rather than module constants, so test
files never have to import from conftest (which pytest's default import mode
does not put on the path).
"""

import math
from dataclasses import dataclass, field

import matplotlib
import pytest

matplotlib.use("Agg")


def euclidean_distance(coords: dict) -> dict:
    """Exact Euclidean arc costs for every ordered pair of distinct nodes."""
    return {
        (i, j): math.dist(coords[i], coords[j])
        for i in coords
        for j in coords
        if i != j
    }


@dataclass
class CrossInstance:
    """Depot at the origin, two customers out along each axis.

    Capacity 10 against demand 5 each means at most two customers per vehicle,
    so k_min = 20/10 = 2 and every vehicle is full. The three ways to split
    four customers into two pairs cost:

        {1,2} + {3,4}  ->  (1+1+2) + (1+1+2)           =  8.000  <- optimum
        {1,3} + {2,4}  ->  (1+sqrt2+1) + (2+2sqrt2+2)  = 10.243
        {1,4} + {2,3}  ->  (1+sqrt5+2) + (2+sqrt5+1)   = 10.472
    """

    depot: int = 0
    capacity: float = 10.0
    k_min: int = 2
    optimal_distance: float = 8.0
    coords: dict = field(
        default_factory=lambda: {
            0: (0.0, 0.0),
            1: (1.0, 0.0),
            2: (2.0, 0.0),
            3: (0.0, 1.0),
            4: (0.0, 2.0),
        }
    )
    demand: dict = field(
        default_factory=lambda: {0: 0.0, 1: 5.0, 2: 5.0, 3: 5.0, 4: 5.0}
    )

    @property
    def customers(self) -> list[int]:
        return [i for i in self.coords if i != self.depot]

    @property
    def distance(self) -> dict:
        return euclidean_distance(self.coords)


@pytest.fixture
def cross():
    return CrossInstance()
