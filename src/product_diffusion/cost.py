"""Pseudo-Boolean cost of a hiring plan.

The decision variable ``x_v`` is 1 when person ``v`` is hired. With a positive
weight ``A`` on each hire and a penalty ``B`` on each person left unreached,

    C(x) = A * sum_v x_v  +  B * sum_v prod_{u in N[v]} (1 - x_u).

The product is the indicator that nobody in the closed neighborhood was hired,
so a plan that reaches everyone pays only for the hires. The same polynomial
is the diagonal of the cost operator used by QAOA.

``exactly_one_cost`` is the squared penalty ``(1 - sum_{u in N[v]} x_u)^2``.
It vanishes only when each person is reached by exactly one hire. That is a
different constraint, kept here so the case can show where the two disagree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

from product_diffusion.network import SocialNetwork


@dataclass(frozen=True)
class Polynomial:
    """Multilinear polynomial in binary variables, keyed by sorted index tuples."""

    n: int
    terms: tuple[tuple[tuple[int, ...], float], ...]

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("a cost needs at least one variable")
        for monomial, _coefficient in self.terms:
            if monomial != tuple(sorted(set(monomial))):
                raise ValueError(f"monomial {monomial} is not a sorted set of indices")
            if monomial and (monomial[0] < 0 or monomial[-1] >= self.n):
                raise ValueError(f"monomial {monomial} is outside 0..{self.n - 1}")

    def evaluate(self, bits: Sequence[int]) -> float:
        if len(bits) != self.n:
            raise ValueError(f"expected {self.n} bits, got {len(bits)}")
        total = 0.0
        for monomial, coefficient in self.terms:
            if all(bits[index] for index in monomial):
                total += coefficient
        return float(total)

    def as_dict(self) -> dict[tuple[int, ...], float]:
        return dict(self.terms)


def _freeze(n: int, accumulator: Mapping[tuple[int, ...], float]) -> Polynomial:
    terms = tuple(
        (monomial, float(coefficient))
        for monomial, coefficient in sorted(accumulator.items(), key=lambda item: (len(item[0]), item[0]))
        if abs(coefficient) > 1e-12
    )
    return Polynomial(n=n, terms=terms)


def _add(accumulator: dict[tuple[int, ...], float], monomial: tuple[int, ...], value: float) -> None:
    accumulator[monomial] = accumulator.get(monomial, 0.0) + value


def at_least_one_cost(network: SocialNetwork, weight: float = 1.0, penalty: float = 2.0) -> Polynomial:
    """Cost that is ``weight * |hired|`` on dominating sets, and higher otherwise.

    ``penalty`` is charged once for each vertex whose closed neighborhood
    contains no hire. Any ``penalty > weight / 2`` is large enough on the
    advertising network; the case uses ``penalty = 2``.
    """
    if weight <= 0:
        raise ValueError("weight must be positive")
    if penalty < 0:
        raise ValueError("penalty must be non-negative")

    accumulator: dict[tuple[int, ...], float] = {}
    for neighborhood in network.closed_neighborhood_indices():
        for degree in range(len(neighborhood) + 1):
            sign = -1.0 if degree % 2 else 1.0
            for subset in combinations(neighborhood, degree):
                _add(accumulator, subset, penalty * sign)
    for index in range(network.n):
        _add(accumulator, (index,), weight)
    return _freeze(network.n, accumulator)


def exactly_one_cost(network: SocialNetwork, weight: float = 1.0, penalty: float = 2.0) -> Polynomial:
    """Squared penalty, zero only when every vertex is reached exactly once.

    Binary variables satisfy ``x_u^2 = x_u``, so

        (1 - sum_u x_u)^2 = 1 - sum_u x_u + 2 * sum_{u < w} x_u x_w.
    """
    if weight <= 0:
        raise ValueError("weight must be positive")
    if penalty < 0:
        raise ValueError("penalty must be non-negative")

    accumulator: dict[tuple[int, ...], float] = {}
    for neighborhood in network.closed_neighborhood_indices():
        _add(accumulator, (), penalty)
        for index in neighborhood:
            _add(accumulator, (index,), -penalty)
        for left, right in combinations(neighborhood, 2):
            _add(accumulator, (left, right), 2.0 * penalty)
    for index in range(network.n):
        _add(accumulator, (index,), weight)
    return _freeze(network.n, accumulator)
