"""Exact ground state, and the greedy heuristic that misses it."""

from __future__ import annotations

from dataclasses import dataclass

from product_diffusion.cost import Polynomial
from product_diffusion.network import SocialNetwork


@dataclass(frozen=True)
class Plan:
    bits: tuple[int, ...]
    cost: float

    def hired(self, names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(name for name, bit in zip(names, self.bits, strict=True) if bit)

    @property
    def size(self) -> int:
        return sum(self.bits)


def enumerate_plans(cost: Polynomial) -> list[Plan]:
    """Score every hiring plan. Intended for the small instances in this case."""
    if cost.n > 16:
        raise ValueError("exhaustive enumeration is limited to 16 people")
    plans: list[Plan] = []
    for key in range(1 << cost.n):
        bits = tuple((key >> index) & 1 for index in range(cost.n))
        plans.append(Plan(bits=bits, cost=float(cost.evaluate(bits))))
    return plans


def ground_states(cost: Polynomial, tolerance: float = 1e-8) -> list[Plan]:
    plans = enumerate_plans(cost)
    best = min(plan.cost for plan in plans)
    return [plan for plan in plans if plan.cost <= best + tolerance]


def greedy_dominating_set(network: SocialNetwork) -> tuple[str, ...]:
    """Hire, repeatedly, whoever still reaches the most unreached people.

    Ties break toward the earliest name in ``network.names``. On the
    advertising network the first choice is Mario, and no later choice can
    finish the cover without a third hire.
    """
    neighborhoods = network.closed_neighborhoods()
    uncovered = set(network.names)
    hired: list[str] = []
    while uncovered:
        def newly_reached(name: str) -> tuple[int, int]:
            reached = len(set(neighborhoods[name]) & uncovered)
            return (reached, -network.names.index(name))

        choice = max(network.names, key=newly_reached)
        if newly_reached(choice)[0] == 0:
            raise RuntimeError("remaining people are not reachable")
        hired.append(choice)
        uncovered -= set(neighborhoods[choice])
    return tuple(hired)


def bitstring(bits: tuple[int, ...] | list[int]) -> str:
    """Little-endian bitstring: qubit 0, the first person, is the rightmost bit."""
    return "".join(str(bit) for bit in reversed(bits))
