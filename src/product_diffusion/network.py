"""Friendship graph on which a product travels one hop."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SocialNetwork:
    """Undirected graph of reciprocal friendships.

    Hiring a person spreads the product to that person and to their friends,
    and no further. The people who hear it are the closed neighborhood of the
    hire.
    """

    names: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if len(set(self.names)) != len(self.names):
            raise ValueError("names must be unique")
        unknown = {person for edge in self.edges for person in edge} - set(self.names)
        if unknown:
            raise ValueError(f"edges mention unknown people: {sorted(unknown)}")
        for person, friend in self.edges:
            if person == friend:
                raise ValueError("friendships are between two distinct people")

    @property
    def n(self) -> int:
        return len(self.names)

    def neighbors(self) -> dict[str, frozenset[str]]:
        adjacent: dict[str, set[str]] = {name: set() for name in self.names}
        for person, friend in self.edges:
            adjacent[person].add(friend)
            adjacent[friend].add(person)
        return {name: frozenset(friends) for name, friends in adjacent.items()}

    def closed_neighborhoods(self) -> dict[str, tuple[str, ...]]:
        """Return N[v] = {v} ∪ N(v), ordered as in ``names``."""
        order = {name: index for index, name in enumerate(self.names)}
        adjacent = self.neighbors()
        return {
            name: tuple(sorted(adjacent[name] | {name}, key=order.__getitem__))
            for name in self.names
        }

    def closed_neighborhood_indices(self) -> tuple[tuple[int, ...], ...]:
        position = {name: index for index, name in enumerate(self.names)}
        return tuple(
            tuple(position[member] for member in neighborhood)
            for neighborhood in self.closed_neighborhoods().values()
        )

    def is_dominating(self, hired: set[str]) -> bool:
        """True when every person is hired or has a hired friend."""
        unknown = hired - set(self.names)
        if unknown:
            raise ValueError(f"unknown people: {sorted(unknown)}")
        neighborhoods = self.closed_neighborhoods()
        return all(hired & set(neighborhoods[name]) for name in self.names)

    def informant(self, hired: set[str]) -> dict[str, str | None]:
        """Who carries the product to each person under this hiring plan.

        A hired person is their own source. Anyone else is assigned the
        earliest hired member of their closed neighborhood, or ``None`` when
        the plan leaves them unreached.
        """
        neighborhoods = self.closed_neighborhoods()
        source: dict[str, str | None] = {}
        for name in self.names:
            if name in hired:
                source[name] = name
                continue
            carriers = [member for member in neighborhoods[name] if member in hired]
            source[name] = carriers[0] if carriers else None
        return source

    @classmethod
    def advertising(cls) -> SocialNetwork:
        """Six-person network of the product-diffusion case."""
        return cls(
            names=("Mario", "Sarah", "Raúl", "Ana", "Enrique", "Saúl"),
            edges=(
                ("Mario", "Sarah"),
                ("Mario", "Raúl"),
                ("Mario", "Ana"),
                ("Enrique", "Sarah"),
                ("Enrique", "Raúl"),
                ("Saúl", "Ana"),
            ),
        )

    @classmethod
    def cycle(cls, n: int) -> SocialNetwork:
        """Cycle graph on vertices ``0`` … ``n-1``, used as a control instance."""
        if n < 3:
            raise ValueError("a cycle needs at least three vertices")
        names = tuple(str(index) for index in range(n))
        edges = tuple((names[index], names[(index + 1) % n]) for index in range(n))
        return cls(names=names, edges=edges)
