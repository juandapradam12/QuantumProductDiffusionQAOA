"""Map a multilinear binary cost to a diagonal Pauli operator.

On computational-basis states the substitution

    x_q = (I - Z_q) / 2

makes the operator expectation equal to the classical cost. Products of
variables become products of ``(I - Z)/2`` and therefore Pauli strings whose
weight is at most the size of the largest closed neighborhood. No slack bits
are introduced: qubit ``q`` remains person ``q``.
"""

from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp

from product_diffusion.cost import Polynomial


def cost_operator(cost: Polynomial) -> tuple[SparsePauliOp, float]:
    """Return the non-identity part of the cost operator and the identity shift.

    The shift is the coefficient of ``I``. Adding it back to an expectation
    recovers the classical cost; it does not change the QAOA unitary beyond a
    global phase, so the circuit is built from the non-identity part alone.
    """
    accumulated: dict[str, float] = {}
    for monomial, coefficient in cost.terms:
        _accumulate_monomial(monomial, coefficient, cost.n, accumulated)

    identity = "I" * cost.n
    shift = float(accumulated.pop(identity, 0.0))
    kept = [
        (label, coeff)
        for label, coeff in accumulated.items()
        if abs(coeff) > 1e-12
    ]
    if not kept:
        raise ValueError("cost operator has no non-trivial Pauli terms")
    operator = SparsePauliOp.from_list(kept).simplify()
    return operator, shift


def pauli_terms(operator: SparsePauliOp) -> list[dict[str, float | int | str]]:
    """Non-identity Pauli terms as plain records, sorted by weight then label."""
    records = []
    for pauli, coefficient in zip(operator.paulis, operator.coeffs):
        label = pauli.to_label()
        weight = label.count("Z") + label.count("X") + label.count("Y")
        records.append(
            {
                "label": label,
                "weight": int(weight),
                "coefficient": round(float(coefficient.real), 12),
            }
        )
    records.sort(key=lambda record: (int(record["weight"]), str(record["label"])))
    return records


def _accumulate_monomial(
    monomial: tuple[int, ...],
    coefficient: float,
    n: int,
    accumulated: dict[str, float],
) -> None:
    """Add the Pauli expansion of ``coefficient * prod_{q in monomial} x_q``."""
    if not monomial:
        label = "I" * n
        accumulated[label] = accumulated.get(label, 0.0) + coefficient
        return

    scale = coefficient / (2 ** len(monomial))
    for mask in range(1 << len(monomial)):
        letters = ["I"] * n
        sign = 1.0
        for bit, qubit in enumerate(monomial):
            if (mask >> bit) & 1:
                letters[qubit] = "Z"
                sign = -sign
        # Qiskit labels are little-endian: qubit 0 is the rightmost character.
        label = "".join(reversed(letters))
        accumulated[label] = accumulated.get(label, 0.0) + scale * sign
