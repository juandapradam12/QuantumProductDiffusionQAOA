"""QAOA for a diagonal cost operator, with a multi-start classical search.

The prepared state at depth ``p`` is

    |β, γ⟩ = prod_{ℓ=1}^{p} exp(-i β_ℓ H_M) exp(-i γ_ℓ H_C) |+⟩^{⊗ n},

    H_M = sum_q X_q.

``H_C`` is the non-identity part of the cost operator. Each restart draws
``β ∈ [0, π]`` and ``γ ∈ [0, 2π]`` and minimizes the exact statevector
expectation with COBYLA. The reported distribution is the Born distribution
of the best parameters, not a shot sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from product_diffusion.classical import bitstring
from product_diffusion.cost import Polynomial


@dataclass(frozen=True)
class DepthResult:
    depth: int
    parameter_names: tuple[str, ...]
    best_parameters: tuple[float, ...]
    best_energy: float
    final_energies: tuple[float, ...]
    best_trace: tuple[float, ...]
    probabilities: tuple[float, ...]
    gate_counts: dict[str, int]
    compiled_depth: int

    def probability_of(self, bits: tuple[int, ...]) -> float:
        key = 0
        for index, bit in enumerate(bits):
            key |= int(bit) << index
        return float(self.probabilities[key])


def run_depth(
    operator: SparsePauliOp,
    shift: float,
    cost: Polynomial,
    *,
    depth: int,
    restarts: int,
    seed: int,
    maxiter: int,
) -> DepthResult:
    """Minimize the depth-``depth`` QAOA energy and return its Born distribution."""
    if depth < 1:
        raise ValueError("depth must be at least 1")
    if restarts < 1:
        raise ValueError("at least one restart is required")

    circuit = _compiled_qaoa(operator, depth)
    parameter_names = tuple(parameter.name for parameter in circuit.parameters)
    gate_counts = {name: int(circuit.count_ops().get(name, 0)) for name in ("h", "rx", "rz", "cx")}

    def energy(theta: np.ndarray) -> float:
        return _expectation(circuit, operator, theta) + shift

    rng = np.random.default_rng(seed + 1000 * depth)
    best_energy = float("inf")
    best_theta = np.zeros(circuit.num_parameters)
    best_trace: list[float] = []
    finals: list[float] = []

    for _ in range(restarts):
        trace: list[float] = []

        def objective(theta: np.ndarray, trace: list[float] = trace) -> float:
            value = float(energy(theta))
            trace.append(value)
            return value

        start = _initial_point(parameter_names, rng)
        result = minimize(
            objective,
            start,
            method="COBYLA",
            options={"maxiter": maxiter, "rhobeg": 0.5},
        )
        finals.append(float(result.fun))
        if result.fun < best_energy:
            best_energy = float(result.fun)
            best_theta = np.asarray(result.x, dtype=float).copy()
            best_trace = trace

    probabilities = _probabilities(circuit, best_theta)
    # Recompute the energy from the state so the stored value matches the distribution.
    best_energy = float(np.dot(probabilities, _classical_costs(cost)))
    return DepthResult(
        depth=depth,
        parameter_names=parameter_names,
        best_parameters=tuple(float(value) for value in best_theta),
        best_energy=best_energy,
        final_energies=tuple(finals),
        best_trace=tuple(best_trace),
        probabilities=tuple(float(value) for value in probabilities),
        gate_counts=gate_counts,
        compiled_depth=int(circuit.depth()),
    )


def _compiled_qaoa(operator: SparsePauliOp, depth: int) -> QuantumCircuit:
    ansatz = QAOAAnsatz(cost_operator=operator, reps=depth)
    return transpile(ansatz, basis_gates=["h", "rx", "rz", "cx"], optimization_level=1)


def _initial_point(names: tuple[str, ...], rng: np.random.Generator) -> np.ndarray:
    point = np.empty(len(names), dtype=float)
    for index, name in enumerate(names):
        if "β" in name or "beta" in name.lower():
            point[index] = rng.uniform(0.0, np.pi)
        else:
            point[index] = rng.uniform(0.0, 2.0 * np.pi)
    return point


def _bind(circuit: QuantumCircuit, theta: np.ndarray) -> QuantumCircuit:
    assignment = {
        parameter: float(value)
        for parameter, value in zip(circuit.parameters, theta, strict=True)
    }
    return circuit.assign_parameters(assignment)


def _expectation(circuit: QuantumCircuit, operator: SparsePauliOp, theta: np.ndarray) -> float:
    state = Statevector.from_instruction(_bind(circuit, theta))
    return float(np.real(state.expectation_value(operator)))


def _probabilities(circuit: QuantumCircuit, theta: np.ndarray) -> np.ndarray:
    state = Statevector.from_instruction(_bind(circuit, theta))
    return np.real(state.probabilities())


def _classical_costs(cost: Polynomial) -> np.ndarray:
    values = np.empty(1 << cost.n, dtype=float)
    for key in range(1 << cost.n):
        bits = [(key >> index) & 1 for index in range(cost.n)]
        values[key] = cost.evaluate(bits)
    return values


def basis_label(key: int, n: int) -> str:
    bits = tuple((key >> index) & 1 for index in range(n))
    return bitstring(bits)
