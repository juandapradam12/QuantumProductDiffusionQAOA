"""Assemble the advertising-network case and the records behind the figures."""

from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp, Statevector

from product_diffusion.classical import enumerate_plans, greedy_dominating_set, ground_states, bitstring
from product_diffusion.cost import Polynomial, at_least_one_cost, exactly_one_cost
from product_diffusion.hamiltonian import cost_operator, pauli_terms
from product_diffusion.network import SocialNetwork
from product_diffusion.qaoa import run_depth


def run_case_study(
    *,
    seed: int = 7,
    restarts: int = 40,
    maxiter: int = 250,
    depths: tuple[int, ...] = (1, 2, 3),
    weight: float = 1.0,
    penalty: float = 2.0,
) -> dict:
    network = SocialNetwork.advertising()
    cost = at_least_one_cost(network, weight=weight, penalty=penalty)
    plans = _plans(network, cost)
    optimal_cost = min(plan["cost"] for plan in plans)
    for plan in plans:
        plan["optimal"] = abs(plan["cost"] - optimal_cost) <= 1e-8

    greedy = greedy_dominating_set(network)
    greedy_bits = tuple(int(name in greedy) for name in network.names)
    greedy_cost = float(cost.evaluate(greedy_bits))

    operator, shift = cost_operator(cost)
    diagonal_error = _diagonal_error(cost, operator, shift)
    squared_ground = _ground_records(network, exactly_one_cost(network, weight=weight, penalty=penalty))

    print(
        f"Ground state { _hired_label(_optimal_names(plans)) } "
        f"at cost {optimal_cost:g}; diagonal error {diagonal_error:.2e}",
        flush=True,
    )

    depth_records = []
    for depth in depths:
        print(f"QAOA depth {depth}: {restarts} restarts", flush=True)
        result = run_depth(
            operator,
            shift,
            cost,
            depth=depth,
            restarts=restarts,
            seed=seed,
            maxiter=maxiter,
        )
        optimal_probability = sum(
            result.probabilities[index]
            for index, plan in enumerate(plans)
            if plan["optimal"]
        )
        dominating_probability = sum(
            result.probabilities[index]
            for index, plan in enumerate(plans)
            if plan["dominating"]
        )
        print(
            f"  energy {result.best_energy:.4f}  "
            f"P(optimum) {optimal_probability:.4f}  "
            f"P(dominating) {dominating_probability:.4f}",
            flush=True,
        )
        depth_records.append(
            {
                "depth": result.depth,
                "parameter_names": list(result.parameter_names),
                "best_parameters": [round(value, 10) for value in result.best_parameters],
                "best_energy": result.best_energy,
                "final_energies": list(result.final_energies),
                "best_trace": list(result.best_trace),
                "probabilities": list(result.probabilities),
                "gate_counts": result.gate_counts,
                "compiled_depth": result.compiled_depth,
                "optimal_probability": optimal_probability,
                "dominating_probability": dominating_probability,
            }
        )

    optimum = next(plan for plan in plans if plan["optimal"])
    channels = network.informant(set(optimum["hired"]))

    return {
        "weight": weight,
        "penalty": penalty,
        "seed": seed,
        "restarts": restarts,
        "maxiter": maxiter,
        "names": list(network.names),
        "edges": [list(edge) for edge in network.edges],
        "closed_neighborhoods": {
            name: list(members) for name, members in network.closed_neighborhoods().items()
        },
        "plans": plans,
        "optimal_cost": optimal_cost,
        "optimal_hired": list(optimum["hired"]),
        "channels": channels,
        "greedy_hired": list(greedy),
        "greedy_cost": greedy_cost,
        "identity_shift": round(shift, 12),
        "diagonal_error": diagonal_error,
        "pauli_terms": pauli_terms(operator),
        "squared_ground": squared_ground,
        "squared_spectrum": _squared_spectrum(network, weight, penalty),
        "penalty_sweep": _penalty_sweep(network, weight),
        "cycle": _cycle_contrast(penalty=penalty, weight=weight),
        "depths": depth_records,
    }


def _plans(network: SocialNetwork, cost: Polynomial) -> list[dict]:
    records = []
    for plan in enumerate_plans(cost):
        hired = list(plan.hired(network.names))
        records.append(
            {
                "bits": list(plan.bits),
                "bitstring": bitstring(plan.bits),
                "hired": hired,
                "cost": round(plan.cost, 10),
                "dominating": network.is_dominating(set(hired)),
            }
        )
    return records


def _ground_records(network: SocialNetwork, cost: Polynomial) -> list[dict]:
    records = []
    for plan in ground_states(cost):
        hired = list(plan.hired(network.names))
        records.append(
            {
                "hired": hired,
                "bitstring": bitstring(plan.bits),
                "cost": round(plan.cost, 10),
                "dominating": network.is_dominating(set(hired)),
            }
        )
    return records


def _penalty_sweep(network: SocialNetwork, weight: float) -> list[dict]:
    """Best dominating cost and best non-dominating cost across penalty values."""
    rows = []
    penalties = [round(step * 0.05, 2) for step in range(0, 61)]
    for penalty in penalties:
        cost = at_least_one_cost(network, weight=weight, penalty=penalty)
        plans = enumerate_plans(cost)
        dominating = []
        infeasible = []
        for plan in plans:
            hired = plan.hired(network.names)
            record = {"hired": list(hired), "cost": round(plan.cost, 10)}
            if network.is_dominating(set(hired)):
                dominating.append(record)
            else:
                infeasible.append(record)
        best_dominating = min(dominating, key=lambda record: record["cost"])
        best_infeasible = min(infeasible, key=lambda record: record["cost"])
        ground = ground_states(cost)
        ground_hired = [list(plan.hired(network.names)) for plan in ground]
        rows.append(
            {
                "penalty": penalty,
                "best_dominating_cost": best_dominating["cost"],
                "best_dominating_hired": best_dominating["hired"],
                "best_infeasible_cost": best_infeasible["cost"],
                "best_infeasible_hired": best_infeasible["hired"],
                "ground_hired": ground_hired,
                "ground_is_minimum_dominating_set": all(
                    network.is_dominating(set(hired)) and len(hired) == len(best_dominating["hired"])
                    for hired in ground_hired
                ),
            }
        )
    return rows


def _squared_spectrum(network: SocialNetwork, weight: float, penalty: float) -> dict:
    plans = enumerate_plans(exactly_one_cost(network, weight=weight, penalty=penalty))
    costs = sorted({round(plan.cost, 10) for plan in plans})
    dominating_costs = sorted(
        {
            round(plan.cost, 10)
            for plan in plans
            if network.is_dominating(set(plan.hired(network.names)))
        }
    )
    return {
        "ground_cost": dominating_costs[0],
        "next_cost": costs[1],
        "next_dominating_cost": dominating_costs[1],
    }


def _cycle_contrast(penalty: float, weight: float) -> dict:
    cycle = SocialNetwork.cycle(4)
    return {
        "names": list(cycle.names),
        "edges": [list(edge) for edge in cycle.edges],
        "squared_ground": _ground_records(cycle, exactly_one_cost(cycle, weight=weight, penalty=penalty)),
        "product_ground": _ground_records(cycle, at_least_one_cost(cycle, weight=weight, penalty=penalty)),
    }


def _diagonal_error(cost: Polynomial, operator: SparsePauliOp, shift: float) -> float:
    worst = 0.0
    identity = SparsePauliOp.from_list([("I" * cost.n, shift)])
    full = (operator + identity).simplify()
    for key in range(1 << cost.n):
        bits = [(key >> index) & 1 for index in range(cost.n)]
        expected = cost.evaluate(bits)
        state = Statevector.from_int(key, 2**cost.n)
        observed = float(state.expectation_value(full).real)
        worst = max(worst, abs(observed - expected))
    return float(worst)


def _optimal_names(plans: list[dict]) -> list[str]:
    return next(plan["hired"] for plan in plans if plan["optimal"])


def _hired_label(names: list[str]) -> str:
    return ", ".join(names) if names else "nobody"
