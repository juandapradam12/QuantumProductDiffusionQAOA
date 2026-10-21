import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from product_diffusion.cost import at_least_one_cost, exactly_one_cost
from product_diffusion.hamiltonian import cost_operator
from product_diffusion.network import SocialNetwork


def _reference_product(bits, neighborhoods, weight, penalty):
    cost = weight * sum(bits)
    for neighborhood in neighborhoods:
        if all(bits[index] == 0 for index in neighborhood):
            cost += penalty
    return cost


def _reference_squared(bits, neighborhoods, weight, penalty):
    cost = weight * sum(bits)
    for neighborhood in neighborhoods:
        covered = sum(bits[index] for index in neighborhood)
        cost += penalty * (1 - covered) ** 2
    return cost


def test_polynomials_match_the_direct_formulas():
    network = SocialNetwork.advertising()
    neighborhoods = network.closed_neighborhood_indices()
    product = at_least_one_cost(network, weight=1.0, penalty=2.0)
    squared = exactly_one_cost(network, weight=1.0, penalty=2.0)
    for key in range(2**network.n):
        bits = [(key >> index) & 1 for index in range(network.n)]
        assert product.evaluate(bits) == _reference_product(bits, neighborhoods, 1.0, 2.0)
        assert squared.evaluate(bits) == _reference_squared(bits, neighborhoods, 1.0, 2.0)


def test_operator_diagonal_matches_every_hiring_plan():
    network = SocialNetwork.advertising()
    cost = at_least_one_cost(network)
    operator, shift = cost_operator(cost)
    full = (operator + SparsePauliOp.from_list([("I" * network.n, shift)])).simplify()
    worst = 0.0
    for key in range(2**network.n):
        bits = [(key >> index) & 1 for index in range(network.n)]
        state = Statevector.from_int(key, 2**network.n)
        observed = float(np.real(state.expectation_value(full)))
        worst = max(worst, abs(observed - cost.evaluate(bits)))
    assert worst < 1e-8
