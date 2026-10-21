from product_diffusion.classical import bitstring, greedy_dominating_set, ground_states
from product_diffusion.cost import at_least_one_cost, exactly_one_cost
from product_diffusion.network import SocialNetwork


def test_closed_neighborhoods_match_the_friendships():
    network = SocialNetwork.advertising()
    assert network.closed_neighborhoods() == {
        "Mario": ("Mario", "Sarah", "Raúl", "Ana"),
        "Sarah": ("Mario", "Sarah", "Enrique"),
        "Raúl": ("Mario", "Raúl", "Enrique"),
        "Ana": ("Mario", "Ana", "Saúl"),
        "Enrique": ("Sarah", "Raúl", "Enrique"),
        "Saúl": ("Ana", "Saúl"),
    }


def test_minimum_plan_hires_ana_and_enrique():
    network = SocialNetwork.advertising()
    cost = at_least_one_cost(network, weight=1.0, penalty=2.0)
    ground = ground_states(cost)
    assert len(ground) == 1
    assert ground[0].hired(network.names) == ("Ana", "Enrique")
    assert ground[0].cost == 2.0
    assert network.is_dominating({"Ana", "Enrique"})
    assert bitstring(ground[0].bits) == "011000"


def test_empty_plan_pays_the_penalty_six_times():
    network = SocialNetwork.advertising()
    cost = at_least_one_cost(network, weight=1.0, penalty=2.0)
    assert cost.evaluate((0, 0, 0, 0, 0, 0)) == 12.0


def test_half_penalty_ties_mario_with_the_minimum_plan():
    network = SocialNetwork.advertising()
    ground = ground_states(at_least_one_cost(network, penalty=0.5))
    hired = {plan.hired(network.names) for plan in ground}
    assert ("Ana", "Enrique") in hired
    assert ("Mario",) in hired
    assert any(not network.is_dominating(set(plan.hired(network.names))) for plan in ground)


def test_penalty_below_one_half_accepts_an_incomplete_plan():
    network = SocialNetwork.advertising()
    low = ground_states(at_least_one_cost(network, penalty=0.4))
    assert any(not network.is_dominating(set(plan.hired(network.names))) for plan in low)
    high = ground_states(at_least_one_cost(network, penalty=2.0))
    assert all(network.is_dominating(set(plan.hired(network.names))) for plan in high)


def test_greedy_opens_with_mario_and_hires_three_people():
    network = SocialNetwork.advertising()
    hired = greedy_dominating_set(network)
    assert hired[0] == "Mario"
    assert len(hired) == 3
    assert network.is_dominating(set(hired))
    assert hired == ("Mario", "Sarah", "Ana")


def test_squared_penalty_on_the_cycle_does_not_dominate():
    cycle = SocialNetwork.cycle(4)
    squared = ground_states(exactly_one_cost(cycle, penalty=2.0))
    product = ground_states(at_least_one_cost(cycle, penalty=2.0))
    assert squared
    assert all(not cycle.is_dominating(set(plan.hired(cycle.names))) for plan in squared)
    assert all(len(plan.hired(cycle.names)) == 1 for plan in squared)
    assert all(cycle.is_dominating(set(plan.hired(cycle.names))) for plan in product)
    assert all(len(plan.hired(cycle.names)) == 2 for plan in product)


def test_squared_penalty_agrees_on_the_advertising_network():
    network = SocialNetwork.advertising()
    ground = ground_states(exactly_one_cost(network, penalty=2.0))
    assert [plan.hired(network.names) for plan in ground] == [("Ana", "Enrique")]


def test_channels_of_the_minimum_plan():
    network = SocialNetwork.advertising()
    assert network.informant({"Ana", "Enrique"}) == {
        "Mario": "Ana",
        "Sarah": "Enrique",
        "Raúl": "Enrique",
        "Ana": "Ana",
        "Enrique": "Enrique",
        "Saúl": "Ana",
    }
