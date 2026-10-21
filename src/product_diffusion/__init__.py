"""Product diffusion on a friendship network, solved with QAOA."""

from product_diffusion.classical import ground_states, greedy_dominating_set
from product_diffusion.cost import at_least_one_cost, exactly_one_cost
from product_diffusion.network import SocialNetwork

__all__ = [
    "SocialNetwork",
    "at_least_one_cost",
    "exactly_one_cost",
    "ground_states",
    "greedy_dominating_set",
]
