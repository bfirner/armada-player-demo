# Copyright Bernhard Firner, 2020
# Should be run with pytest:
# > python3 -m pytest

import numpy
import pytest
import torch

import ship
import utility
from game_constants import (ArmadaTypes)
from game_engine import handleAttack
from random_agent import (RandomAgent)
from world_state import (WorldState)

# Initialize ships from the test ship list
keys, ship_templates = utility.parseShips('data/test_ships.csv')

# Test the defense tokens by comparing the results of the test ships with and without those tokens

def a_vs_b(ship_a, ship_b, trials, attack_range):
    """This uses a random agent to choose actions during attacks from ship_a to ship_b.

    Args:
      ship_a ((Ship, str)): Attacker and hull zone tuple.
      ship_b ((Ship, str)): Defender and hull zone tuple.
      trials (int): Number of trials in average calculation.
      range (str): Attack range.
    Returns:
      state_log (List[List[("state" or "action", (WorldState or action tuple))]])
    
    """
    agent = RandomAgent()
    state_log = []
    for trial in range(trials):
        # Reset ship b for each trial
        ship_b.reset()
        world_state = WorldState()
        world_state.addShip(ship_a, 0)
        world_state.addShip(ship_b, 1)
        num_rolls = 0
        while ship_b.damage_cards() < ship_b.hull():
            num_rolls += 1
            # Handle the attack and receive the updated world state
            try:
                world_state = handleAttack(world_state=world_state, attacker=(ship_a, "front"),
                                           defender=(ship_b, "front"), attack_range=attack_range,
                                           offensive_agent=agent, defensive_agent=agent,
                                           state_log=state_log)
            except RuntimeError:
                # This is fine, the random agent will do illegal things plenty of times
                pass
    return state_log


def test_random_agent():
    """Test all possible defense tokens are used and accuracy target everything."""
    attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[], player_number=1)

    ship_a = ship.Ship(name="Ship A", template=ship_templates["All Defense Tokens"], upgrades=[], player_number=1)
    ship_b = ship.Ship(name="Ship B", template=ship_templates["All Defense Tokens"], upgrades=[], player_number=2)

    # Verify that all of the defense tokens are being used and accuracied.
    use_counts = torch.zeros(len(ArmadaTypes.defense_tokens))
    accuracy_targets = torch.zeros(2 * len(ArmadaTypes.defense_tokens))

    # Test with 10 trials at each range to compensate for the natural variability in rolls
    attacks = []
    for attack_range in ['long', 'medium', 'short']:
        attacks = attacks + a_vs_b(ship_a, ship_b, 100, attack_range)
    # Loop through all attacks and increment the used tokens
    for attack in attacks:
        if 'state' == attack[0] and attack[1].sub_phase == "attack - resolve damage":
            # Check the spent token types
            use_counts += torch.Tensor(attack[1].attack.spent_types)

    # We aren't handling the salvo token yet, but check that the others are being spent.
    for tidx, token in enumerate(ArmadaTypes.defense_tokens):
        if 'salvo' != ArmadaTypes.defense_tokens[tidx]:
            assert 0 < use_counts[tidx].item()

    # Check accuracy usage
    for attack in attacks:
        if 'state' == attack[0] and attack[1].sub_phase == "attack - spend defense tokens":
            # Check the spent tokens from the previous resolve attack effects phase
            tokens_left = len(attack[1].attack.accuracy_tokens)
            accuracy_targets[:tokens_left] += torch.Tensor(attack[1].attack.accuracy_tokens)

    # Check that accuracies are used against any of the tokens.
    print("accuracy targets are {}".format(accuracy_targets))
    for tidx, token in enumerate(ArmadaTypes.defense_tokens):
        if 'salvo' != ArmadaTypes.defense_tokens[tidx]:
            assert 0 < accuracy_targets[tidx]
