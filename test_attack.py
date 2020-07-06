# Should be run with pytest:
# > python3 -m pytest

import numpy
import pytest

import ship
import utility
from game_engine import handleAttack
from simple_agent import (SimpleAgent)
from world_state import (WorldState)

# Initialize ships from the test ship list
keys, ship_templates = utility.parseShips('data/test_ships.csv')

# Test the defense tokens by comparing the results of the test ships with and without those tokens

def a_vs_b(ship_a, ship_b, trials, attack_range):
    """This function calculates the average time to destruction when a shoots at b.

    Args:
      ship_a ((Ship, str)): Attacker and hull zone tuple.
      ship_b ((Ship, str)): Defender and hull zone tuple.
      trials (int): Number of trials in average calculation.
      range (str): Attack range.
    
    """
    roll_counts = []
    agent = SimpleAgent()
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
            world_state = handleAttack(world_state=world_state, attacker=(ship_a, "front"),
                                       defender=(ship_b, "front"), attack_range=attack_range,
                                       offensive_agent=agent, defensive_agent=agent)
        roll_counts.append(num_rolls)
    np_counts = numpy.array(roll_counts)
    return np_counts.mean()


def test_brace():
    """Test that brace increases the number of attacks required to destroy a ship."""

    attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[], player_number=1)

    no_brace = ship.Ship(name="No Defense Tokens", template=ship_templates["No Defense Tokens"], upgrades=[], player_number=2)
    one_brace = ship.Ship(name="Single Brace", template=ship_templates["Single Brace"], upgrades=[], player_number=2)
    two_brace = ship.Ship(name="Double Brace", template=ship_templates["Double Brace"], upgrades=[], player_number=2)

    # Test with 1000 trials to compensate for the natural variability in rolls
    for attack_range in ['long', 'medium']:
        no_brace_attacks = a_vs_b(attacker, no_brace, 1000, attack_range)
        one_brace_attacks = a_vs_b(attacker, one_brace, 1000, attack_range)
        two_brace_attacks = a_vs_b(attacker, two_brace, 1000, attack_range)

    # Only test brace vs no brace at short range since with the test setup the ships reaches 0 hull
    # before spending all of the brace tokens.
    for attack_range in ['short']:
        no_brace_attacks = a_vs_b(attacker, no_brace, 1000, attack_range)
        one_brace_attacks = a_vs_b(attacker, one_brace, 1000, attack_range)

        assert(no_brace_attacks < one_brace_attacks)
        assert(one_brace_attacks < two_brace_attacks)


#def test_scatter():
#    """Test that scatter increases the number of attacks required to destroy a ship."""
#
#    attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[], player_number=1)
#
#    no_scatter = ship.Ship(name="No Defense Tokens", template=ship_templates["No Defense Tokens"], upgrades=[], player_number=2)
#    two_scatter = ship.Ship(name="Double Scatter", template=ship_templates["Double Scatter"], upgrades=[], player_number=2)
#
#    # Test with 1000 trials to compensate for the natural variability in rolls
#    for attack_range in ['long', 'medium', 'short']:
#        no_scatter_attacks = a_vs_b(attacker, no_scatter, 1000, attack_range)
#        two_scatter_attacks = a_vs_b(attacker, two_scatter, 1000, attack_range)
#
#        assert(no_scatter_attacks < two_scatter_attacks)
#
#
#def test_evade():
#    """Test that evade increases the number of attacks required to destroy a ship at long or medium range."""
#
#    attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[], player_number=1)
#
#    no_evade = ship.Ship(name="No Defense Tokens", template=ship_templates["No Defense Tokens"], upgrades=[], player_number=2)
#    two_evade = ship.Ship(name="Double Evade", template=ship_templates["Double Evade"], upgrades=[], player_number=2)
#
#    # Test with 1000 trials to compensate for the natural variability in rolls
#    no_evade_attacks_long = a_vs_b(attacker, no_evade, 1000, "long")
#    no_evade_attacks_medium = a_vs_b(attacker, no_evade, 1000, "medium")
#    no_evade_attacks_short = a_vs_b(attacker, no_evade, 1000, "short")
#    two_evade_attacks_long = a_vs_b(attacker, two_evade, 1000, "long")
#    two_evade_attacks_medium = a_vs_b(attacker, two_evade, 1000, "medium")
#    two_evade_attacks_short = a_vs_b(attacker, two_evade, 1000, "short")
#
#    # Evades should increase the time to destruction
#    assert(no_evade_attacks_long < two_evade_attacks_long)
#    assert(no_evade_attacks_medium < two_evade_attacks_medium)
#    # Evades do not work at short range
#    assert(pytest.approx(two_evade_attacks_short, 0.1) == no_evade_attacks_short)
