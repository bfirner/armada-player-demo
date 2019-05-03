# Should be run with pytest:
# > python3 -m pytest

import numpy
import pytest

import ship
import utility
from game_engine import handleAttack

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
    for trial in range(trials):
        # Reset ship b for each trial
        ship_b.reset()
        world_state = {
            'ships': [ ship_a, ship_b ]
        }
        num_rolls = 0
        while 0 < ship_b.hull():
            num_rolls += 1
            a_colors, a_roll = ship_a.roll("front", attack_range)

            world_state = {
                'attack': {
                    'range': attack_range,
                    'defender': ship_b,
                    'pool_colors': a_colors,
                    'pool_faces': a_roll,
                }
            }
            # Handle the attack and receive the updated world state
            world_state = handleAttack(world_state, (ship_a, "front"), (ship_b, "front"), attack_range)
        roll_counts.append(num_rolls)
    np_counts = numpy.array(roll_counts)
    return np_counts.mean()

def test_brace():
    """Test that brace increases the number of attacks required to destroy a ship."""
    attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[], player_number=1)

    no_brace = ship.Ship(name="No Defense Tokens", template=ship_templates["No Defense Tokens"], upgrades=[], player_number=2)
    one_brace = ship.Ship(name="Single Brace", template=ship_templates["Single Brace"], upgrades=[], player_number=2)
    two_brace = ship.Ship(name="Double Brace", template=ship_templates["Double Brace"], upgrades=[], player_number=2)

    for attack_range in ['long', 'medium', 'short']:
        no_brace_attacks = a_vs_b(attacker, no_brace, 1000, attack_range)
        one_brace_attacks = a_vs_b(attacker, one_brace, 1000, attack_range)
        two_brace_attacks = a_vs_b(attacker, two_brace, 1000, attack_range)

        assert(no_brace_attacks < one_brace_attacks)
        assert(one_brace_attacks < two_brace_attacks)
