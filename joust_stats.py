#! /usr/bin/python3

# Just a simple script that compares outcomes when ships of different types joust.
# No maneuvering is taken into account.

import numpy
import random

import argparse
import ship
import sys
import utility
from dice import ArmadaDice
from base_agent import spendDefenseTokens


# Seed with time or a local source of randomness
random.seed()

parser = argparse.ArgumentParser(description='Process dice counts.')
parser.add_argument('--ship1', type=str, required=True, help='Name of a ship or "all"')
parser.add_argument('--ship2', type=str, required=True, help='Name of a ship or "all"')
parser.add_argument('--ranges', type=str, nargs='+', required=True, help='Ranges (short, medium, long)')
# TODO Allow specification of hull zones

args = parser.parse_args()

keys, ship_templates = utility.parseShips('data/armada-ship-stats.csv')
#print("keys are", keys)
#print("ships are ", ship_templates)

first_ship_names = []
if 'all' == args.ship1:
    first_ship_names = [name for name in ship_templates.keys()]
else:
    first_ship_names = [args.ship1]
    if args.ship1 not in ship_templates.keys():
        print("Unrecognized ship name {}".format(args.ship1))
        print("Recognized ship names are:\n")
        for name in ship_templates.keys():
            print("\t{}".format(name))
        exit(1)

if 'all' == args.ship2:
    second_ship_names = [name for name in ship_templates.keys()]
else:
    second_ship_names = [args.ship2]
    if args.ship2 not in ship_templates.keys():
        print("Unrecognized ship name {}".format(args.ship2))
        print("Recognized ship names are:\n")
        for name in ship_templates.keys():
            print("\t{}".format(name))
        exit(1)

for distance in args.ranges:
    if distance not in ["long", "medium", "short"]:
        print("Unknown range for ship combat: {}".format(distance))
        sys.exit(1)

# TODO This function should eventually part of a larger system to handle game logic.
def handleAttack(world_state, attacker, defender, attack_range):
    """This function handles the attack sub-phase of the ship phase.

    Args:
        world_state (table) : Contains the list of ships, objects, etc that comprise the current game.
        attacker (Tuple(Ship, hull zone)) : A tuple of the attacking ship and the string that denotes a hull zone.
        defender (Tuple(Ship, hull zone)) : A tuple of the defending ship and the string that denotes a hull zone.
        attack_range (str) : "long", "medium", or "short". TODO Remove this once ships have locations.
    Returns:
        world_state (table) : The world state after the attack completes.
    """
    attack_ship, attack_hull = attacker[0], attacker[1]
    pool_colors, pool_faces = attack_ship.roll(attack_hull, attack_range)
    world_state['attack'] = {
        'range': attack_range,
        'defender': defender[0],
        'defending zone': defender[1],
        'pool_colors': pool_colors,
        'pool_faces': pool_faces,
        # Keep track of which tokens are spent.
        # Tokens cannot be spent multiple times in a single attack.
        'spent_tokens': {}
    }
    spent_tokens = world_state['attack']['spent_tokens']
    redirect_hull = None
    redirect_amount = None

    # Handle defense tokens while valid ones are spent
    def token_index(token, defender):
        return [idx for idx in range(len(defender.defense_tokens)) if token in defender.defense_tokens[idx]]

    def spend(idx, defender):
        tokens = defender.defense_tokens
        if 'green' in tokens[idx]:
            defender.defense_tokens[idx] = tokens[idx].replace('green', 'red')
        else:
            # Token is red if it is not red, discard it
            defender.defense_tokens = tokens[0:idx] + tokens[idx+1:]

    token, token_targets = spendDefenseTokens(world_state, ("ship phase", "attack - spend defense tokens"))
    while None != token and token in defender[0].defense_tokens:
        # Spend the token and resolve the effect
        spend(token_index(token, defender))
        # TODO If a token has already been spent it cannot be spent again, we should enforce that here.
        if "brace" in token:
            spent_tokens['brace'] = True
        elif "scatter" in token:
            spent_tokens['scatter'] = True
        elif "contain" in token:
            spent_tokens['contain'] = True
        elif "redirect" in token:
            spent_tokens['redirect'] = True
            redirect_hull, redirect_amount = token_targets
        elif "evade" in token:
            # Need to evade a specific die
            die_index = token_targets
            matches = [idx for idx in range(len(pool_colors)) if color == pool_colors[idx] and face == pool_faces[idx]]
            if die_index >= len(pool_colors):
                print("Warning: could not find specified evade token target.")
            else:
                spent_tokens['evade'] = True
                if 'long' == attack_range:
                    # Remove the die
                    pool_colors = pool_colors[:die_index] + pool_colors[die_index+1:]
                    pool_faces = pool_faces[:die_index] + pool_faces[die_index+1:]
                elif 'medium' == attack_range:
                    # Reroll the die
                    pool_faces[die_index] = ArmadaDice.random_roll(pool_faces[die_index])
        # See if the agent will spend another defense token
        token, token_target = spendDefenseTokens(world_state, ("ship phase", "attack - spend defense tokens"))

    # TODO The defender may have ways of modifying the attack pool at this point

    # Now calculate the damage done to the defender
    # Apply effect on the entire attack at this point
    damage = ArmadaDice.pool_damage(pool_faces)
    if 'scatter' in spent_tokens:
        damage = 0

    if 'brace' in spent_tokens:
        damage = damage // 2 + damage % 2

    if 'redirect' in spent_tokens:
        # Redirect to the target hull zone, but don't redirect more damage than is present
        redirect_amount = min(damage, redirect_amount)
        defender[0].damage(redirect_hull, redirect_amount)
        damage = damage - redirect_amount
        world_state['attack']['{} damage'.format(redirect_hull)] = redirect_amount

    # Deal remaining damage to the defender
    world_state['attack']['{} damage'.format(defender[1])] = damage
    defender[0].damage(defender[1], damage)

    # TODO Log the world state into a game log

    # Clear the attack-only states and return the world state with modified ship status
    world_state['attack'] = {}
    return world_state

    

# Loop through all pairs and have them joust
for ship_name_1 in first_ship_names:
    ship_1 = ship.Ship(name=ship_name_1, template=ship_templates[ship_name_1], upgrades=[], player_number=1)
    for ship_name_2 in second_ship_names:
        for attack_range in ["long", "medium", "short"]:
            # Make sure we are actually rolling dice
            a_colors, a_roll = ship_1.roll("front", attack_range)
            if 0 < len(a_colors):
                roll_counts = []
                print("{} vs {} at range {}".format(ship_name_1, ship_name_2, attack_range))
                for trial in range(250):
                    # Reset ship b for each trial
                    ship_2 = ship.Ship(name=ship_name_2, template=ship_templates[ship_name_2], upgrades=[], player_number=2)
                    world_state = {
                        'ships': [ ship_1, ship_2 ]
                    }
                    num_rolls = 0
                    while 0 < ship_2.hull():
                        num_rolls += 1
                        a_colors, a_roll = ship_1.roll("front", attack_range)

                        world_state = {
                            'attack': {
                                'range': attack_range,
                                'defender': ship_2,
                                'pool_colors': a_colors,
                                'pool_faces': a_roll,
                            }
                        }
                        # Handle the attack and receive the updated world state
                        world_state = handleAttack(world_state, (ship_1, "front"), (ship_2, "front"), attack_range)
                    roll_counts.append(num_rolls)
                np_counts = numpy.array(roll_counts)
                print("Ship {} destroys {} in {} average rolls, stddev = {}, at range {}.".format(ship_name_1, ship_name_2, np_counts.mean(), np_counts.var()**0.5, attack_range))
        # TODO(take defense tokens into account)
        #  Make a base player class that allows you to bind a function for this stuff
        # TODO(take accuracy into account)
        #  Again, need to create an interface to plug in an agent
        # TODO(take upgrades and admirals into account)
        #  Much more complicated
        # TODO(Add in positions and movement)
        # TODO(repeat for double arcs, just side arcs)

