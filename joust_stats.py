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

keys, ship_templates = utility.parseShips('../../armada-ship-stats.csv')
#print("keys are", keys)
#print("ships are ", ship_templates)

first_ship_names = []
if 'all' == args.ship1:
    first_ship_names = [name for name in ship_templates.keys()]
else:
    first_ship_names = [args.ship1]

if 'all' == args.ship2:
    second_ship_names = [name for name in ship_templates.keys()]
else:
    second_ship_names = [args.ship2]

for distance in args.ranges:
    if distance not in ["long", "medium", "short"]:
        print("Unknown range for ship combat: {}".format(distance))
        sys.exit(1)

# Loop through all pairs and have them joust
for ship_name_1 in first_ship_names:
    ship_1 = ship.Ship(name=ship_name_1, template=ship_templates[ship_name_1], upgrades=[], player_number=1)
    for ship_name_2 in second_ship_names:
        for distance in ["long", "medium", "short"]:
            # Make sure we are actually rolling dice
            a_colors, a_roll = ship_1.roll("front", distance)
            if 0 < len(a_colors):
                roll_counts = []
                print("{} vs {} at distance {}".format(ship_name_1, ship_name_2, distance))
                for trial in range(250):
                    # Reset ship b for each trial
                    ship_2 = ship.Ship(name=ship_name_2, template=ship_templates[ship_name_2], upgrades=[], player_number=2)
                    num_rolls = 0
                    while 0 < ship_2.hull():
                        num_rolls += 1
                        a_colors, a_roll = ship_1.roll("front", distance)

                        world_state = {
                            'attack': {
                                'range': distance,
                                'defender': ship_2,
                                'pool_colors': a_colors,
                                'pool_faces': a_roll,
                            }
                        }
                        world_state = spendDefenseTokens(world_state, "spend defense tokens", 1)
                        ship_2.damage("front", ArmadaDice.pool_damage(world_state['attack']['pool_faces']))
                    roll_counts.append(num_rolls)
                np_counts = numpy.array(roll_counts)
                print("Ship {} destroys {} in {} average rolls, stddev = {}, at distance {}.".format(ship_name_1, ship_name_2, np_counts.mean(), np_counts.var()**0.5, distance))
        # TODO(take defense tokens into account)
        #  Make a base player class that allows you to bind a function for this stuff
        # TODO(take accuracy into account)
        #  Again, need to create an interface to plug in an agent
        # TODO(take upgrades and admirals into account)
        #  Much more complicated
        # TODO(Add in positions and movement)
        # TODO(repeat for double arcs, just side arcs)

