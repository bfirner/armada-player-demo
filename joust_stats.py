#! /usr/bin/python3

# Just a simple script that compares outcomes when ships of different types joust.
# No maneuvering is taken into account.

import csv
import numpy
import random

import ship
from dice import ArmadaDice
from base_agent import spendDefenseTokens


# Seed with time or a local source of randomness
random.seed()

def parseShips(filename):
    keys = {}
    ship_templates = {}
    with open(filename, newline='') as ships:
        shipreader = csv.reader(ships, delimiter=',', quotechar='|')
        rowcount = 0
        for row in shipreader:
            # parse the header first to find the column keys
            if ( 0 == rowcount ):
                count = 0
                for key in row:
                    count = count + 1
                    keys[count] = key
            else:
                newship = {}
                count = 0
                # Fill in all of the information on this vessel
                for key in row:
                    count = count + 1
                    newship[keys[count]] = key
                # Create a new ship template
                ship_templates[newship['Ship Name']] = newship
            rowcount = rowcount + 1
    return keys, ship_templates


keys, ship_templates = parseShips('../../armada-ship-stats.csv')
#print("keys are", keys)
#print("ships are ", ship_templates)

def print_roll(colors, roll):
    for i in range(0, len(colors)):
        print("{}: {} {}".format(i, colors[i], roll[i]))

# TODO make this a command line argument

# Loop through all pairs and have them joust
for ship_name_1 in ship_templates:
    ship_1 = ship.Ship(name=ship_name_1, template=ship_templates[ship_name_1], upgrades=[], player_number=1)
    for ship_name_2 in ship_templates:
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

