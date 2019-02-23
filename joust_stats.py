#! /usr/bin/python3

# Just a simple script that compares outcomes when ships of different types joust.
# No maneuvering is taken into account.

import csv
import random

import ship
from dice import ArmadaDice


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
for ship_name_a in ship_templates:
    ship_a = ship.Ship(name=ship_name_a, template=ship_templates[ship_name_a], upgrades=[])
    for ship_name_b in ship_templates:
        for distance in ["long", "medium", "short"]:
            # Reset ship b for each distane
            ship_b = ship.Ship(name=ship_name_b, template=ship_templates[ship_name_b], upgrades=[])
            print("{} vs {} at distance {}".format(ship_name_a, ship_name_b, distance))
            a_colors, a_roll = ship_a.roll("front", distance)
            b_colors, b_roll = ship_b.roll("front", distance)
            #print("Ship a rolls:")
            #print_roll(a_colors, a_roll)
            #print("For {} damage.".format(ArmadaDice.pool_damage(a_roll)))
            ship_b.damage("front", ArmadaDice.pool_damage(a_roll))
            # Make sure we are actually rolling dice
            if 0 < len(a_colors):
                num_rolls = 1
                while 0 < ship_b.hull():
                    num_rolls += 1
                    a_colors, a_roll = ship_a.roll("front", distance)
                    #print("Ship a rolls:")
                    #print_roll(a_colors, a_roll)
                    #print("For {} damage.".format(ArmadaDice.pool_damage(a_roll)))
                    ship_b.damage("front", ArmadaDice.pool_damage(a_roll))
                print("Ship {} destroys {} in {} rolls and distance {}.".format(ship_name_a, ship_name_b, num_rolls, distance))

        #print("Ship b rolls:")
        #for i in range(0, len(b_colors)):
        #    print("{}: {} {}".format(i, b_colors[i], b_roll[i]))
        #print("For {} damage.".format(ArmadaDice.pool_damage(b_roll)))
        # TODO(calculate number of rolls to deplete the other ship's hull)
        # Probably only necessary to go from a to b
        # TODO(run multiple simulations)
        # TODO(take defense tokens into account)
        # TODO(take upgrades and admirals into account)
        # TODO(repeat for double arcs, just side arcs)

