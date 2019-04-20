# Utility functions.

import csv

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

def print_roll(colors, roll):
    for i in range(0, len(colors)):
        print("{}: {} {}".format(i, colors[i], roll[i]))
