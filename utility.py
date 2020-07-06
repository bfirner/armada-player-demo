# Utility functions.

import csv
from game_constants import (ArmadaTypes)

# Array of matching token indexes
def token_index(token, defender):
    return [idx for idx in range(len(defender.defense_tokens)) if token in defender.defense_tokens[idx]]

def adjacent_hulls(hull_zone):
    # TODO This does not handle huge ships
    adjacent_map = {"left": ["rear", "front"],
                    "front": ["left", "right"],
                    "right": ["front", "rear"],
                    "rear": ["right", "left"]
                    }
    if hull_zone not in adjacent_map.keys():
        raise Exception("Unknown hull zone given to adjacent_hulls function.")
    return adjacent_map[hull_zone]

def tokens_available(token, defender, accuracy_tokens = None):
    """Return a tuple indicating if a red or green token is available.

    Arguments:
        token (str)   : The token types (one of ArmadaTypes.defense_tokens) 
        defender(Ship): The defending ship whose tokens to check.
    Returns:
        tuple(bool, bool): True if a green or red token is available, respectively.
    """
    green = False
    red = False
    token_offset = ArmadaTypes.defense_tokens.index(token)
    green_offset, green_size = defender.get_index("green_defense_tokens")
    red_offset, red_size = defender.get_index("red_defense_tokens")
    green_offset += token_offset
    red_offset += token_offset
    green_sum = defender.encoding[green_offset].item()
    red_sum = defender.encoding[green_offset].item()
    if accuracy_tokens:
        green_sum = max(0., green_sum - accuracy_tokens[token_offset])
        red_sum -= max(0., red_sum - accuracy_tokens[token_offset + len(ArmadaTypes.defense_tokens)])
    return (green_sum, red_sum)

def max_damage_index(pool_faces):
    for face in ["hit_crit", "hit_hit", "crit", "hit"]:
        if face in pool_faces:
            return pool_faces.index(face)
    return None

def face_index(face, pool_faces):
    return [idx for idx in range(len(pool_faces)) if face in pool_faces[idx]]

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
