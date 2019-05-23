# Utility functions.

import csv

# Array of matching token indexes
def token_index(token, defender):
    return [idx for idx in range(len(defender.defense_tokens)) if token in defender.defense_tokens[idx]]

# Array of matching token indexes with greenest tokens first
def greenest_token_index(token, defender):
    greens = []
    reds = []
    for idx, name in enumerate(defender.defense_tokens):
        if "green {}".format(token) == name:
            greens.append(idx)
        elif "red {}".format(token) == name:
            reds.append(idx)
    return greens + reds

# Return the index of the first, greenest token of the given type that can be spent.
def greenest_token(name, defender, accuracy_tokens):
    green_tokens = [idx for idx in token_index('green {}'.format(name), defender) if not idx in
            accuracy_tokens]
    red_tokens = [idx for idx in token_index('red {}'.format(name), defender) if not idx in
            accuracy_tokens]
    if green_tokens or red_tokens:
        return green_tokens[0] if green_tokens else red_tokens[0]
    else:
        return None

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
