#! /usr/bin/python3

#
# Copyright Bernhard Firner, 2019-2020
#
# Game constants describe the game phases and other immutable parts of the gameplay.
# It is important to notice that the order of many phases are modified by different
# upgrades or effects so the ordering of events is not immutable.
# 


class ArmadaDimensions:
    # Ship sizes in mm (from the Armada FAQ) in (width, height) tuples
    # TODO Should differentiate between plastic and cardboard dimensions
    # TODO Add huge ship
    size_names = ['small', 'medium', 'large', 'huge']
    # The rules reference gives sizes in mm, even though the ruler and game area are measured in
    # feet. By default the game unit will always be feet in this project, but here we will store
    # these variables in mm so that it is more obvious to a reader if they are out of date.
    ship_bases_mm = {
        "small": (43,71),
        "medium": (63,102),
        "large": (77.5,129)
        # TODO Huge
    }
    ship_bases_feet = {
        "small": (ship_bases_mm['small'][0]/304.8,ship_bases_mm['small'][1]/304.8),
        "medium": (ship_bases_mm['medium'][0]/304.8,ship_bases_mm['medium'][1]/304.8),
        "large": (ship_bases_mm['large'][0]/304.8,ship_bases_mm['large'][1]/304.8),
        # TODO Huge
    }
    # TODO Cardboard size (used to determine attack range)
    ruler_distance_mm = [71.5, 125, 185, 245, 305]
    ruler_distance_feet = [distance / 304.8 for distance in ruler_distance_mm]
    ruler_range_mm = {
        "short": 123,
        "medium": 187,
        "long": 305
    }
    ruler_range_feet = {
        "short": ruler_range_mm["short"] / 304.8,
        "medium": ruler_range_mm["medium"] / 304.8,
        "long": ruler_range_mm["long"] / 304.8
    }
    # TODO Squad bases
    # Shield dial protrubrence
    # Shield dials count towards overlapping of obstacles, squadrons, and ships

class ArmadaTypes:
    defense_tokens = ["evade", "brace", "scatter", "contain", "redirect", "salvo"]
    token_colors = ["green", "red"]
    green = token_colors.index('green')
    red = token_colors.index('red')
    max_defense_tokens = 6
    max_command_dials = 6
    ranges = ["short", "medium", "long"]
    # TODO Expand this for huge ships
    hull_zones = ["left", "right", "front", "rear", "left-auxiliary", "right-auxiliary"]
    adjacent_hull_zones = {
        "left": ("front", "rear"),
        "right": ("front", "rear"),
        "front": ("left", "right"),
        "rear": ("left", "right")}
    adjacent_huge_hull_zones = {
        "left": ("front", "left-auxiliary"),
        "right": ("front", "right-auxiliary"),
        "left-auxiliary": ("left", "rear"),
        "right-auxiliary": ("right", "rear"),
        "front": ("left", "right"),
        "rear": ("left-auxiliary", "right-auxiliary")}

class ArmadaPhases:
    main_phases = [
        "command phase",                    # Assign command dials to all ships
        "ship phase",                       # Players alternate activating ships or passing
        "squadron phase",                   # Players alternate activating two squadrons at a time
        "status phase"                      # Ready defense tokens and exhausted upgrades, increment round number or finish the game
    ]

    # This enumeration may seem a little silly, but it allows for more
    # flexibility in the future if certain upgrades create new actions in some
    # of these phases or change the order of actions.
    sub_phases = {
        "command phase": [
            "assign command dials"
        ],
        "ship phase": [
            "reveal command dial", # Choose to take the dial as a command or token. Perform squadron and/or engineering.
            "attack - declare",    # Declare a target to attack
            "attack - roll attack dice",       # Roll the initial dice pool
            "attack - resolve attack effects", # Manipulate dice in the dice pool
            "attack - spend defense tokens",   # The defender spends tokens to modify damage or the dice pool
            "attack - resolve damage",         # Resolve damage and critical effects
            "attack - declare additional squadron target", # Only if attacking squadrons
            "execute maneuver",    # Resolve maneuver commands and move.
        ],
        "squadron phase": [
            "move squadron",
            "attack"
        ],
        "status phase": [
            "ready defense tokens",
            "unexhaust upgrades",
            "increment round number"
        ]
    }
