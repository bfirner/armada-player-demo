#! /usr/bin/python3

#
# Copyright Bernhard Firner, 2019
#
# Game constants describe the game phases and other immutable parts of the gameplay.
# It is important to notice that the order of many phases are modified by different
# upgrades or effects so the ordering of events is not immutable.
# 


class ArmadaDimensions:
    # Ship sizes in mm (from the Armada FAQ) in (width, height) tuples
    # TODO Should differentiate between plastic and cardboard dimensions
    ship_bases = {
        "small": (42,71),
        "medium": (63,102),
        "large": (77.5,129)
    }
    # TODO Cardboard size (used to determine attack range)
    # TODO Squad bases

class ArmadaTypes:
    defense_tokens = ["evade", "brace", "scatter", "contain", "redirect", "salvo"]
    token_colors = ["green", "red"]
    max_defense_tokens = 6
    ranges = ["short", "medium", "long"]
    # TODO Expand this for huge ships
    hull_zones = ["left", "right", "front", "rear"]
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
