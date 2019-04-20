#! /usr/bin/python3

# Game constants describe the game phases and other immutable parts of the gameplay.
# It is important to notice that the order of many phases are modified by different
# upgrades or effects so the ordering of events is not immutable.
# 


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
