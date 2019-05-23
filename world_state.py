#! /usr/bin/python3

#
# Copyright Bernhard Firner, 2019
#
# This file holds the world state implementation, which keeps track of the location of objects, the
# game phase, object attributes, etc.
#

class AttackState:
    """The state of a single attack."""

    def __init__(self, attack_range, defender, defending_hull, pool_colors, pool_faces):
        self.range = attack_range
        self.defender = defender
        self.defending_hull = defending_hull
        self.pool_colors = pool_colors
        self.pool_faces = pool_faces
        # Keep track of which tokens are spent.
        # Tokens cannot be spent multiple times in a single attack.
        self.spent_tokens = {},
        # A token targeted with an accuracy cannot be spent.
        self.accuracy_tokens = []

    def __str__(self):
        return str("Attack to {} at range {}: {}".format(self.defender, self.range,
            self.pool_colors))

    def __repr__(self):
        return str("Attack to {} at range {}: {}".format(self.defender, self.range,
            self.pool_colors))

class WorldState:
    """Object with the complete game state."""

    def __init__(self):
        # Initial game state
        self.main_phase = "setup phase"
        self.sub_phase = ""
        self.full_phase = "setup phase"

        # No current attack
        self.attack = None
        # TODO Distinct player states
        self.ships = []
        self.ship_players = {}

    def setPhase(self, main, sub):
        """Set the current phase."""
        self.main_phase = main
        self.sub_phase = sub
        self.full_phase = main + " - " + sub

    def setSubPhase(self, sub):
        """Modify the sub phase."""
        self.sub_phase = sub
        self.full_phase = self.main_phase + " - " + sub

    def addShip(self, ship, player_number):
        self.ships.append(ship)
        self.ship_players[ship] = player_number

    def updateAttack(self, attack_dict):
        self.attack = attack_dict
