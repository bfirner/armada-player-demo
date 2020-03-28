#! /usr/bin/python3

#
# Copyright Bernhard Firner, 2019
#
# This file holds the world state implementation, which keeps track of the location of objects, the
# game phase, object attributes, etc.
#

from game_constants import (ArmadaPhases, ArmadaTypes)

import copy

class AttackState:
    """The state of a single attack."""

    def __init__(self, attack_range, attacker, attacking_hull, defender, defending_hull,
            pool_colors, pool_faces):
        self.range = attack_range
        self.attacker = attacker
        self.attacking_hull = attacking_hull
        self.defender = defender
        self.defending_hull = defending_hull
        self.pool_colors = pool_colors
        self.pool_faces = pool_faces
        # Keep track of which tokens are spent.
        self.spent_types = [False] * len(ArmadaTypes.defense_tokens)
        # A token targeted with an accuracy cannot be spent.
        # A list of indices of tokens that were the target of an accuracy.
        self.accuracy_tokens = [False] * len(defender.defense_tokens)

    def flip_token(self, ship, index):
        """
        Flip a defense token from green to red or discard it if it is red.

        Args:
            index (int): Index of the token to flip or discard

        Returns:
            bool: True if the token remains, false otherwise
        """
        tcolor, ttype = tuple(ship.defense_tokens[index].split(' '))
        if 'green' == tcolor:
            ship.defense_tokens[index] = 'red ' + ttype
            return True
        else:
            # Token is red if it is not green, so discard it instead of flipping
            ship.defense_tokens = ship.defense_tokens[0:index] + ship.defense_tokens[index+1:]
            ship.spent_tokens = ship.spent_tokens[0:index] + ship.spent_tokens[index+1:]
            self.accuracy_tokens = self.accuracy_tokens[0:index] + self.accuracy_tokens[index+1:]
            return False

    def defender_spend_token(self, index):
        """
        Mark a token as spent and change its color or discard it. This token cannot be spent again
        during the attack.

        Args:
            index (int): Index of the token to spend.

        Returns:
            str: Token type
        """
        # Flip the token
        _, ttype = tuple(self.defender.defense_tokens[index].split(' '))
        if self.flip_token(self.defender, index):
            # Mark it as spent if it remains
            self.defender.spent_tokens[index] = True
        self.spent_types[ArmadaTypes.defense_tokens.index(ttype)] = True
        return ttype

    def attacker_spend_token(self, index):
        """
        Mark a token as spent and change its color or discard it.

        Args:
            index (int): Index of the token to spend.
        """
        # Flip the token
        self.flip_token(self.attacker, index)

    def token_type_spent(self, token_type):
        """Return true if the given token type has been spent in this attack."""
        return self.spent_types[ArmadaTypes.defense_tokens.index(token_type)]

    def __str__(self):
        return str("Attack to {} at range {}: {}".format(self.defender, self.range,
            list(zip(self.pool_colors, self.pool_faces))))

    def __repr__(self):
        return str("Attack to {} at range {}: {}".format(self.defender, self.range,
            list(zip(self.pool_colors, self.pool_faces))))

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
        self.setSubPhase(sub)

    def setSubPhase(self, sub):
        """Modify the sub phase."""
        # Handle leaving the spend defense tokens phase
        if self.sub_phase == "attack - spend defense tokens":
            for ship in self.ships:
                ship.leave_spend_defense_tokens()
        self.sub_phase = sub
        self.full_phase = self.main_phase + " - " + sub

    def addShip(self, ship, player_number):
        self.ships.append(ship)
        self.ship_players[ship] = player_number

    def updateAttack(self, attack_state):
        self.attack = attack_state

    def clone(self):
        """Return a clone of the object using new memory."""
        return copy.deepcopy(self)
