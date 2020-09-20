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
        # Keep track of which tokens are spent and what effects are enabled.
        self.defense_effects = [0.] * len(ArmadaTypes.defense_tokens)
        self.spent_types = [0.] * len(ArmadaTypes.defense_tokens)
        # A token targeted with an accuracy cannot be spent.
        # A count of token types that were the target of an accuracy.
        # Double because there are both green and red types
        # TODO FIXME This has changed from ship tokens to token types, fix everywhere it is used
        self.accuracy_tokens = [0.] * 2 * len(ArmadaTypes.defense_tokens)

    @staticmethod
    def encode_size():
        """Get the size of the AttackState encoding.

        Returns:
            int: Size of the encoding (number of Tensor elements)
        """
        # 1 each for player number, remaining hull, current speed, ship type
        size = 4
        # Ship size
        size += len(ArmadaDimensions.ship_bases)
        # Encode defense tokens as a number of each kind. Count color as a type as well.
        size += len(ArmadaTypes.defense_tokens) * len(ArmadaTypes.token_colors)
        # Shield value for each hull zone
        size += len(ArmadaTypes.hull_zones)
        # TODO Upgrades
        # Command dials
        size += ArmadaTypes.max_command_dials
        return size

    def accuracy_defender_token(self, token_type, color_type):
        """
        Spend an accuracy on one of the defener's tokens.

        Args:
            token_type (int): Token type to spend (see ArmadaTypes.defense_tokens)
            color_type (int): 0 for green, 1 for red (see ArmadaTypes.token_colors)
        """
        offset = token_type
        offset += color_type * len(ArmadaTypes.defense_tokens)
        self.accuracy_tokens[offset] += 1.

    # TODO FIXME HERE Update this function with the new arguments
    def defender_spend_token(self, token_type, color_type):
        """
        Mark a token as spent and change its color or discard it. This token cannot be spent again
        during the attack.

        Args:
            token_type (str): Token type to spend.
            color_type (int): 0 for green, 1 for red
        """
        self.defender.spend_token(token_type, color_type)
        self.spent_types[ArmadaTypes.defense_tokens.index(token_type)] = 1.
        self.defense_effects[ArmadaTypes.defense_tokens.index(token_type)] = 1

    def attacker_spend_token(self, token_type, color_type):
        """
        Mark a token as spent and change its color or discard it.

        Args:
            token_type (str): Token type to spend.
            color_type (int): 0 for green, 1 for red
        """
        # Flip the token
        self.attacker.spend_token(token_type, color_type)

    def token_type_spent(self, token_type):
        """Return true if the given token type has been spent in this attack."""
        return self.spent_types[ArmadaTypes.defense_tokens.index(token_type)]

    def leave_spend_defense_tokens(self):
        """No tokens should be in the spent state outside of this phase."""
        # This shouldn't need to be run, the attack will stop existing and this goes away
        #self.spent_types = [0.] * len(ArmadaTypes.defense_tokens)

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
        self.round = 0

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
            self.attack.leave_spend_defense_tokens()
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
