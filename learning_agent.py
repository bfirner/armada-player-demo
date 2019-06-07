#
# Copyright Bernhard Firner, 2019
#
import random

import ship
from dice import ArmadaDice
from base_agent import BaseAgent 
from game_constants import (ArmadaPhases, ArmadaTypes)
from utility import token_index

import torch


class LearningAgent(BaseAgent):

    def __init__(self):
        """Initialize the simple agent with a cuople of simple state handlers."""
        handler = {
                "attack - resolve attack effects": self.resolveAttackEffects,
                "attack - spend defense tokens": self.spendDefenseTokens
        }
        super(LearningAgent, self).__init__(handler)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Make some constants here to avoid magic numbers in the rest of this code
        self.max_defense_tokens = ArmadaTypes.max_defense_tokens
        self.hot_token_size = len(ArmadaTypes.defense_tokens) + len(ArmadaTypes.token_colors) + 1
        self.hot_die_size = 9
        self.max_die_slots = 16

    def encodeAttackState(self, world_state):
        """
        Args:
            world_state (WorldState) : Current world state
        Returns:
            (torch.Tensor)           : Encoding of the world state used as network input.
        """
        enc_size = 0
        # Encode the dice pool and faces, defense tokens, shields, and hull.
        # The input is much simpler if it is of fixed size. The inputs are:
        # attacker hull and shields - 5
        # defender hull and shields - 5
        enc_size += 2*5
        # 
        # Need to encode each tokens separately, with a one-hot vector for the token type, color,
        # and whether it was targetted with an accuracy. The maximum number of tokens is six, for
        # the SSD huge ship. The order of the tokens should be randomized with each encoding to
        # ensure that the model doesn't learn things positionally or ignore later entries that are
        # less common.
        # spent token types - 5
        enc_size += len(ArmadaTypes.defense_tokens)
        # max_defense_tokens * [type - 5, color - 2, accuracy targeted - 1] = 48
        enc_size += self.max_defense_tokens * self.hot_token_size
        #
        # attack range - 3 (1-hot encoding)
        enc_size += len(ArmadaTypes.ranges)
        #
        # We have a couple of options for the dice pool.
        # The simplest approach is to over-provision a vector with say 16 dice. There would be 3
        # color vectors and 6 face vectors for a total of 9*16=144 inputs.
        # 16 * [ color - 3, face - 6]
        enc_size += self.max_die_slots * self.hot_die_size
        # During training we want the model to react properly to a die in any location in the
        # vector, so we randomize the dice locations so that the entire vector is used (otherwise
        # the model would be poorly trained for rolls with a very large number of dice)
        # Total encoding size: 5 + 5 + 5 + 48 + 3 + 144 = 210
        assert(enc_size == 210)
        encoding = torch.zeros(1, enc_size).to(self.device)

        # Now populate the tensor
        attack = world_state.attack
        defender = attack['defender']
        attacker = attack['attacker']
        spent_tokens = attack['spent_tokens']
        accuracy_tokens = attack['accuracy_tokens']
        pool_colors = attack['pool_colors']
        pool_faces = attack['pool_faces']

        # Hull zones will of course increase with huge ships
        # Future proof this code by having things basing math upon the length of this table
        hull_zones = ["left", "front", "right", "rear"]

        cur_offset = 0

        # Attacker shields and hull
        for offset, zone in enumerate(hull_zones):
            encoding[0, cur_offset + offset] = attacker.shields[zone]
        encoding[0, cur_offset + len(hull_zones)] = attacker.hull()
        cur_offset += len(hull_zones) + 1

        # Defender shields and hull
        for offset, zone in enumerate(hull_zones):
            encoding[0, 6 + offset] = defender.shields[zone]
        encoding[0, cur_offset + len(hull_zones)] = defender.hull()
        cur_offset += len(hull_zones) + 1

        # Defense tokens

        # Spent tokens
        for offset, token in enumerate(ArmadaTypes.defense_tokens):
            encoding[0, cur_offset + offset] = 1 if token in spent_tokens else 0
        cur_offset += len(ArmadaTypes.defense_tokens)

        # Hot encoding the defenders tokens
        # Each hot encoding is: [type - 5, color - 2, accuracy targeted - 1]
        slots = random.sample(range(self.max_defense_tokens), len(defender.defense_tokens))
        for token_idx, slot in enumerate(slots):
            slot_offset = cur_offset + slot * self.hot_token_size
            token = defender.defense_tokens[token_idx]
            # Encode the token type
            for offset, ttype in enumerate(ArmadaTypes.defense_tokens):
                encoding[0, slot_offset + offset] = 1 if ttype in token else 0
            slot_offset = slot_offset + len(ArmadaTypes.defense_tokens)
            # Encode the token color
            for offset, color in enumerate(ArmadaTypes.token_colors):
                encoding[0, slot_offset + offset] = 1 if color == token[0:len(color)] else 0
            slot_offset = slot_offset + len(ArmadaTypes.token_colors)
            # Encode whether an accuracy has been spent on this token
            encoding[0, slot_offset] = 1 if token_idx in accuracy_tokens else 0

        # Move the current encoding offset to the position after the token section
        cur_offset += self.hot_token_size * self.max_defense_tokens

        # Attack range
        for offset, arange in enumerate(ArmadaTypes.ranges):
            encoding[0, cur_offset + offset] = 1 if arange == attack["range"] else 0
        cur_offset += len(ArmadaTypes.ranges)

        # Each die will be represented with a hot_die_size vector
        # There are max_die_slots slots for these, and we will fill them in randomly
        slots = random.sample(range(self.max_die_slots), len(pool_colors))
        for die_idx, slot in enumerate(slots):
            slot_offset = cur_offset + slot * self.hot_die_size
            # Encode die colors
            for offset, color in enumerate(ArmadaDice.die_colors):
                encoding[0, slot_offset + offset] = 1 if color == pool_colors[die_idx] else 0
            slot_offset += len(ArmadaDice.die_colors)
            # Encode die faces
            for offset, face in enumerate([face for face in ArmadaDice.die_faces.keys()]):
                encoding[0, slot_offset + offset] = 1 if face == pool_faces[die_idx] else 0

        # Sanity check on the encoding size and the data put into it
        assert encoding.size(1) == cur_offset + self.hot_die_size * self.max_die_slots

        return encoding

    # This agent deals with the "resolve attack effects" step.
    def resolveAttackEffects(self, world_state):
        """
        Args:
            world_state (WorldState)   : Contains the list of ships and dice pool.
            current_step (string) : This function only operates on the "resolve attack effects" step.
            TODO The current_step should be rolled into the world state
        Returns:
            (array[varies]) : Array could contain a tuple of a die with an accuracy face and a defender's
                              defense token, but can also be other effects, such as additional dice to
                              roll or selected dice to reroll. Currently we only handle spending accuracy
                              icons so the only return type is the tuple: (index, token).
                              In the future we may support other tuples such as:
                                ("add", colors), ("reroll", indices), or ("remove", indices)
        """
        # We only handle one sub-phase in this function
        assert world_state.full_phase == "attack - resolve attack effects"

        # Encode the state, forward through the network, decode the result, and return the result.
        pass

    # This agent deals with the "spend defense tokens" step.
    def spendDefenseTokens(self, world_state):
        """
        Args:
            world_state (table)   : Contains the list of ships and dice pool.
            current_step (string) : This function only operates on the "spend defense tokens" step.
        Returns:
            (str, varies) : A tuple of the token name to spend and the token targets of die to target
                            with the token.  If no token will be spent then both values of the tuple are
                            None and if the defense token has no target then the second value is None.
                            For evade tokens the second value is the index of the die to target.
                            For redirect tokens the second value is a tuple of (str, int) for the
                            target hull zone and the amount of damage to direct to that hull zone.
        """
        # We only handle one sub-phase in this function
        assert world_state.full_phase == "attack - spend defense tokens"

        # Encode the state, forward through the network, decode the result, and return the result.
        pass
