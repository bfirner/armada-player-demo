#
# Copyright Bernhard Firner, 2019
#
import random

import ship
from armada_encodings import (Encodings)
from dice import ArmadaDice
from base_agent import BaseAgent 
from game_constants import (ArmadaPhases, ArmadaTypes)
from utility import token_index

import torch


class LearningAgent(BaseAgent):

    def __init__(self, model=None):
        """Initialize the simple agent with a couple of simple state handlers.
        
        Args:
            model (torch.nn.Module or None): If None this agent will pass for all supported states
        """
        handler = {
                "attack - resolve attack effects": self.resolveAttackEffects,
                "attack - spend defense tokens": self.spendDefenseTokens
        }
        super(LearningAgent, self).__init__(handler)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        if None != self.model:
            self.model = self.model.to(self.device)
            self.model.eval()

        self.attack_enc_size = Encodings.calculateAttackSize()
        self.memory = []
        self.remembering = False

    def rememberStateActions(self):
        """Clear existing memory and begin logging pairs of input states and actions."""
        self.remembering = True
        self.memory = []

    def returnStateActions(self):
        """Return the remembered action states and clear them."""
        results = self.memory
        self.memory = []
        self.remembering = False
        return results

    def encodeAttackState(self, world_state):
        """
        Args:
            world_state (WorldState) : Current world state
        Returns:
            (torch.Tensor)           : Encoding of the world state used as network input.
        """
        encoding = torch.zeros(1, self.attack_enc_size).to(self.device)

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
        slots = random.sample(range(ArmadaTypes.max_defense_tokens), len(defender.defense_tokens))
        for token_idx, slot in enumerate(slots):
            slot_offset = cur_offset + slot * Encodings.hot_token_size
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
        cur_offset += Encodings.hot_token_size * ArmadaTypes.max_defense_tokens

        # Attack range
        for offset, arange in enumerate(ArmadaTypes.ranges):
            encoding[0, cur_offset + offset] = 1 if arange == attack["range"] else 0
        cur_offset += len(ArmadaTypes.ranges)

        # Each die will be represented with a hot_die_size vector
        # There are max_die_slots slots for these, and we will fill them in randomly
        # During training we want the model to react properly to a die in any location in the
        # vector, so we randomize the dice locations so that the entire vector is used (otherwise
        # the model would be poorly trained for rolls with a very large number of dice)
        slots = random.sample(range(Encodings.max_die_slots), len(pool_colors))
        for die_idx, slot in enumerate(slots):
            slot_offset = cur_offset + slot * Encodings.hot_die_size
            # Encode die colors
            for offset, color in enumerate(ArmadaDice.die_colors):
                encoding[0, slot_offset + offset] = 1 if color == pool_colors[die_idx] else 0
            slot_offset += len(ArmadaDice.die_colors)
            # Encode die faces
            for offset, face in enumerate([face for face in ArmadaDice.die_faces.keys()]):
                encoding[0, slot_offset + offset] = 1 if face == pool_faces[die_idx] else 0

        # Sanity check on the encoding size and the data put into it
        assert encoding.size(1) == cur_offset + Encodings.hot_die_size * Encodings.max_die_slots

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

        if None == self.model:
            # Return no action
            return []
        # Encode the state, forward through the network, decode the result, and return the result.
        # TODO
        return []

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

        if None == self.model:
            # Return no action
            return None, None
        # Encode the state, forward through the network, decode the result, and return the result.
        as_enc = self.encodeAttackState(world_state)
        action = self.model.forward("def_tokens", as_enc)[0]
        # Remember this state action pair if in memory mode
        if self.remembering:
            self.memory.append((as_enc, action))
        # Clean off the lifetime prediction
        with torch.no_grad():
            action = torch.round(action[:Encodings.calculateSpendDefenseTokensSize()])
        # The output of the model is a one-hot encoding of the token type and color, die index (used
        # with evade), and hull zone index and amount of damage (for redirect)
        ttype_begin = 0
        tcolor_begin = ttype_begin + len(ArmadaTypes.defense_tokens)
        die_begin = tcolor_begin + len(ArmadaTypes.token_colors)
        hull_begin = die_begin + Encodings.max_die_slots
        redirect_ammount = hull_begin + len(ArmadaTypes.hull_zones)

        # TODO The selection slots are not guaranteed to make any sense, especially from an
        # untrained neural network. There should be some sanity checks here.
        return None, None

        # No token
        if (1.0 not in action[ttype_begin:tcolor_begin].tolist() or
            1.0 not in action[tcolor_begin:die_begin].tolist()):
            return None, None

        token = ArmadaTypes.defense_tokens[action[ttype_begin:tcolor_begin].tolist().index(1.0)]
        color = ArmadaTypes.token_colors[action[tcolor_begin:die_begin].tolist().index(1.0)]
        if "evade" == token:
            die_idx = action[die_begin:hull_begin].tolist().index(1.0)
            return color + " " + token, die_idx

        # TODO This only supports redirecting to a single hull zone currently
        if "redirect" == token:
            target_hull = ArmadaTypes.hull_zones[action[hull_begin:redirect_amount].tolist().index(1.0)]
            amount = action[redirect_amount].item()
            return color + " " + token, (target_hull, amount)

        # Other defense tokens with no targets
        return color + " " + token, None
