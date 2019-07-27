#
# Copyright Bernhard Firner, 2019
#
import random

import ship
from armada_encodings import (Encodings)
from dice import ArmadaDice
from base_agent import BaseAgent 
from game_constants import (ArmadaPhases, ArmadaTypes)
from utility import token_index, adjacent_hulls

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
            (List[int])              : Token slot mapping (source to encoding index)
            (List[int])              : Die slot mapping (source to encoding index)
        """
        encoding = torch.zeros(1, self.attack_enc_size).to(self.device)

        # Now populate the tensor
        attack = world_state.attack
        defender = attack.defender
        attacker = attack.attacker

        # Hull zones will of course increase with huge ships
        # Future proof this code by having things basing math upon the length of this table
        hull_zones = ArmadaTypes.hull_zones

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
            encoding[0, cur_offset + offset] = 1 if attack.spent_types[offset] else 0
        cur_offset += len(ArmadaTypes.defense_tokens)

        # Hot encoding the defenders tokens
        # Each hot encoding is: [type - 5, color - 2, accuracy targeted - 1, spent - 1]
        token_slots = random.sample(range(ArmadaTypes.max_defense_tokens), len(defender.defense_tokens))
        for token_idx, slot in enumerate(token_slots):
            slot_offset = cur_offset + slot * Encodings.hot_token_size
            # Process if the defender has a token for this slot
            if token_idx < len(defender.defense_tokens):
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
                encoding[0, slot_offset] = 1 if attack.accuracy_tokens[token_idx] else 0
                # Encode if this particular token has been spent
                encoding[0, slot_offset + 1] = 1 if attack.spent_tokens[token_idx] else 0

        # Move the current encoding offset to the position after the token section
        cur_offset += Encodings.hot_token_size * ArmadaTypes.max_defense_tokens

        # Attack range
        for offset, arange in enumerate(ArmadaTypes.ranges):
            encoding[0, cur_offset + offset] = 1 if arange == attack.range else 0
        cur_offset += len(ArmadaTypes.ranges)

        # Each die will be represented with a hot_die_size vector
        # There are max_die_slots slots for these, and we will fill them in randomly
        # During training we want the model to react properly to a die in any location in the
        # vector, so we randomize the dice locations so that the entire vector is used (otherwise
        # the model would be poorly trained for rolls with a very large number of dice)
        die_slots = random.sample(range(Encodings.max_die_slots), len(attack.pool_colors))
        for die_idx, slot in enumerate(die_slots):
            slot_offset = cur_offset + slot * Encodings.hot_die_size
            # Encode die colors
            for offset, color in enumerate(ArmadaDice.die_colors):
                encoding[0, slot_offset + offset] = 1 if color == attack.pool_colors[die_idx] else 0
            slot_offset += len(ArmadaDice.die_colors)
            # Encode die faces
            for offset, face in enumerate([face for face in ArmadaDice.die_faces.keys()]):
                encoding[0, slot_offset + offset] = 1 if face == attack.pool_faces[die_idx] else 0

        # Sanity check on the encoding size and the data put into it
        assert encoding.size(1) == cur_offset + Encodings.hot_die_size * Encodings.max_die_slots

        return encoding, token_slots, die_slots

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
        as_enc, token_slots, die_slots = self.encodeAttackState(world_state)
        action = self.model.forward("def_tokens", as_enc)[0]
        # Remember this state action pair if in memory mode
        if self.remembering:
            self.memory.append((as_enc, action))
        # Don't return the lifetime prediction (used in train_defense_tokens.py)
        #with torch.no_grad():
        #    action = torch.round(action[:Encodings.calculateSpendDefenseTokensSize()])
        # Determine which token should be spent. Choose the max token or no token.
        _, token_idx = action[:len(ArmadaTypes.defense_tokens) + 1].max(0)
        # Return now if no token will be spent
        if token_idx.item() == len(ArmadaTypes.defense_tokens):
            return None, None

        # Check for an invalid response from the agent
        # TODO Perhaps this should be penalized in some way
        if len(token_slots) <= token_idx.item():
            return None, None

        # Check the token in the input encoding.
        src_token_idx = token_slots[token_idx.item()]
        # Handle the token
        # First check for validity. If the selected token isn't valid then use no token.
        if src_token_idx >= len(world_state.attack.defender.defense_tokens):
            return None, None

        # Verify that this token has not be the target of an accuracy and that it can be spent
        if src_token_idx in world_state.attack.accuracy_tokens:
            return None, None

        src_token = world_state.attack.defender.defense_tokens[src_token_idx]
        tcolor, ttype = tuple(src_token.split(' '))

        # Handle the token, decoding the returned action based upon the token type.
        if "evade" == ttype:
            begin = len(ArmadaTypes.defense_tokens) + 1
            end = begin + Encodings.max_die_slots
            _, die_idx = action[begin:end].max(0)
            # Check for an invalid response from the agent
            # TODO Perhaps this should be penalized in some way
            if len(die_slots) <= die_idx.item():
                return None, None
            src_die_slot = die_slots[die_idx.item()]
            return src_token_idx, src_die_slot

        # TODO This only supports redirecting to a single hull zone currently
        if "redirect" == ttype:
            begin = len(ArmadaTypes.defense_tokens) + 1 + Encodings.max_die_slots
            end = begin + len(ArmadaTypes.hull_zones)
            # The encoding has a value for each hull zone. We should check if an upgrade allows the
            # defender to redirect to nonadjacent or multiple hull zones, but for now we will just
            # handle the base case. TODO
            adj_hulls = adjacent_hulls(world_state.attack.defending_hull)

            # Redirect to whichever hull has the greater value TODO
            redir_hull = None
            redir_amount = 0

            for hull in adj_hulls:
                hull_redir_amount = round(action[begin + ArmadaTypes.hull_zones.index(hull)].item())
                if hull_redir_amount > redir_amount:
                    redir_hull = hull
                    redir_amount = hull_redir_amount

            # Agent is asking to redirect nothing
            if None == redir_hull:
                return None, None
            return src_token_idx, (redir_hull, redir_amount)

        # Other defense tokens with no targets
        return src_token_idx, None
