#
# Copyright Bernhard Firner, 2019-2020
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
                "ship phase - attack - resolve attack effects": self.resolveAttackEffects,
                "ship phase - attack - spend defense tokens": self.spendDefenseTokens
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
        assert world_state.sub_phase == "attack - resolve attack effects"

        if None == self.model:
            # Return no action
            return None
        # Encode the state, forward through the network, decode the result, and return the result.
        # TODO
        return None

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
        # TODO FIXME The return type is totally messed up, a list of actions should be allowable
        # from here
        # We only handle one sub-phase in this function
        assert world_state.sub_phase == "attack - spend defense tokens"

        if None == self.model:
            # Return no action
            return []
        # Encode the state, forward through the network, decode the result, and return the result.
        as_enc, die_slots = Encodings.encodeAttackState(world_state)
        as_enc = as_enc.to(self.device)
        if not self.model.with_novelty:
            # Forward through the policy net randomly, otherwise return random actions
            if self.randprob >= random.random():
                # Take a random action
                action = self.random_agent("def_tokens", as_enc)[0]
            else:
                action = self.model.forward("def_tokens", as_enc)[0]
            # Remember this state action pair if in memory mode
            if self.remembering:
                self.memory.append((world_state.attack, as_enc, action))
        else:
            action, novelty = self.model.forward("def_tokens", as_enc)
            # Remove the batch dimension
            action = action[0]
            novelty = novelty[0]
            # Remember this state action pair if in memory mode
            if self.remembering:
                self.memory.append((world_state.attack, as_enc, action, novelty))
        # Don't return the lifetime prediction (used in train_defense_tokens.py)
        #with torch.no_grad():
        #    action = torch.round(action[:Encodings.calculateSpendDefenseTokensSize()])
        # Determine which tokens should be spent. Spend any above a threshold.
        with torch.no_grad():
            spend_green = action[:len(ArmadaTypes.defense_tokens)] > math.log(0.5)
            spend_red = action[len(ArmadaTypes.defense_tokens):2*len(ArmadaTypes.defense_tokens)] > math.log(0.5)
            spent_tokens = spend_green + spend_red
            # Return now if no token will be spent
            if (0 == spent_tokens).all():
                return []

        # Handle the tokens
        green = ArmadaTypes.token_colors.index('green')
        red = ArmadaTypes.token_colors.index('red')

        # First check for validity. If the selected token isn't valid then use no token.
        # TODO Perhaps this should be penalized in some way
        green_idx, green_len = Ship.get_index('defense_tokens_green')
        red_idx, red_len = Ship.get_index('defense_tokens_red')
        defender_green_tokens = world_state.attack.defender.encoding[green_idx:green_idx + green_len]
        defender_red_tokens = world_state.attack.defender.encoding[red_idx:red_idx + red_len]
        if (spend_green > defender_green_tokens).any() or (spend_red > defender_red_tokens).any():
            return []

        # Verify that these tokens have not been the target of an accuracy and that they can be
        # spent
        for idx in len(ArmadaTypes.defense_tokens):
            if 0 < world_state.attack.accuracy_tokens[idx] and 0 < spent_tokens[idx]:
                return []
        # TODO FIXME Those last two checks (for token availability and non-accuracy status) should
        # be enforced via a fixed input to the network that suppresses token outputs if they are not
        # available. This would make learning simpler.

        actions = []
        # Handle the token, decoding the returned action based upon the token type.
        evade_index = ArmadaTypes.defense_tokens.index("evade")
        if 0 < defender_green_tokens[evade_index].item() + defender_red_tokens[evade_index].item():
            begin = Encodings.getSpendDefenseTokensEvadeOffset()
            end = begin + Encodings.max_die_slots
            # Get the index of the maximum die response
            _, die_idx = action[begin:end].max(0)
            # Check for an invalid response from the agent
            # TODO Perhaps this should be penalized in some way
            # TODO FIXME This should also be supressed through die availability.
            # TODO FIXME Also handle extreme range with 2 die targets
            if len(die_slots) <= die_idx.item():
                pass
            else:
                color = green if 0 < defender_green_tokens[evade_index].item() else red
                src_die_slot = die_slots[die_idx.item()]
                actions.append(("evade", (src_die_slot)))

        # TODO This only supports redirecting to a single hull zone currently
        redir_index = ArmadaTypes.defense_tokens.index("redirect")
        if 0 < spent_tokens[ArmadaTypes.defense_tokens.index("redirect")]:
            begin = Encodings.getSpendDefenseTokensRedirectOffset()
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

            # Make sure there is actual redirection
            if None != redir_hull:
                color = green if 0 < defender_green_tokens[redir_index].item() else red
                actions.append(("redirect", (redir_hull, redir_amount)))

        # Other defense tokens with no targets
        for tindx, token_type in enumerate(ArmadaTypes.defense_tokens):
            if token_type not in ["evade", "redirect"]:
                if 0 < spent_tokens[tindx]:
                    color = green if 0 < defender_green_tokens[tindx].item() else red
                    actions.append((token_type, (color, None)))

        return actions
