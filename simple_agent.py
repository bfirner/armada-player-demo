#
# Copyright Bernhard Firner, 2019-2020
#
import ship
from dice import ArmadaDice
from base_agent import BaseAgent 
from game_constants import (ArmadaPhases, ArmadaTypes)
from utility import (token_index, tokens_available, max_damage_index, face_index)


class SimpleAgent(BaseAgent):

    def __init__(self):
        """Initialize the simple agent with a couple of simple state handlers."""
        handler = {
                "ship phase - attack - resolve attack effects": self.resolveAttackEffects,
                "ship phase - attack - spend defense tokens": self.spendDefenseTokens
        }
        super(SimpleAgent, self).__init__(handler)

    @staticmethod
    def accuracy_token(accuracies, acc_index, targets, tokens, token_type):
        """Use accuracy dice on all of the specified tokens.

        Arguments:
            accuracies   (List[int]): List of die faces with an accuracy icon.
            acc_index          (int): Current position in the accuracies list.
            targets    (List[tuple]): Targets to return from the agent to the engine.
            tokens (tuple(int, int)): Number of green and red tokens of the chosen type.
            token_type         (int): Index (in ArmadaTypes.defense_tokens) of the token type.
        Returns:
            int: The updated acc_index
        """
        target_idx = 0
        while acc_index < len(accuracies) and target_idx < tokens[0]:
            targets.append((accuracies[acc_index], token_type, ArmadaTypes.green))
            acc_index += 1
            target_idx += 1
        target_idx = 0
        while acc_index < len(accuracies) and target_idx < tokens[1]:
            targets.append((accuracies[acc_index], token_type, ArmadaTypes.red))
            acc_index += 1
            target_idx += 1
        return acc_index

    # This agent deals with the "resolve attack effects" step.
    def resolveAttackEffects(self, world_state):
        """
        Args:
            world_state (table)   : Contains the list of ships and dice pool.
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
        assert world_state.full_phase == "ship phase - attack - resolve attack effects"

        attack = world_state.attack
        defender = attack.defender
        pool_colors = attack.pool_colors
        pool_faces = attack.pool_faces
        spent_types = world_state.attack.spent_types

        # Very simple agent. We will just go through a few simple rules.

        # First, no damage means don't do anything
        damage = ArmadaDice.pool_damage(pool_faces)
        if 0 == damage:
            return None

        # Since we are only dealing with spending accuracies at this time return here if there are none.
        accuracies = face_index("accuracy", pool_faces)
        if 0 == len(accuracies):
            return None

        # Accuracy to maximize damage. This could be nonoptimal in some instances (for example if we
        # target a brace but the damage is more than sufficient to destroy the defender on the defending
        # hull zone but a redirect can save the defender because shields remain on an adjacent zone).
        # Since this is just an example agent this is okay.

        # If at long range and the defender has an evade and no brace accuracy the evade. Don't worry
        # about double evades, this is a basic agent.
        # If the defender has a brace and an evade target the evade if a single die has the same damage
        # reduction as the brace.
        # If there are multiple accuracies then target multiple things.
        brace_damage = damage // 2 + damage % 2
        max_index = max_damage_index(pool_faces)
        max_damage = ArmadaDice.pool_damage(pool_faces[max_index:max_index])
        # TODO If the range is medium then calculate the expected value of the reroll for the
        # calculation that follows
        evade_damage = damage - max_damage

        evades = tokens_available("evade", defender)
        braces = tokens_available("brace", defender)
        scatters = tokens_available("scatter", defender)
        # Skipping redirects and contains

        targets = []
        # Always target scatter if they are present.
        acc_index = self.accuracy_token(accuracies=accuracies, acc_index=0, targets=targets,
                                        tokens=scatters,
                                        token_type=ArmadaTypes.defense_tokens.index('scatter'))
        # Exit if there is nothing else to spend
        if len(accuracies) == acc_index:
            return ("accuracy", targets)

        # 1) If this is at short range then only worry about braces (not worrying about effects like
        # Mon-Mothma for the simple agent)
        # 2) If there are enough accuracies to cover everything else then just cover them
        # 3) Otherwise try to cover the more effective token, either braces first or evades first
        if 'short' == attack.range:
            acc_index = self.accuracy_token(accuracies=accuracies, acc_index=acc_index,
                                            targets=targets, tokens=braces,
                                            token_type=ArmadaTypes.defense_tokens.index('brace'))
        elif ((0 < braces[0] + braces[1] and 0 == evades[0] + evades[1]) or
              (len(accuracies) - acc_index >= (braces[0] + braces[1] + evades[0] + evades[1])) or
              brace_damage < evade_damage):
            acc_index = self.accuracy_token(accuracies=accuracies, acc_index=acc_index,
                                            targets=targets, tokens=braces,
                                            token_type=ArmadaTypes.defense_tokens.index('brace'))
            acc_index = self.accuracy_token(accuracies=accuracies, acc_index=acc_index,
                                            targets=targets, tokens=evades,
                                            token_type=ArmadaTypes.defense_tokens.index('evade'))
        else:
            acc_index = self.accuracy_token(accuracies=accuracies, acc_index=acc_index,
                                            targets=targets, tokens=evades,
                                            token_type=ArmadaTypes.defense_tokens.index('evade'))
            acc_index = self.accuracy_token(accuracies=accuracies, acc_index=acc_index,
                                            targets=targets, tokens=braces,
                                            token_type=ArmadaTypes.defense_tokens.index('brace'))

        # Return targets
        if 0 < len(targets):
            return ("accuracy", targets)
        else:
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
        # We only handle one sub-phase in this function
        assert world_state.full_phase == "ship phase - attack - spend defense tokens"

        attack = world_state.attack
        defender = attack.defender
        pool_colors = attack.pool_colors
        pool_faces = attack.pool_faces
        spent_types = world_state.attack.spent_types
        accuracy_tokens = attack.accuracy_tokens

        # Very simple agent. We will just go through a few simple rules.

        # First, no damage means don't do anything
        if 0 == ArmadaDice.pool_damage(pool_faces):
            return []

        # No need to spend any more tokens if we have already scattered
        if attack.token_type_spent('scatter'):
            return []

        # Scatter has highest priority. Note that it may be smarter to evade an
        # attack with only one damage die so this isn't the smartest agent, but
        # this is just a basic agent so this is okay.
        scatter = tokens_available("scatter", defender, accuracy_tokens)
        if not attack.token_type_spent('scatter'):
            for color in range(len(ArmadaTypes.token_colors)):
                if 0 < scatter[color]:
                    return [('scatter', (color, None))]

        evade = tokens_available("evade", defender, accuracy_tokens)
        if not attack.token_type_spent('evade') and 'short' != attack.range:
            # If the range is long we can evade the die with the largest damage.
            # The best action is actually more complicated because removing a die may
            # not be necessary if we will brace and the current damage is an even
            # number. However, it may still be useful in that case to remove a critical
            # face. We will leave handling things like that to a smarter system.
            for color in range(len(ArmadaTypes.token_colors)):
                if 0 < evade[color]:
                    return [('evade', (color, max_damage_index(pool_faces)))]

        # Brace if damage > 1
        brace = tokens_available("brace", defender, accuracy_tokens)
        if not attack.token_type_spent('brace') and 1 < ArmadaDice.pool_damage(pool_faces):
            for color in range(len(ArmadaTypes.token_colors)):
                if 0 < brace[color]:
                    return [('brace', (color, None))]

        # Redirect to preserve shields
        # Should really check adjacent shields and figure out what to redirect, but we will leave that
        # to a smarter agent.
        redirect = tokens_available("redirect", defender, accuracy_tokens)
        if not attack.token_type_spent('redirect') and 0 < ArmadaDice.pool_damage(pool_faces):
            for color in range(len(ArmadaTypes.token_colors)):
                if 0 < redirect[color]:
                    # TODO Now handle redirect
                    pass

        # Return a tuple of two Nones if we won't spend a token.
        return []

