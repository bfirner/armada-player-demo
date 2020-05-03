#
# Copyright Bernhard Firner, 2020
#
import ship
from dice import ArmadaDice
from base_agent import BaseAgent 
from game_constants import (ArmadaPhases, ArmadaTypes)
from utility import (token_index, greenest_token_index, greenest_token, max_damage_index, face_index)

import random


class RandomAgent(BaseAgent):

    def __init__(self):
        """Initialize the agent with a couple of state handlers."""
        handler = {
                "ship phase - attack - resolve attack effects": self.resolveAttackEffects,
                "ship phase - attack - spend defense tokens": self.spendDefenseTokens
        }
        super(RandomAgent, self).__init__(handler)

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
                              icons so the only return type is the tuple:
                                ("accuracy", [die index, defender token index]).
                              In the future we may support other tuples such as:
                                ("add", colors), ("reroll", indices), or ("remove", indices)
        """
        # We only handle one sub-phase in this function
        assert world_state.full_phase == "ship phase - attack - resolve attack effects"

        attack = world_state.attack
        defender = attack.defender
        pool_colors = attack.pool_colors
        pool_faces = attack.pool_faces
        spent_tokens = attack.defender.spent_tokens

        # Currently we aren't checking for any game modifiers from cards or objectives so the only
        # possible choices from the agent during the resolve attack effects phase are to spend
        # accuracy icons to lock down defense tokens.

        # Since we are only dealing with spending accuracies at this time return here if there are none.
        accuracies = face_index("accuracy", pool_faces)
        if 0 == len(accuracies):
            return None
        # Keep track of which tokens are locked with accuracies
        targets = []
        # Don't attempt to accuracy more things that what exists.
        if len(accuracies) > len(attack.defender.defense_tokens):
            accuracies = accuracies[:len(attack.defender.defense_tokens)]
        for acc in accuracies:
            # Randomly choose to use the accuracy or not. It is okay if this is a bit biased towards
            # using the accuracy so we will just choose randomly one of the existing tokens or no
            # token.
            target = random.randint(0, len(attack.defender.defense_tokens) - len(targets))
            if target < (len(attack.defender.defense_tokens) - len(targets)):
                # Avoid targetting the same die more than once. It is cancelled by the accuracy so it is
                # not possible to target multiple times.
                while 0 < len([t for t in targets if t[1] == target]):
                    target += 1
                targets.append((acc, target))
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
        spent_tokens = attack.defender.spent_tokens
        accuracy_tokens = attack.accuracy_tokens

        # Very random agent. We will just go through a few random actions but won't do anything
        # incredibly stupid.

        # Maybe if there is no damage we shouldn't do anything, but we could always be trying to
        # discard tokens for some reason. The agent should at least learn to not always spend
        # tokens.

        # Let's skip spending more tokens if we have already decided to scatter though or if there
        # are no dice in the pool.
        if attack.token_type_spent('scatter') or 0 == len(attack.pool_faces):
            return (None, None)

        # Randomly pick tokens to use from the token types available that have not been targetted
        # with an accuracy.
        for tindx in range(len(defender.defense_tokens)):
            ttype = defender.token_type(tindx)
            if (not defender.spent_tokens[tindx] and not accuracy_tokens[tindx] and
                not attack.token_type_spent(ttype) and 0 == random.randint(0, 1)):
                if ttype in ["brace", "scatter", "contain", "salvo"]:
                    return (ttype, (tindx, None))
                else:
                    # Provide targets for the token
                    if 'evade' == ttype and attack.range != 'short':
                        # Choose one random dice, or two at extreme range.
                        if 1 == len(attack.pool_faces):
                            dice = 0
                        else:
                            dice = random.randint(0, len(attack.pool_faces) - 1)
                        if attack.range == 'extreme' and 1 < len(attack.pool_faces):
                            # Select a random number that is not the last one possible. If we
                            # selected the same number as before then it could not be the last
                            # number. Switch the die to the last number.
                            dice2 = random.randint(0, len(attack.pool_faces) - 2)
                            if dice2 == dice:
                                dice2 = len(attack.pool_faces) - 1
                            return (ttype, (tindx, dice, dice2))
                        else:
                            return (ttype, (tindx, dice))
                    elif 'redirect' == ttype:
                        # Choose an adjacent hull zone to suffer damage and redirect a random
                        # amount.
                        # Select an adjacent hull zone to the current one.
                        hull = defender.adjacent_zones(attack.defending_hull)[random.randint(0, 1)]
                        # Redirection must also specify the amount of damage to send to
                        # each hull zone so return (ttype, (tindx, (hull, damage)))
                        damage = random.randint(0, ArmadaDice.pool_damage(pool_faces))
                        return (ttype, (tindx, (hull, damage)))
                        # TODO Advanced projectors or the foresight title allow redirection to
                        # multiple hull zones.

        # If no token was selected then return no action.
        return (None, None)

