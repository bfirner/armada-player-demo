#
# Copyright Bernhard Firner, 2020
#
import torch
import ship
from dice import ArmadaDice
from base_agent import BaseAgent 
from game_constants import (ArmadaPhases, ArmadaTypes)
from utility import (max_damage_index, face_index)

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

        # Currently we aren't checking for any game modifiers from cards or objectives so the only
        # possible choices from the agent during the resolve attack effects phase are to spend
        # accuracy icons to lock down defense tokens.

        # Since we are only dealing with spending accuracies at this time return here if there are none.
        accuracies = face_index("accuracy", pool_faces)
        if 0 == len(accuracies):
            return None
        # Keep track of which tokens are locked with accuracies
        gidx, glen = defender.get_index('green_defense_tokens')
        ridx, rlen = defender.get_index('red_defense_tokens')
        green_tokens = defender.encoding[gidx:gidx + glen]
        red_tokens = defender.encoding[ridx:ridx + rlen]
        with torch.no_grad():
            total_tokens = int(green_tokens.sum().item() + red_tokens.sum().item())
        # Make a counter to make the random selection process simpler
        # Begin with a placeholder 0 here to simplify the loops
        cumulative_tokens = [0]
        for i in range(green_tokens.size(0)):
            cumulative_tokens.append(cumulative_tokens[-1] + green_tokens[i].item())
        for i in range(red_tokens.size(0)):
            cumulative_tokens.append(cumulative_tokens[-1] + red_tokens[i].item())
        # Chop off the placeholder 0
        cumulative_tokens = cumulative_tokens[1:]
        # Don't attempt to accuracy more things that what exists.
        num_acc_to_use = random.randint(0, min(len(accuracies), total_tokens))
        # Shuffle the correct number of accuracies with 0s for non accuracied tokens
        selection = [1] * num_acc_to_use + [0] * (total_tokens - num_acc_to_use)
        random.shuffle(selection)
        # Loop through the tokens and assign the accuracies to the correct token types.
        targets = []
        acc_idx = 0
        for acc, select in enumerate(selection):
            if 1 == select:
                # Grab the token section that contains this index
                token_idx = next(i for i in range(glen + rlen) if cumulative_tokens[i] > acc)
                if token_idx < glen:
                    # Still in the green token section
                    targets.append((acc_idx,
                                    token_idx,
                                    ArmadaTypes.green))
                    acc_idx += 1
                else:
                    # Targetting a red token
                    targets.append((acc_idx,
                                    token_idx - glen,
                                    ArmadaTypes.red))
                    acc_idx += 1
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
        accuracy_tokens = attack.accuracy_tokens

        # Very random agent. We will just go through a few random actions but won't do anything
        # incredibly stupid.

        # Maybe if there is no damage we shouldn't do anything, but we could always be trying to
        # discard tokens for some reason. The agent should at least learn to not always spend
        # tokens.

        # Let's skip spending more tokens if we have already decided to scatter though or if there
        # are no dice in the pool.
        if attack.token_type_spent('scatter') or 0 == len(attack.pool_faces):
            return []

        # Randomly pick tokens to use from the token types available that have not been targetted
        # with an accuracy.
        # TODO FIXME Accuracy tokens change to selecting token types rather than specific tokens
        # TODO FIXME The index being returned only refers to spending green (0) or red (1)
        actions = []
        for tindx, ttype in enumerate(ArmadaTypes.defense_tokens):
            green_present, red_present = world_state.attack.defender.token_count(tindx)
            # Here we are going to assume that rules prevent anyone from using an accuracy on an
            # already spent token.
            total_green = green_present - attack.accuracy_tokens[tindx]
            total_red = red_present - attack.accuracy_tokens[tindx + len(ArmadaTypes.defense_tokens)]
            # Maybe spend this token if one is available and hasn't been spent by the defender
            if (0 < total_green + total_red and
                not attack.token_type_spent(ttype) and 0 == random.randint(0, 1)):
                # Choose which type of token to spend (red or green)
                spend_type = 0
                if total_green < random.randint(1, total_red + total_green):
                    spend_type = 1
                if ttype in ["brace", "scatter", "contain", "salvo"]:
                    actions.append((ttype, (spend_type, None)))
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
                            actions.append((ttype, (spend_type, dice, dice2)))
                        else:
                            actions.append((ttype, (spend_type, dice)))
                    elif 'redirect' == ttype:
                        # Choose an adjacent hull zone to suffer damage and redirect a random
                        # amount.
                        # Select an adjacent hull zone to the current one.
                        hull = defender.adjacent_zones(attack.defending_hull)[random.randint(0, 1)]
                        # Redirection must also specify the amount of damage to send to
                        # each hull zone so return (ttype, (spend_type, (hull, damage)))
                        damage = random.randint(0, ArmadaDice.pool_damage(pool_faces))
                        actions.append((ttype, (spend_type, [(hull, damage)])))
                        # TODO Advanced projectors or the foresight title allow redirection to
                        # multiple hull zones.
        return actions
