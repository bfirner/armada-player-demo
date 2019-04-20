#! /usr/bin/python3

# The agent must choose which actions to take. An agent takes the form of:
#     (world state, current step, active player) -> action
#
# There are quite a few steps in Armada.

# World states should contain a dice pool and a list of ships.
# TODO make a world state class

import ship
from dice import ArmadaDice
from game_constants import ArmadaPhases

# This agent deals with the "spend defense tokens" step.
def spendDefenseTokens(world_state, current_step):
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
    assert current_step[1] == "attack - spend defense tokens"

    attack = world_state['attack']
    defender = attack['defender']
    pool_colors = attack['pool_colors']
    pool_faces = attack['pool_faces']
    spent_tokens = attack['spent_tokens']

    # Very simple agent. We will just go through a few simple rules.

    # Helper functions
    def token_index(token, defender):
        return [idx for idx in range(len(defender.defense_tokens)) if token in defender.defense_tokens[idx]]

    def greenest_token(name, defender):
        green_tokens = token_index('green {}'.format(name), defender)
        red_tokens = token_index('red {}'.format(name), defender)
        if green_tokens or red_tokens:
            return green_tokens[0] if green_tokens else red_tokens[0]
        else:
            return None

    # First, no damage means don't do anything
    if 0 == ArmadaDice.pool_damage(pool_faces):
        return None, None

    # No need to spend any more tokens if we have already scattered
    if 'scatter' in spent_tokens:
        return (None, None)

    # Scatter has highest priority. Note that it may be smarter to evade an
    # attack with only one damage die so this isn't the smartest agent, but
    # this is just a basic agent so this is okay.
    scatter = greenest_token("scatter", defender)
    if scatter:
        return (scatter, None)

    evade = greenest_token("evade", defender)
    if evade:
        # If the range is long we can evade the die with the largest damage.
        # The best action is actually more complicated because removing a die may
        # not be necessary if we will brace and the current damage is an even
        # number. However, it may still be useful in that case to remove a critical
        # face. We will leave handling things like that to a smarter system.
        if 'short' != attack['range']:
            for face in ["hit_crit", "hit_hit", "crit", "hit"]:
                if face in pool_faces:
                    return (evade, pool_faces.index(face))
            # If we made it here then there were no dice worth cancelling or rerolling.

    # Brace if damage > 1
    brace = greenest_token("brace", defender)
    if brace and 1 < ArmadaDice.pool_damage(pool_faces):
        return (brace, None)

    # Redirect to preserve shields
    # Should really check adjacent shields and figure out what to redirect, but we will leave that
    # to a smarter agent.
    redirect = greenest_token("redirect", defender)
    if redirect and 0 < ArmadaDice.pool_damage(pool_faces):
        # TODO Now handle redirect
        pass

    # Return a tuple of two Nones if we won't spend a token.
    return (None, None)

