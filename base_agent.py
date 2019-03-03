#! /usr/bin/python3

# The agent must choose which actions to take. An agent takes the form of:
#     (world state, current step, active player) -> action
#
# There are quite a few steps in Armada.

# World states should contain a dice pool and a list of ships.
# TODO make a world state class

import ship
from dice import ArmadaDice

phases = ["spend defense tokens"]

# This agent deals with the "spend defense tokens" step.
def spendDefenseTokens(world_state, current_step, active_player):
    """
    Args:
        world_state (table)   : Contains the list of ships and dice pool.
        current_step (string) : This function only operates on the "spend defense tokens" step.
        active_player (int)   : This function only works if there are two ships.
                                The ship belonging to the non-active player will spend tokens.
    Returns:
        A modified world state. The defending ship's defense token state is
        modified, the dice pool may be updated (for example by an evasion
        token), and damage is assigned. The attacker may choose a critical
        effect after this step.
    """
    # We only handle one state in this function
    assert current_step == "spend defense tokens"

    attack = world_state['attack']
    defender = attack['defender']
    pool_colors = attack['pool_colors']
    pool_faces = attack['pool_faces']

    # Very simple agent. We will just go through a few simple rules.
    # First, no damage means don't do anything
    if 0 == ArmadaDice.pool_damage(pool_faces):
        return world_state
    def token_index(token, defender):
        return [idx for idx in range(len(defender.defense_tokens)) if token in defender.defense_tokens[idx]]

    def spend(idx, defender):
        tokens = defender.defense_tokens
        if 'green' in tokens[idx]:
            defender.defense_tokens[idx] = tokens[idx].replace('green', 'red')
        else:
            # Must be red, discard it
            defender.defense_tokens = tokens[0:idx] + tokens[idx+1:]

    # Scatter has highest priority
    scatters = token_index('scatter', defender)
    evades = token_index('evade', defender)
    if 0 < len(scatters):
        # Spend the token
        spend(scatters[0], defender)
        # Remove the dice pool
        pool_colors = []
        pool_faces = []
    elif 0 < len(evades):
        # If the range is long we can evade the die with the largest damage.
        # The best action is actually more complicated because removing a die may
        # not be necessary if we will brace and the current damage is an even
        # number. However, it may still be useful in that case to remove a critical
        # face. We will leave handling things like that to a smarter system.
        if 'long' == attack['range'] or 'medium' == attack['range']:
            # Find the largest damage die and cancel it. Ignoring crits for simplicity.
            # Negate the damage to get the lists in reverse order
            sorted_pool = sorted(zip(pool_faces, pool_colors), key=lambda face_color : -ArmadaDice.face_to_damage[face_color[0]])
            sorted_faces, sorted_colors = zip(*sorted_pool)
            sorted_faces = list(sorted_faces)
            sorted_colors = list(sorted_colors)

            # Cancel or reroll the first. Rerolling a black die may not be a good
            # idea, but they normally wouldn't be at medium range either.
            if 'long' == attack['range']:
                pool_colors = sorted_colors[1:]
                pool_faces = sorted_faces[1:]
            else:
                # Reroll at medium range
                sorted_faces[0] = ArmadaDice.random_roll(sorted_colors[0])
                pool_colors = sorted_colors
                pool_faces = sorted_faces
            # Mark the token as spent
            spend(evades[0], defender)

    # Brace if damage > 1
    if 1 < ArmadaDice.pool_damage(pool_faces):
        # TODO
        pass

    # Redirect to preserve shields
    if 0 < ArmadaDice.pool_damage(pool_faces):
        # TODO Now handle redirect
        pass

    # Return the modified state
    world_state['attack']['pool_colors'] = pool_colors
    world_state['attack']['pool_faces'] = pool_faces
    return world_state

