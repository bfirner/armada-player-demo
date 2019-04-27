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

# Helper functions. Ideally these would be private functions

# Array of matching token indexes
def token_index(token, defender):
    return [idx for idx in range(len(defender.defense_tokens)) if token in defender.defense_tokens[idx]]

# Array of matching token indexes with greenest tokens first
def greenest_token_index(token, defender):
    greens = []
    reds = []
    for idx, name in enumerate(defender.defense_tokens):
        if "green {}".format(token) == name:
            greens.append(idx)
        elif "red {}".format(token) == name:
            reds.append(idx)
    return greens + reds

# The first, greenest token of the given type.
def greenest_token(name, defender):
    green_tokens = token_index('green {}'.format(name), defender)
    red_tokens = token_index('red {}'.format(name), defender)
    if green_tokens or red_tokens:
        return green_tokens[0] if green_tokens else red_tokens[0]
    else:
        return None

def max_damage_index(pool_faces):
    for face in ["hit_crit", "hit_hit", "crit", "hit"]:
        if face in pool_faces:
            return pool_faces.index(face)
    return None

def face_index(face, pool_faces):
    return [idx for idx in range(len(pool_faces)) if face in pool_faces[idx]]

# This agent deals with the "resolve attack effects" step.
def resolveAttackEffects(world_state, current_step):
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
    assert current_step[1] == "attack - resolve attack effects"

    attack = world_state['attack']
    defender = attack['defender']
    pool_colors = attack['pool_colors']
    pool_faces = attack['pool_faces']
    spent_tokens = attack['spent_tokens']

    # Very simple agent. We will just go through a few simple rules.

    # First, no damage means don't do anything
    damage = ArmadaDice.pool_damage(pool_faces)
    if 0 == damage:
        return []

    # Since we are only dealing with spending accuracies at this time return here if there are none.
    accuracies = face_index("accuracy", pool_faces)
    if not accuracies:
        return []
    # Keep track of which accuracy we will spend next
    acc_index = 0

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

    evades = greenest_token_index("evade", defender)
    braces = greenest_token_index("brace", defender)
    scatters = greenest_token_index("scatter", defender)
    # Skipping redirects and contains

    targets = []
    # Always target scatter if they are present.
    if scatters:
        for idx in scatters:
            if acc_index < len(accuracies):
                targets.append(("accuracy", accuracies[acc_index], idx))
                acc_index += 1
    # Exit if there is nothing else to spend
    if len(accuracies) == acc_index:
        return targets

    # 1) If this is at short range then only worry about braces (not worrying about effects like
    # Mon-Mothma for the simple agent)
    # 2) If there are enough accuracies to cover everything else then just cover them
    # 3) Otherwise try to cover the more effective token, either braces first or evades first
    if 'short' == attack['range']:
        for idx in braces:
            if acc_index < len(accuracies):
                targets.append(("accuracy", accuracies[acc_index], idx))
                acc_index += 1
    elif ((braces and not evades) or
          (len(accuracies) - acc_index >= (len(braces) + len(evades))) or
          brace_damage < evade_damage):
        for idx in braces:
            if acc_index < len(accuracies):
                targets.append(("accuracy", accuracies[acc_index], idx))
                acc_index += 1
        for idx in evades:
            if acc_index < len(accuracies):
                targets.append(("accuracy", accuracies[acc_index], idx))
                acc_index += 1
    else:
        for idx in evades:
            if acc_index < len(accuracies):
                targets.append(("accuracy", accuracies[acc_index], idx))
                acc_index += 1
        for idx in braces:
            if acc_index < len(accuracies):
                targets.append(("accuracy", accuracies[acc_index], idx))
                acc_index += 1

    # Return targets
    return targets

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
    if not 'evade' in spent_tokens and evade:
        # If the range is long we can evade the die with the largest damage.
        # The best action is actually more complicated because removing a die may
        # not be necessary if we will brace and the current damage is an even
        # number. However, it may still be useful in that case to remove a critical
        # face. We will leave handling things like that to a smarter system.
        if 'short' != attack['range']:
            return (evade, max_damage_index(pool_faces))
            # If we made it here then there were no dice worth cancelling or rerolling.

    # Brace if damage > 1
    brace = greenest_token("brace", defender)
    if not 'brace' in spent_tokens and brace and 1 < ArmadaDice.pool_damage(pool_faces):
        return (brace, None)

    # Redirect to preserve shields
    # Should really check adjacent shields and figure out what to redirect, but we will leave that
    # to a smarter agent.
    redirect = greenest_token("redirect", defender)
    if not 'redirect' in spent_tokens and redirect and 0 < ArmadaDice.pool_damage(pool_faces):
        # TODO Now handle redirect
        pass

    # Return a tuple of two Nones if we won't spend a token.
    return (None, None)

