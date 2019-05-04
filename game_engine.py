# Game engine for Armada. Currently only partially supports the attack step.

import logging

from dice import ArmadaDice
from base_agent import (resolveAttackEffects, spendDefenseTokens)


# TODO This function should eventually part of a larger system to handle game logic.
def handleAttack(world_state, attacker, defender, attack_range):
    """This function handles the attack sub-phase of the ship phase.

    Args:
        world_state (table) : Contains the list of ships, objects, etc that comprise the current game.
        attacker (Tuple(Ship, hull zone)) : A tuple of the attacking ship and the string that denotes a hull zone.
        defender (Tuple(Ship, hull zone)) : A tuple of the defending ship and the string that denotes a hull zone.
        attack_range (str) : "long", "medium", or "short". TODO Remove this once ships have locations.
    Returns:
        world_state (table) : The world state after the attack completes.
    """
    attack_ship, attack_hull = attacker[0], attacker[1]
    pool_colors, pool_faces = attack_ship.roll(attack_hull, attack_range)
    world_state['attack'] = {
        'range': attack_range,
        'defender': defender[0],
        'defending zone': defender[1],
        'pool_colors': pool_colors,
        'pool_faces': pool_faces,
        # Keep track of which tokens are spent.
        # Tokens cannot be spent multiple times in a single attack.
        'spent_tokens': {},
        # A token targeted with an accuracy cannot be spent.
        'accuracy_tokens': []
    }
    spent_tokens = world_state['attack']['spent_tokens']
    acc_tokens = world_state['attack']['accuracy_tokens']
    redirect_hull = None
    redirect_amount = None

    # Handle defense tokens while valid ones are spent
    def token_index(token, defender):
        return [idx for idx in range(len(defender.defense_tokens)) if token in defender.defense_tokens[idx]]

    def spend(idx, defender):
        tokens = defender.defense_tokens
        name = tokens[idx]
        if 'green' in tokens[idx]:
            defender.defense_tokens[idx] = tokens[idx].replace('green', 'red')
        else:
            # Token is red if it is not green, discard it
            defender.defense_tokens = tokens[0:idx] + tokens[idx+1:]
        return name

    def get_token_type(index, defender):
        if "red" in defender.defense_tokens[index]:
            return defender.defense_tokens[index][len("red "):]
        else:
            return defender.defense_tokens[index][len("green "):]


    # TODO Roll the phase into the world state
    attack_effect_targets = resolveAttackEffects(world_state, ("ship phase", "attack - resolve attack effects"))

    spent_dice = []
    for effect_tuple in attack_effect_targets:
        action = effect_tuple[0]
        if "accuracy" == action:
            die_index, token_index = effect_tuple[1], effect_tuple[2]
            # Mark this die as spent, they will be removed from the pool after we handle operations
            # that require the indexes to stay the same
            spent_dice.append(die_index)
            # An token targeted with an accuracy cannot be spent normally
            acc_tokens.append(token_index)
            token_type = get_token_type(token_index, defender[0])
    # Remove the spent dice
    for index in sorted(spent_dice, reverse=True):
        del pool_faces[index]
        del pool_colors[index]


    token, token_targets = spendDefenseTokens(world_state, ("ship phase", "attack - spend defense tokens"))
    while None != token and token < len(defender[0].defense_tokens):
        # Spend the token and resolve the effect
        token_type = spend(token, defender[0])
        # TODO If a token has already been spent it cannot be spent again, we should enforce that here.
        if "brace" in token_type:
            spent_tokens['brace'] = True
        elif "scatter" in token_type:
            spent_tokens['scatter'] = True
        elif "contain" in token_type:
            spent_tokens['contain'] = True
        elif "redirect" in token_type:
            spent_tokens['redirect'] = True
            redirect_hull, redirect_amount = token_targets
        elif "evade" in token_type:
            # Need to evade a specific die
            die_index = token_targets
            if die_index >= len(pool_colors):
                print("Warning: could not find specified evade token target.")
            else:
                spent_tokens['evade'] = True
                if 'long' == attack_range:
                    # Remove the die
                    pool_colors = pool_colors[:die_index] + pool_colors[die_index+1:]
                    pool_faces = pool_faces[:die_index] + pool_faces[die_index+1:]
                elif 'medium' == attack_range:
                    # Reroll the die
                    pool_faces[die_index] = ArmadaDice.random_roll(pool_colors[die_index])
                world_state['attack']['pool_colors'] = pool_colors
                world_state['attack']['pool_faces'] = pool_faces
        # See if the agent will spend another defense token
        token, token_targets = spendDefenseTokens(world_state, ("ship phase", "attack - spend defense tokens"))

    # TODO The defender may have ways of modifying the attack pool at this point

    # Now calculate the damage done to the defender
    # Apply effect on the entire attack at this point
    damage = ArmadaDice.pool_damage(pool_faces)
    if 'scatter' in spent_tokens:
        damage = 0

    if 'brace' in spent_tokens:
        damage = damage // 2 + damage % 2

    if 'redirect' in spent_tokens:
        # Redirect to the target hull zone, but don't redirect more damage than is present
        redirect_amount = min(damage, redirect_amount)
        defender[0].damage(redirect_hull, redirect_amount)
        damage = damage - redirect_amount
        world_state['attack']['{} damage'.format(redirect_hull)] = redirect_amount

    # Deal remaining damage to the defender
    world_state['attack']['{} damage'.format(defender[1])] = damage
    defender[0].damage(defender[1], damage)

    # TODO Log the world state into a game log
    logging.info(world_state)

    # Clear the attack-only states and return the world state with modified ship status
    world_state['attack'] = {}
    return world_state
