# Game engine for Armada. Currently only partially supports the attack step.

import logging

from dice import ArmadaDice
from game_constants import (ArmadaTypes)
from world_state import (AttackState, WorldState)


# TODO This function should eventually part of a larger system to handle game logic.
def handleAttack(world_state, attacker, defender, attack_range, offensive_agent, defensive_agent,
                 state_log=None):
    """This function handles the attack sub-phase of the ship phase.

    Args:
        world_state (WorldState) : Contains the list of ships, objects, etc that comprise the current game.
        attacker (Tuple(Ship, hull zone)) : A tuple of the attacking ship and the string that denotes a hull zone.
        defender (Tuple(Ship, hull zone)) : A tuple of the defending ship and the string that denotes a hull zone.
        attack_range (str) : "long", "medium", or "short". TODO Remove this once ships have locations.
        offensive_agent (BaseAgent) : An agent to handle actions on the offensive side.
        defensive_agent (BaseAgent) : An agent to handle actions on the defensive side.
        state_log (StateActionLog)  : A logger for game states and actions.
    Returns:
        world_state (table) : The world state after the attack completes.
    """
    world_state.setPhase("ship phase", "attack - declare")
    # Log if the log is present
    if state_log is not None:
        state_log.append(('state', world_state.clone()))

    world_state.setPhase("ship phase", "attack - roll attack dice")

    # TODO Effects that occur before the roll should happen here (obstruction, Sato, etc)
    # TODO FIXME Cloning the world state during logging is extremely slow. It would be better to
    # simply encode the world state here directly, but then users of the log need easy ways to
    # extract information from the encoding. The world state object itself should be modified to
    # make it easier to serialize and deserialize itself, or query from a serialized version of
    # itself.

    attack_ship, attack_hull = attacker[0], attacker[1]
    pool_colors, pool_faces = attack_ship.roll(attack_hull, attack_range)
    attack = AttackState(attack_range, attacker[0], attacker[1], defender[0], defender[1], pool_colors, pool_faces)
    world_state.updateAttack(attack)

    # Log if the log is present
    if state_log is not None:
        state_log.append(('state', world_state.clone()))

    spent_types = world_state.attack.spent_types
    defense_effects = world_state.attack.defense_effects
    acc_tokens = world_state.attack.accuracy_tokens
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
    world_state.setSubPhase("attack - resolve attack effects")

    # Log if the log is present
    if state_log is not None:
        state_log.append(('state', world_state.clone()))

    attack_effect_tuple = offensive_agent.handle(world_state = world_state)
    # Log if the log is present
    if state_log is not None:
        state_log.append(('action', attack_effect_tuple))

    # TODO FIXME The action and world state should be encoded and logged

    while attack_effect_tuple is not None:
        action = attack_effect_tuple[0]
        if "accuracy" == action:
            spent_dice = []
            for acc_action in attack_effect_tuple[1]:
                die_index, token_type, token_color = acc_action[0], acc_action[1], acc_action[2]
                # Mark this die as spent, it will be removed from the pool after we handle operations
                # that require the indexes to stay the same
                spent_dice.append(die_index)
                # A token type targeted with an accuracy cannot be spent normally
                attack.accuracy_defender_token(token_type, token_color)
            # Remove the spent dice
            for index in sorted(spent_dice, reverse=True):
                del pool_faces[index]
                del pool_colors[index]
        # TODO Handle more effects at some point
        # Get the next attack effect
        # TODO Maybe we should recreate the attack stack if there is internal logging.
        #attack = AttackState(attack_range, attacker[0], attacker[1], defender[0], defender[1], pool_colors, pool_faces)
        #attack.accuracy_tokens = acc_tokens
        #attack.spent_types = spent_types
        world_state.updateAttack(attack)
        # Log if the log is present
        if state_log is not None:
            state_log.append(('state', world_state.clone()))
        attack_effect_tuple = offensive_agent.handle(world_state)
        # Log if the log is present
        if state_log is not None:
            state_log.append(('action', attack_effect_tuple))
        # TODO FIXME The action and world state should be encoded and logged

    world_state.setSubPhase("attack - spend defense tokens")
    # Log if the log is present
    if state_log is not None:
        state_log.append(('state', world_state.clone()))

    # The defense agent returns: ("effect name", (args...))
    # For a standard defense token the first argument is the token index.
    effect_list = defensive_agent.handle(world_state)
    while 0 != len(effect_list):
        # Log if the log is present
        if state_log is not None:
            state_log.append(('action', effect_list))
        for effect, effect_args in effect_list:
            # TODO Only handling basic token effects for now
            if effect in ArmadaTypes.defense_tokens:
                token_type = effect
                color_type = effect_args[0]
                # Spend the token and resolve the effect
                # If a token has already been spent it cannot be spent again, we should enforce that here.
                if attack.token_type_spent(token_type):
                    raise RuntimeError("Cannot spend the same token twice during the spend defense tokens sub phase!")
                world_state.attack.defender_spend_token(token_type, color_type)
                # Only redirect and evade have additional targets
                if "redirect" == token_type:
                    redirects = effect_args[1]
                elif "evade" == token_type:
                    # Need to evade a specific die (or multiple if allowed by distance or by card effect)
                    for die_index in effect_args[1:]:
                        if die_index >= len(pool_colors):
                            print("Warning: could not find specified evade token target.")
                        else:
                            if 'long' == attack_range or 'extreme' == attack_range:
                                # Remove the die
                                pool_colors = pool_colors[:die_index] + pool_colors[die_index+1:]
                                pool_faces = pool_faces[:die_index] + pool_faces[die_index+1:]
                            elif 'medium' == attack_range:
                                # Reroll the die
                                pool_faces[die_index] = ArmadaDice.random_roll(pool_colors[die_index])
                            world_state.attack.pool_colors = pool_colors
                            world_state.attack.pool_faces = pool_faces
            else:
                raise RuntimeError("Effect {} is currently not handled.".format(effect))
            # Log the updated state if the log is present
            if state_log is not None:
                state_log.append(('state', world_state.clone()))
        # See if the agent will trigger another effect.
        effect_list = defensive_agent.handle(world_state)
    # Log once no action is taken (the current state is associated with no action)
    if state_log is not None:
        state_log.append(('action', effect_list))

    # TODO The defender may have ways of modifying the attack pool at this point

    # Now calculate the damage done to the defender
    # Apply effect on the entire attack at this point
    damage = ArmadaDice.pool_damage(pool_faces)
    if defense_effects[ArmadaTypes.defense_tokens.index('scatter')]:
        damage = 0

    if defense_effects[ArmadaTypes.defense_tokens.index('brace')]:
        damage = damage // 2 + damage % 2

    world_state.setSubPhase("attack - resolve damage")
    damage_cards = 0
    if defense_effects[ArmadaTypes.defense_tokens.index('redirect')]:
        # Redirect to the target hull zone(s), but don't redirect more damage than is present
        for redirect_hull, redirect_amount in redirects:
            redirect_amount = min(damage, redirect_amount)
            # Remember if damage will be directed to the hull to see if the standard or XX9
            # critical should be triggered.
            # This would be a weird thing to do, but technically you could redirect damage to the
            # hull if you want shields on the defending hull zone for some reason (or because you
            # are a random agent or dumb network doing something dumb).
            damage_cards += defender[0].shield_damage(redirect_hull, redirect_amount)
            # Reduce the damage left
            damage = damage - redirect_amount

    # Deal remaining damage to the shield in the defending hull zone
    damage_cards += defender[0].shield_damage(attack.defending_hull, damage)
    # TODO FIXME Handle criticals and the contain token
    defender[0].damage(attack.defending_hull, damage_cards)
    # TODO FIXME HERE Logging
    #world_state.attack['{} damage'.format(defender[1])] = damage


    # Log the final attack state if the log is present
    if state_log is not None:
        state_log.append(('state', world_state.clone()))
    # TODO Log the world state into a game log
    logging.info(world_state)

    # Clear the attack-only states and return the world state with modified ship status
    world_state.attack = None
    return world_state
