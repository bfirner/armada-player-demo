# Copyright Bernhard Firner, 2020

import random
import torch

from armada_encodings import (Encodings)
from game_constants import (ArmadaTypes)
from game_engine import handleAttack
from world_state import (WorldState)


def a_vs_b(ship_a, ship_b, agent_a, agent_b, ship_a_hull, trials, attack_range):
    """This function runs multiple trials of ship_a firing upon ship_b.

    Args:
        ship_a ((Ship, str)): Attacker and hull zone tuple.
        ship_b ((Ship, str)): Defender and hull zone tuple.
        agent_a  (BaseAgent): Agent to control the actions of ship a.
        agent_b  (BaseAgent): Agent to control the actions of ship b.
        ship_a_hull (str)   : Attacking hull zone.
        trials (int): Number of trials in average calculation.
        range (str): Attack range.
    Returns:
        List[(str, world_state or attack effect tuple)]
    """
    state_log = []
    failures = 0
    for _ in range(trials):
        # Reset ship b for each trial
        ship_b.reset()
        world_state = WorldState()
        world_state.addShip(ship_a, 0)
        world_state.addShip(ship_b, 1)
        num_rolls = 0
        # Don't attempt forever in the case of some catastrophic reoccurring error.
        attempts = 0
        while ship_b.damage_cards() < ship_b.hull() and attempts < 250:
            attempts += 1
            # Handle the attack and receive the updated world state
            try:
                roll_log = []
                world_state = handleAttack(world_state=world_state, attacker=(ship_a, ship_a_hull),
                                           defender=(ship_b, "front"), attack_range=attack_range,
                                           offensive_agent=agent_a, defensive_agent=agent_b,
                                           state_log=state_log)
                # Only add these actions to the returned log if they are all legal.
                state_log += roll_log
                num_rolls += 1
            except RuntimeError as err:
                # This is fine, the random agent will do illegal things plenty of times
                pass
        if 250 == attempts:
            raise RuntimeError("Too many failures for ship firing simulation.")
    return state_log


def get_n_examples(n_examples, ship_a, ship_b, agent):
    """Smallest function to get 'n' examples of a_vs_b.

    This function is meant to be called in parallel to more quickly create training data.

    Arguments:
        n_examples (int)     : The number of examples to generate.
        ship_a (Ship)        : The attacking ship.
        ship_b (Ship)        : The defending ship.
        agent (LearningAgent): The agent to choose actions.
    Return:
        List[examples]       : List of the examples world states and actions.
    """
    attacks = []
    for _ in range(n_examples):
        # At short range all hull zones can attack, otherwise only the front zone can attack.
        attack_range = random.choice(['long', 'medium', 'short', 'short', 'short', 'short'])
        if attack_range == 'short':
            # Only testing the small ship hull zones
            hull = random.choice(['left', 'right', 'front', 'rear'])
        else:
            hull = 'front'
        attacks.append(
            a_vs_b(ship_a=ship_a, ship_b=ship_b, agent_a=agent, agent_b=agent,
                   ship_a_hull=hull, trials=1, attack_range=attack_range))
    return attacks


def collect_attack_batches(batch, labels, attacks, subphase):
    """A generator to collect training batches from a list of attack logs.

    Collect all of the actions taken during the resolve attack effects stage of a trial and
    associate them with the number of additional attacks required to end the trial.  Only sample a
    single action-state pair from each trial into the training batch though. This avoids the network
    simply memorizing the output of specific scenarios in the event of fairly unique events (for
    example a certain unlikely dice roll or combination of shields and hull in the defending ship).

    Args:
        batch (torch.Tensor)        : Training tensor to fill. First dimension is the batch size.
        labels (torch.Tensor)       : Labels tensor. First dimension must match the batch argument.
        attacks (List[List[tuples]]): Each sublist is all of the states and actions from a sequence.
        subphase (str)              : Name of the subphase where state/action pairs are collected.
    Returns:
        Number of items filled.
    """
    # Variables for collection
    batch_size = batch.size(0)
    target_device = batch.device
    # collect_state records what we are doing inside of the sample loop
    collect_state = 0
    # The state and (state, action) pairs collected from the current trial
    last_state = None
    state_actions_attacks = []
    # Counter for the training target
    attack_count = 0
    cur_sample = 0
    for attack in attacks:
        if 'state' == attack[0]:
            if attack[1].sub_phase == "attack - declare":
                # If we are transitioning into a new attack then increase the attack counter.
                attack_count += 1
            if attack[1].sub_phase == subphase:
                last_state = attack[1]
                collect_state = 1
            elif (attack[1].sub_phase == "attack - resolve damage" and
                    attack[1].attack.defender.damage_cards() >= attack[1].attack.defender.hull()):
                # The trial has completed, calculate the total number of attacks and choose a
                # sample for training.
                collect_state = 3
            else:
                # Not in the desired subphase and the attack trial is not complete.
                # Waiting for the next attack in the trial or for the trial to end.
                collect_state = 2
        elif 'action' == attack[0] and 1 == collect_state:
            # Collect the actions associated with last_state. The attack count will later be
            # corrected to be the number of total attacks from that state rather than the current
            # attack.
            state_actions_attacks.append((last_state, attack[1], attack_count))
        # Choose a training sample and calculate attacks.
        if 3 == collect_state and 0 < len(state_actions_attacks):
            # Collect a single sample
            selected = random.choice(state_actions_attacks)
            state_actions_attacks = []
            # The training label for the last attacks should be 0, for the next to last 1, etc.
            labels[cur_sample] = attack_count - selected[2]
            action_encoding = Encodings.encodeAction(selected[0].sub_phase, selected[1])
            state_encoding, die_slots = Encodings.encodeAttackState(selected[0])
            batch[cur_sample] = torch.cat(
                (action_encoding.to(target_device), state_encoding.to(target_device)))
            cur_sample += 1
            attack_count = 0
        # When a full batch is collected return it immediately.
        if cur_sample == batch_size:
            yield(cur_sample)
            cur_sample = 0
            
    # If there are leftover samples then train again
    if 0 < cur_sample:
        yield(cur_sample)
