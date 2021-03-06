# Copyright Bernhard Firner, 2020

import random
import torch

from armada_encodings import (Encodings)
from game_constants import (ArmadaPhases, ArmadaTypes)
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
        # Begin at round 1
        world_state.round = 1
        # Don't attempt forever in the case of some catastrophic reoccurring error.
        attempts = 0
        while ship_b.damage_cards() < ship_b.hull() and world_state.round <= ArmadaPhases.max_rounds:
            attempts += 1
            # Handle the attack and receive the updated world state
            #try:
            world_state = handleAttack(world_state=world_state, attacker=(ship_a, ship_a_hull),
                                       defender=(ship_b, "front"), attack_range=attack_range,
                                       offensive_agent=agent_a, defensive_agent=agent_b,
                                       state_log=state_log)
            # Record the final state with the incremented round number.
            world_state.setPhase("status phase", "increment round number")
            world_state.round += 1
            state_log.append(('state', world_state.clone()))
            #except RuntimeError as err:
            #    # This is fine, the random agent will do illegal things plenty of times
            #    pass
        if 250 == attempts:
            raise RuntimeError("Too many failures for ship firing simulation.")
    return state_log


def get_n_examples(n_examples, ship_a, ship_b, agent):
    """Smallest function to get 'n' examples of a_vs_b.

    This function is meant to be called in parallel to more quickly create training data.

    Arguments:
        n_examples (int)       : The number of examples to generate.
        ship_a (Ship or [Ship]): The attacking ship (or multiple possibilities).
        ship_b (Ship or [Ship]): The defending ship (or multiple possibilities).
        agent (LearningAgent)  : The agent to choose actions.
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
        attacker = ship_a
        defender = ship_b
        # Allow multiple possible attackers and defenders
        if type(attacker) is list:
            attacker = random.choice(attacker)
        if type(defender) is list:
            defender = random.choice(defender)
        attacks.append(
            a_vs_b(ship_a=attacker, ship_b=defender, agent_a=agent, agent_b=agent,
                   ship_a_hull=hull, trials=1, attack_range=attack_range))
    return attacks


def collect_attack_batches(batch_size, attacks, subphase):
    """A generator to collect training batches from a list of attack logs.

    Collect all of the actions taken during the resolve attack effects stage of a trial and
    associate them with the round number when the trial ended.  Only sample a single action-state
    pair from each trial into the training batch though. This avoids the network simply memorizing
    the output of specific scenarios in the event of fairly unique events (for example a certain
    unlikely dice roll or combination of shields and hull in the defending ship).

    Args:
        batch_size (int)            : Maximum batch size.
        attacks (List[List[tuples]]): Each sublist is all of the states and actions from a sequence.
        subphase (str)              : Name of the subphase where state/action pairs are collected.
    Returns:
        (batch, labels) : Tuple of the batch and labels.
    """
    # TODO FIXME This function can be used as an dataset iterator for a multithreaded dataloader so
    # the batch and labels should not be passed in. Instead this function must create new tensors
    # for each batch.
    world_size = Encodings.calculateWorldStateSize()
    action_size = Encodings.calculateActionSize(subphase)
    attack_size = Encodings.calculateAttackSize()
    input_size = world_size + action_size + attack_size
    batch = torch.zeros(batch_size, input_size)
    labels = torch.zeros(batch_size, 1)

    # Variables for collection
    # collect_state records what we are doing inside of the sample loop
    collect_state = 0
    # The state and (state, action) pairs collected from the current trial
    last_state = None
    state_actions_attacks = []
    # Counter for the training target
    attack_count = 0
    cur_sample = 0
    last_round = 0
    for sequence in attacks:
        # We only collect a single sample per sequence. Collect all of the state action pairs of
        # interest first and then select a single one to use.
        state_action_pairs = []
        for attack in sequence:
            if 'state' == attack[0]:
                last_round = attack[1].round
                if attack[1].sub_phase == subphase:
                    last_state = attack[1]
                    collect_state = 1
                else:
                    # Not in the desired subphase and the attack trial is not complete.
                    # Waiting for the next attack in the trial or for the trial to end.
                    collect_state = 2
            elif 'action' == attack[0] and 1 == collect_state:
                # Collect the actions associated with last_state. The attack count will later be
                # corrected to be the number of total attacks from that state rather than the current
                # attack.
                state_action_pairs.append((last_state, attack[1]))
        # Collect a single sample (as long as one was present)
        if 0 < len(state_action_pairs):
            selected = random.choice(state_action_pairs)
            # The training label is the final round where the ship was destroyed, or 7 if the ship
            # was not destroyed in the 6 game rounds.
            labels[cur_sample] = last_round
            world_size = Encodings.calculateWorldStateSize()
            action_size = Encodings.calculateActionSize(selected[0].sub_phase)
            attack_size = Encodings.calculateAttackSize()
            Encodings.encodeWorldState(world_state=selected[0], encoding=batch[cur_sample,:world_size])
            Encodings.encodeAction(subphase=selected[0].sub_phase,
                                   action_list=selected[1],
                                   encoding=batch[cur_sample,world_size:world_size + action_size])
            Encodings.encodeAttackState(world_state=selected[0],
                                        encoding=batch[cur_sample,world_size + action_size:])
            cur_sample += 1
        # When a full batch is collected return it immediately.
        if cur_sample == batch_size:
            yield((batch[:cur_sample], labels[:cur_sample]))
            # Make new buffers to store training data
            batch = torch.zeros(batch_size, input_size)
            labels = torch.zeros(batch_size, 1)
            cur_sample = 0
            
    # If there are leftover samples that did not fill a batch still return them.
    if 0 < cur_sample:
        yield((batch[:cur_sample], labels[:cur_sample]))
