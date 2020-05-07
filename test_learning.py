# Copyright Bernhard Firner, 2020
# Should be run with pytest:
# > python3 -m pytest

import numpy
import pytest
import random
import torch

import ship
import utility
from armada_encodings import (Encodings)
from game_constants import (ArmadaTypes)
from game_engine import handleAttack
from learning_agent import (LearningAgent)
from random_agent import (RandomAgent)
from world_state import (AttackState, WorldState)

# Initialize ships from the test ship list
keys, ship_templates = utility.parseShips('data/test_ships.csv')

# Test the defense tokens by comparing the results of the test ships with and without those tokens

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
        while 0 < ship_b.hull() and attempts < 250:
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
            except RuntimeError:
                # This is fine, the random agent will do illegal things plenty of times
                pass
        if 250 == attempts:
            raise RuntimeError("Too many failures for ship firing simulation.")
    return state_log


def update_lifetime_network(lifenet, batch, labels, optimizer, eval_only=False):
    """Do a forward and backward pass through the given lifetime network.

    Args:
        lifenet (torch.nn.Module): Trainable torch model
        batch (torch.tensor)     : Training batch
        labels (torch.tensor)    : Training labels
        optimizer (torch.nn.Optimizer) : Optimizer for lifenet parameters.
        eval_only (bool)         : Only evaluate, don't update parameters.
    Returns:
        batch error              : Average absolute error for this batch
    """
    if eval_only:
        lifenet.eval()
    # Forward through the prediction network
    prediction = lifenet.forward(batch)
    # Loss is the lifetime prediction error
    # The output cannot be negative, run through a ReLU to clean that up
    f = torch.nn.ReLU()
    epsilon = 0.001
    normed_predictions = f(prediction[0]) + epsilon
    with torch.no_grad():
        error = (normed_predictions - labels).abs().mean().item()

    # Normal distribution:
    #normal = torch.distributions.normal.Normal(predict_tensor[0], predict_tensor[1])
    #loss = -normal.log_prob(labels)
    # Poisson distribution (works well)
    #poisson = torch.distributions.poisson.Poisson(normed_predictions)
    #loss = -poisson.log_prob(labels)
    # Plain old MSE error
    #loss_fn = torch.nn.MSELoss()
    #loss = loss_fn(prediction, labels)
    # Absolute error
    loss_fn = torch.nn.L1Loss()
    loss = loss_fn(prediction, labels)

    if not eval_only:
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()
    else:
        lifenet.train()
    return error


def collect_attack_batches(batch, labels, attacks, subphase):
    """A generator to collect training batches from a list of attack logs.

    Args:
        batch (torch.Tensor)        : Training tensor to fill. First dimension is the batch size.
        labels (torch.Tensor)       : Labels tensor. First dimension must batch the batch argument.
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
            if attack[1].sub_phase == subphase:
                last_state = attack[1]
                # If we are transitioning into a new attack then increase the attack counter.
                if 0 != collect_state:
                    attack_count += 1
                collect_state = 1
            elif attack[1].sub_phase == "attack - resolve damage" and 0 == attack[1].attack.defender.hull():
                # The trial has completed, calculate the total number of attacks and choose a
                # sample for training.
                collect_state = 3
            else:
                # The trial has completed, calculate the total number of attacks and choose a
                # Waiting for the next attack in the trial or for the trial to end
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
            state_encoding, token_slots, die_slots = Encodings.encodeAttackState(selected[0])
            batch[cur_sample] = torch.cat(
                (action_encoding.to(target_device), state_encoding.to(target_device)))
            cur_sample += 1
            attack_count = 0
        # When a full batch is collected train immediately.
        if cur_sample == batch_size:
            yield(cur_sample)
            cur_sample = 0
            
    # If there are leftover samples then train again
    if 0 < cur_sample:
        yield(cur_sample)


def test_random_agent():
    """Test basic network learning loop.

    Creat a simple network to spend defense tokens. Verify that it extends lifetimes.
    Then train a network to spend accuracy icons. Verify that it reduces lifetimes.
    """
    # Seed the RNG and make sure this test is deterministic
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # TODO FIXME Remember to see the regular Python RNG as well
    randagent = RandomAgent()

    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO FIXME See all RNGs so that this test cannot fail intermittently
    loss_fn = torch.nn.MSELoss()

    attacker = ship.Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[], player_number=1)

    ship_a = ship.Ship(name="Ship A", template=ship_templates["All Defense Tokens"], upgrades=[], player_number=1)
    ship_b = ship.Ship(name="Ship B", template=ship_templates["All Defense Tokens"], upgrades=[], player_number=2)

    # Lifetime network A predicts lifetimes given a state and action pair in the 'resolve attack
    # effects' subphase.
    a_phase = "attack - resolve attack effects"
    a_input_size = Encodings.calculateActionSize(a_phase) + Encodings.calculateWorldStateSize()
    a_network = torch.nn.Sequential(
        torch.nn.Linear(a_input_size, 2 * a_input_size),
        torch.nn.ELU(),
        torch.nn.Linear(2 * a_input_size, 4 * a_input_size),
        torch.nn.ELU(),
        #torch.nn.BatchNorm1d(4 * a_input_size),
        torch.nn.Linear(4 * a_input_size, a_input_size),
        torch.nn.ELU(),
        torch.nn.Linear(a_input_size, 1))
    a_network.to(target_device)
    # Higher learning rates lead to a lot of instability in the training.
    a_optimizer = torch.optim.Adam(a_network.parameters(), lr=0.0005)

    # Create an evaluation dataset
    # Using a large batch for evaluation makes it easier to look at statistics since there is so
    # much noise in the samples.
    eval_size = 250
    eval_batch = torch.Tensor(eval_size, a_input_size).to(target_device)
    eval_labels = torch.Tensor(eval_size, 1).to(target_device)
    eval_attacks = []
    for _ in range(250):
        # At short range all hull zones can attack, otherwise only the front zone can attack.
        attack_range = random.choice(['long', 'medium', 'short', 'short', 'short', 'short'])
        if attack_range == 'short':
            hull = random.choice(ArmadaTypes.hull_zones)
        else:
            hull = 'front'
        eval_attacks = eval_attacks + a_vs_b(
            ship_a = ship_a, ship_b = ship_b, agent_a = randagent, agent_b = randagent,
            ship_a_hull = hull, trials = 1, attack_range = attack_range)

    # Run 2000 trials at each range to create training data for lifetime prediction.
    attacks = []
    for _ in range(2000):
        # At short range all hull zones can attack, otherwise only the front zone can attack.
        attack_range = random.choice(['long', 'medium', 'short', 'short', 'short', 'short'])
        if attack_range == 'short':
            hull = random.choice(ArmadaTypes.hull_zones)
        else:
            hull = 'front'
        attacks = attacks + a_vs_b(
            ship_a = ship_a, ship_b = ship_b, agent_a = randagent, agent_b = randagent,
            ship_a_hull = hull, trials = 1, attack_range = attack_range)

    batch_size = 32
    batch = torch.Tensor(batch_size, a_input_size).to(target_device)
    labels = torch.Tensor(batch_size, 1).to(target_device)
    # Collect all of the actions taken during the resolve attack effects stage of a trial and
    # associate them with the number of additional attacks required to end the trial.
    # Only sample a single action-state pair from each trial into the training batch though. This
    # avoids the network simply memorizing the output of specific scenarios in the event of fairly
    # unique events (for example a certain unlikely dice roll or combination of shields and hull in
    # the defending ship).

    # Keep track of the errors for the purpose of this test
    errors = []
    eval_errors = []

    # Grab a batch to evaluate
    for _ in collect_attack_batches(eval_batch, eval_labels, eval_attacks, a_phase):
        pass

    # Evaluate before training and every batch, which is total overkill
    eval_errors.append(update_lifetime_network(a_network, eval_batch,
                                               eval_labels, None, True))
    # Leave some for evaluation
    for num_samples in collect_attack_batches(batch, labels, attacks[:-eval_size], a_phase):
        errors.append(update_lifetime_network(a_network, batch[:num_samples],
                                              labels[:num_samples], a_optimizer))
        # Evaluate every batch, which is total overkill
        eval_errors.append(update_lifetime_network(a_network, eval_batch,
                                                   eval_labels, None, True))

    # First verify that errors decreased during training.
    #print("Errors for A were {}".format(errors))
    #print("Eval errors for A were {}".format(eval_errors))
    assert eval_errors[0] > eval_errors[-1]

    # Let's examine predictions for different ranges and hull zones.
    # Create a state from resolve attack effects and an empty action.
    world_state = WorldState()
    ship_a = ship.Ship(name="Ship A", template=ship_templates["All Defense Tokens"], upgrades=[], player_number=1)
    ship_b = ship.Ship(name="Ship B", template=ship_templates["All Defense Tokens"], upgrades=[], player_number=2)
    world_state.addShip(ship_a, 0)
    world_state.addShip(ship_b, 1)
    pool_colors, pool_faces = ['black'] * 4, ['hit_crit'] * 4
    world_state.setSubPhase("attack - resolve attack effects")
    ship_b._hull = 1
    attack = AttackState('short', ship_a, 'left', ship_b, 'front', pool_colors, pool_faces)
    world_state.updateAttack(attack)
    action_encoding = Encodings.encodeAction(world_state.sub_phase, None)
    state_encoding, token_slots, die_slots = Encodings.encodeAttackState(world_state)
    batch[0] = torch.cat(
        (action_encoding.to(target_device), state_encoding.to(target_device)))

    # Same dice pool but the defender has full hull
    ship_b._hull = 5
    attack = AttackState('short', ship_a, 'left', ship_b, 'front', pool_colors, pool_faces)
    world_state.updateAttack(attack)
    action_encoding = Encodings.encodeAction(world_state.sub_phase, None)
    state_encoding, token_slots, die_slots = Encodings.encodeAttackState(world_state)
    batch[1] = torch.cat(
        (action_encoding.to(target_device), state_encoding.to(target_device)))

    # Full hull and all blanks
    pool_colors, pool_faces = ['black'] * 4, ['blank'] * 4
    world_state.setSubPhase("attack - resolve attack effects")
    attack = AttackState('short', ship_a, 'left', ship_b, 'front', pool_colors, pool_faces)
    world_state.updateAttack(attack)
    state_encoding, token_slots, die_slots = Encodings.encodeAttackState(world_state)
    batch[2] = torch.cat(
        (action_encoding.to(target_device), state_encoding.to(target_device)))

    # Full hull, all blanks, firing at red range
    pool_colors, pool_faces = ['red'] * 2, ['blank'] * 2
    world_state.setSubPhase("attack - resolve attack effects")
    attack = AttackState('long', ship_a, 'front', ship_b, 'front', pool_colors, pool_faces)
    world_state.updateAttack(attack)
    state_encoding, token_slots, die_slots = Encodings.encodeAttackState(world_state)
    batch[3] = torch.cat(
        (action_encoding.to(target_device), state_encoding.to(target_device)))

    lifetime_out = a_network(batch[:4])

    # The lifetimes should go up as with the above scenarios
    #print("Lifetimes from A are {}".format(lifetime_out))
    for i in range(3):
        assert(lifetime_out[i].item() < lifetime_out[i+1].item())

    # So I take back what I said, in another test train a network to
    # predict lifetimes during the 'attack - spend defense tokens' step and verify that things are
    # as expected there. After that make this third test to actually train the action prediction
    # networks.
    # Lifetime network B predicts lifetimes given a state and action pair in the 'spend defense
    # tokens' subphase.
    b_phase = "attack - spend defense tokens"
    b_input_size = Encodings.calculateActionSize(b_phase) + Encodings.calculateWorldStateSize()
    b_network = torch.nn.Sequential(
        torch.nn.Linear(b_input_size, 2 * b_input_size),
        torch.nn.ELU(),
        torch.nn.Linear(2 * b_input_size, 4 * b_input_size),
        torch.nn.ELU(),
        torch.nn.Linear(4 * b_input_size, b_input_size),
        torch.nn.ELU(),
        torch.nn.Linear(b_input_size, 1))
    b_network.to(target_device)
    b_optimizer = torch.optim.Adam(b_network.parameters(), lr=0.0005)
    batch = torch.Tensor(batch_size, b_input_size).to(target_device)

    # Keep track of the errors for the purpose of this test
    errors = []

    for num_samples in collect_attack_batches(batch, labels, attacks, b_phase):
        errors.append(update_lifetime_network(b_network, batch[:num_samples],
                                              labels[:num_samples], b_optimizer))

    #print("First and last errors in b are {} and {}".format(errors[0], errors[-1]))
    #print("All errors in b are {}".format(errors))

    assert errors[0] > errors[-1]

    # TODO FIXME Add more "reasonable tests" as with network a.

    # TODO FIXME HERE It would make more sense to just move on to training behaviors.
    # Make a new test to start training actions that yield the highest lifetimes.
    # This would probably require some discounting of the future and multiple runs to get the actual
    # best outcomes, but we should be able to train a network the same was as in this test and
    # backpropagate through it to maximize some reward (either destroying the defender by choosing
    # accuracies well or prolonging its lifetime by spending tokens wisely).
    pass
