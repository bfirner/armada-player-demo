# Copyright Bernhard Firner, 2020
# Should be run with pytest:
# > python3 -m pytest
# For profiling:
# > python3 -m pytest test_learning.py --profile
# And then
# >>> import pstats
# >>> p = pstats.Stats('./prof/combined.prof')
# >>> p.strip_dirs()
# >>> p.sort_stats('cumtime')
# >>> p.print_stats(50)
# Or just call snakeviz on the prof file.

import torch.multiprocessing as multiprocessing
import numpy
import pytest
import random
import torch

import utility
from armada_encodings import (Encodings)
from game_constants import (ArmadaPhases, ArmadaTypes)
from learning_agent import (LearningAgent)
from learning_components import (collect_attack_batches, get_n_examples)
from model import (SeparatePhaseModel)
from random_action_dataset import (RandomActionDataset)
from random_agent import (RandomAgent)
from ship import (Ship)
from world_state import (AttackState, WorldState)

# Initialize ships from the test ship list
keys, ship_templates = utility.parseShips('data/test_ships.csv')

# Test the defense tokens by comparing the results of the test ships with and without those tokens


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
    #f = torch.nn.ReLU()
    #epsilon = 0.001
    #normed_predictions = f(prediction[0]) + epsilon
    with torch.no_grad():
        error = (prediction - labels).abs().mean().item()

    with torch.no_grad():
        errors = (prediction - labels).abs()
        for i in range(errors.size(0)):
            # Debug on crazy errors or nan values.
            if errors[i] > 1000 or errors[i] != errors[i]:
                # This is messy debugging code, but sometimes a test may fail after bug
                # introductions so it is helpful to leave this in to speed up debugging.
                world_size = Encodings.calculateWorldStateSize()
                world_encoding = batch[i,:world_size]
                phase_name = ArmadaPhases.main_phases[int(world_encoding[1].item())]
                sub_phase_name = ArmadaPhases.sub_phases[phase_name][int(world_encoding[2].item())]
                print(f"Error {i} is {errors[i]}")
                print(f"\tRound {world_encoding[0].item()}")
                print(f"\tSubphase {sub_phase_name}")
                action_size = Encodings.calculateActionSize(sub_phase_name)
                attack_size = Encodings.calculateAttackSize()
                action_encoding = batch[i,world_size:world_size + action_size]
                attack_state_encoding = batch[i,world_size + action_size:]
                if "attack - resolve attack effects" == sub_phase_name:
                    print(f"\tattack effect encoding is {action_encoding}")
                elif "attack - spend defense tokens" == sub_phase_name:
                    print(f"\tspend defense token encoding is {action_encoding}")
                else:
                    print("Cannot print information about {}".format(sub_phase_name))
                defender = Ship(name="Defender", player_number=1,
                                encoding=attack_state_encoding[:Ship.encodeSize()])
                attacker = Ship(name="Attacker", player_number=1,
                                encoding=attack_state_encoding[Ship.encodeSize():2 * Ship.encodeSize()])
                # print(f"\tAttack state encoding is {attack_state_encoding}")
                print("\tAttacker is {}".format(attacker))
                print("\tDefender is {}".format(defender))
                # TODO FIXME Enough dice in a pool seems to end a ship, but unless the pools are
                # incredibly large this doesn't seem to be happening. Damage does not seem to be
                # accumlating between rounds.
                die_offset = Encodings.getAttackDiceOffset()
                dice_encoding = attack_state_encoding[die_offset:die_offset + Encodings.dieEncodingSize()]
                print("\tDice are {}".format(dice_encoding))
                print(f"\tLabel is {labels[i]}")


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


@pytest.fixture(scope="session")
def create_attack_effects_dataset():
    """Create a training dataset for the "attack - resolve attack effects" subphase.

    Returns:
        (List[(str, world_state or attack effect tuple)],
         List[(str, world_state or attack effect tuple)]): Tuple of training and evaluation data.
    """
    # Seed the RNG and make sure this test is deterministic
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    phase = "attack - resolve attack effects"
    train_dataloader = torch.utils.data.DataLoader(
            dataset=RandomActionDataset(subphase=phase, num_samples=500, batch_size=32), batch_size=None, shuffle=False,
            sampler=None, batch_sampler=None, num_workers=10, pin_memory=True, drop_last=False,
            multiprocessing_context=None)
    eval_dataloader = torch.utils.data.DataLoader(
            dataset=RandomActionDataset(subphase=phase, num_samples=250, batch_size=32, deterministic=True),
            batch_size=None, shuffle=False, sampler=None, batch_sampler=None, num_workers=7,
            pin_memory=True, drop_last=False, multiprocessing_context=None)

    return train_dataloader, eval_dataloader


@pytest.fixture(scope="session")
def create_spend_defense_tokens_dataset():
    """Create a training dataset for the "attack - resolve attack effects" subphase.

    Returns:
        (List[(str, world_state or attack effect tuple)],
         List[(str, world_state or attack effect tuple)]): Tuple of training and evaluation data.
    """
    # Seed the RNG and make sure this test is deterministic
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    phase = "attack - spend defense tokens"
    train_dataloader = torch.utils.data.DataLoader(
            dataset=RandomActionDataset(subphase=phase, num_samples=500, batch_size=32, deterministic=True),
            batch_size=None, shuffle=False,
            sampler=None, batch_sampler=None, num_workers=10, pin_memory=True, drop_last=False,
            multiprocessing_context=None)
    eval_dataloader = torch.utils.data.DataLoader(
            dataset=RandomActionDataset(subphase=phase, num_samples=250, batch_size=32, deterministic=True),
            batch_size=None, shuffle=False,
            sampler=None, batch_sampler=None, num_workers=8, pin_memory=True, drop_last=False,
            multiprocessing_context=None)

    return train_dataloader, eval_dataloader


@pytest.fixture(scope="session")
def create_nonrandom_training_dataset():
    """Create a training dataset.

    Returns:
        (List[(str, world_state or attack effect tuple)],
         List[(str, world_state or attack effect tuple)]): Tuple of training and evaluation data.
    """
    # Seed the RNG and make sure this test is deterministic
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # TODO FIXME Seed all RNGs so that tests cannot fail intermittently

    # TODO FIXME Remember to seed the regular Python RNG as well
    agent = SimpleAgent()

    attacker = Ship(name="Attacker", template=ship_templates["Attacker"], upgrades=[], player_number=1)

    no_brace = Ship(name="No Defense Tokens", template=ship_templates["No Defense Tokens"], upgrades=[], player_number=2)
    one_brace = Ship(name="Single Brace", template=ship_templates["Single Brace"], upgrades=[], player_number=2)
    two_brace = Ship(name="Double Brace", template=ship_templates["Double Brace"], upgrades=[], player_number=2)
    all_tokens = Ship(name="All Tokens", template=ship_templates["All Defense Tokens"], upgrades=[], player_number=2)

    # Generate 100 trials per pairing to compensate for the natural variability in rolls
    processes = []
    queues = []
    target_ships = [no_brace, one_brace, two_brace, all_tokens]
    for target in target_ships:
        queues.append(multiprocessing.Queue())
        processes.append(multiprocessing.Process(
            target=get_n_examples, args=(10, ship_a, target, agent, queues[-1])))
        processes[-1].start()

    data = []
    for p in range(len(target_ships)):
        eval_attacks += queues[p].get()
        processes[p].join()

    return data


@pytest.fixture(scope="session")
def resolve_attack_effects_model(create_attack_effects_dataset):
    """Train some basic lifetime prediction models.

    Create a simple network that predict defending ship lifetimes during the 'resolve attack
    effects' phase. Also creates logs of training and evaluation loss.

    Returns:
        (nn.module, [float], [float]): The model, training errors per epoch, and eval errors per
                                       epoch.
    """
    attacks, eval_attacks = create_attack_effects_dataset

    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Lifetime network A predicts lifetimes given a state and action pair in the 'resolve attack
    # effects' subphase.
    phase_name = "attack - resolve attack effects"
    world_size = Encodings.calculateWorldStateSize()
    action_size = Encodings.calculateActionSize(phase_name)
    attack_size = Encodings.calculateAttackSize()
    input_size = world_size + action_size + attack_size
    # The network size was made large enough that training plateaued to a stable value
    # If the network is too large it has a tendency to start fitting to specific cases.
    # Batchnorm doesn't seem to help out much with this network and task.
    network = torch.nn.Sequential(
        torch.nn.BatchNorm1d(input_size),
        torch.nn.Linear(input_size, 2 * input_size),
        torch.nn.ELU(),
        torch.nn.Linear(2 * input_size, 4 * input_size),
        torch.nn.ELU(),
        #torch.nn.Dropout(),
        #torch.nn.BatchNorm1d(4 * input_size),
        torch.nn.Linear(4 * input_size, 2 * input_size),
        torch.nn.ELU(),
        torch.nn.Linear(2 * input_size, 1))
    network.to(target_device)
    # Higher learning rates lead to a lot of instability in the training.
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0005)

    # Keep track of the errors for the purpose of this test
    errors = []
    eval_errors = []

    # Evaluate before training and every epoch
    for batch in eval_attacks:
        eval_data = batch[0].to(target_device)
        eval_labels = batch[1].to(target_device)
        eval_errors.append(update_lifetime_network(network, eval_data,
                                                   eval_labels, None, True))
    # Train with all of the data for 10 epochs
    for epoch in range(10):
        print("Training resolve_attack_effects_model epoch {}".format(epoch))
        epoch_samples = 0
        train_batches = 0
        for batch in attacks:
            train_data = batch[0].to(target_device)
            train_labels = batch[1].to(target_device)
            errors.append(update_lifetime_network(network, train_data,
                                                  train_labels, optimizer))
            epoch_samples += batch[0].size(0)
            train_batches += 1
        print("Finished epoch with {} batches.".format(train_batches))

        # Evaluate every epoch
        for batch in eval_attacks:
            eval_data = batch[0].to(target_device)
            eval_labels = batch[1].to(target_device)
            eval_errors.append(update_lifetime_network(network, eval_data,
                                                       eval_labels, None, True))

    return network, errors, eval_errors


@pytest.fixture(scope="session")
def spend_defense_tokens_model(create_spend_defense_tokens_dataset):
    """Train some basic lifetime prediction models.

    Create simple networks that predict defending ship lifetimes during the 'spend defese tokens'
    phase and during the 'resolve attack effects' phase.

    Returns:
        (nn.module, [float], [float]): The model, training errors per epoch, and eval errors per
                                       epoch.
    """
    attacks, eval_attacks = create_spend_defense_tokens_dataset

    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Lifetime network B predicts lifetimes given a state and action pair in the 'spend defense
    # tokens' subphase.
    phase_name = "attack - spend defense tokens"
    world_size = Encodings.calculateWorldStateSize()
    action_size = Encodings.calculateActionSize(phase_name)
    attack_size = Encodings.calculateAttackSize()
    input_size = world_size + action_size + attack_size
    network = torch.nn.Sequential(
        torch.nn.BatchNorm1d(input_size),
        torch.nn.Linear(input_size, 2 * input_size),
        torch.nn.ELU(),
        torch.nn.Linear(2 * input_size, 4 * input_size),
        torch.nn.ELU(),
        torch.nn.Linear(4 * input_size, 2 * input_size),
        torch.nn.ELU(),
        torch.nn.Linear(2 * input_size, 1))
    network.to(target_device)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0005)
    batch_size = 32
    batch = torch.Tensor(batch_size, input_size).to(target_device)

    # Keep track of the errors for the purpose of this test
    errors = []
    eval_errors = []

    # Evaluate before training and every epoch
    for batch in eval_attacks:
        eval_data = batch[0].to(target_device)
        eval_labels = batch[1].to(target_device)
        eval_errors.append(update_lifetime_network(network, eval_data,
                                                   eval_labels, None, True))
    # Train with all of the data for 10 epochs
    for epoch in range(10):
        print("Training resolve_spend_defense_tokens_model epoch {}".format(epoch))
        train_batches = 0
        for batch in attacks:
            train_data = batch[0].to(target_device)
            train_labels = batch[1].to(target_device)
            errors.append(update_lifetime_network(network, train_data,
                                                  train_labels, optimizer))
            train_batches += 1
        print("Finished epoch with {} batches.".format(train_batches))
        # Evaluate every epoch
        for batch in eval_attacks:
            eval_data = batch[0].to(target_device)
            eval_labels = batch[1].to(target_device)
            eval_errors.append(update_lifetime_network(network, eval_data,
                                                       eval_labels, None, True))

    return network, errors, eval_errors


def test_policy_learning(spend_defense_tokens_model, resolve_attack_effects_model):
    """Train a model to produce a better than random choice policy for defense tokens.

    The spend_defense_tokens_model will be used to determine the quality of this network's output.
    There will not be an update step as in reinforcement learning, this is just testing the
    mechanism.

    Returns:
        (nn.module, nn.module,): The 'resolve attack effects' and 'spend defense tokens' models.
    """
    def_tokens_model, errors, eval_errors = spend_defense_tokens_model
    def_tokens_model.eval()
    res_attack_model, errors, eval_errors = resolve_attack_effects_model
    res_attack_model.eval()
    prediction_models = {
        "attack - spend defense tokens": def_tokens_model.eval(),
        "attack - resolve attack effects": res_attack_model.eval()
    }
    # Do the training. Use the prediction model lifetime to create the loss target. The loss
    # will be the difference between the max possible round and the predicted round.
    # For the defense token model the higher round is better, for the attack effect model a
    # lower round is better.
    loss_fn = {
        "attack - spend defense tokens": lambda predictions: 7.0 - predictions,
        "attack - resolve attack effects": lambda predictions: predictions - 1.
    }

    # Generate a new learning model
    learning_agent = LearningAgent(SeparatePhaseModel())
    # TODO FIXME The learning agent doesn't really have a training mode
    learning_agent.model.train()
    random_agent = RandomAgent()

    training_ships = ["All Defense Tokens", "All Defense Tokens",
                      "Imperial II-class Star Destroyer", "MC80 Command Cruiser",
                      "Assault Frigate Mark II A", "No Shield Ship", "One Shield Ship",
                      "Mega Die Ship"]
    defenders = []
    attackers = []
    for name in training_ships:
        attackers.append(Ship(name=name, template=ship_templates[name],
                              upgrades=[], player_number=1, device='cpu'))
    for name in training_ships:
        defenders.append(Ship(name=name, template=ship_templates[name],
                              upgrades=[], player_number=2, device='cpu'))

    batch_size = 32
    # Remember the training loss values to test for improvement
    losses = {}
    for subphase in ["attack - spend defense tokens", "attack - resolve attack effects"]:
        losses[subphase] = []
    # This gets samples to use for training. We will use a random agent to generate the states.
    # In reinforcement learning the agent would alternate between random actions and actions
    # from the learning agent to balance exploration of the state space with exploitation of
    # learned behavior.
    samples = get_n_examples(
        n_examples=1000, ship_a=attackers, ship_b=defenders, agent=random_agent)

    # Get a batch for each subphase
    for subphase in ["attack - spend defense tokens", "attack - resolve attack effects"]:
        # The learning_agent will take in the world state and attack state and will produce a new
        # action encoding. Then the prediction network will take in a new tuple with this action and
        # predict a final round. The difference between the random action and the final round with
        # the network's action is the reward.
        world_size = Encodings.calculateWorldStateSize()
        action_size = Encodings.calculateActionSize(subphase)
        attack_size = Encodings.calculateAttackSize()
        for batch, lifetimes in collect_attack_batches(batch_size=batch_size, attacks=samples, subphase=subphase):
            batch = batch.cuda()
            # The learning agent takes in the world state along with the action as the input tensor.
            new_batch = torch.cat((batch[:,:world_size], batch[:,world_size + action_size:]), dim=1).cuda()
            # TODO FIXME Just make a forward function that takes in a phase name
            new_actions = learning_agent.model.models[subphase](new_batch)
            new_action_state = torch.cat((batch[:,:world_size], new_actions, batch[:,world_size + action_size:]), dim=1)
            action_result = prediction_models[subphase](new_action_state)
            loss = loss_fn[subphase](action_result)
            learning_agent.model.get_optimizer(subphase).zero_grad()
            loss.sum().backward()
            learning_agent.model.get_optimizer(subphase).step()
            with torch.no_grad():
                losses[subphase].append(loss.mean().item())
            # In reinforcement learning there would also be a phase where the prediction models are
            # updated.
    for subphase in ["attack - spend defense tokens", "attack - resolve attack effects"]:
        print(f"losses for {subphase} start with {losses[subphase][0:5]} and end with {losses[subphase][-5:]}")
        assert losses[subphase][-1] < losses[subphase][0]
    # TODO FIXME HERE See if the policy networks produce better results than random actions
    learning_agent.model.models.eval()


def test_resolve_attack_effects_model(resolve_attack_effects_model):
    """Test basic network learning loop.

    Test lifetime predictions during the resolve attack effects phase.
    """
    network, errors, eval_errors = resolve_attack_effects_model
    network.eval()
    phase_name = "attack - resolve attack effects"
    world_size = Encodings.calculateWorldStateSize()
    action_size = Encodings.calculateActionSize(phase_name)
    attack_size = Encodings.calculateAttackSize()
    input_size = world_size + action_size + attack_size

    # First verify that errors decreased during training.
    # print("Errors for A were {}".format(errors))
    print("Eval errors for A were {}".format(eval_errors))
    assert eval_errors[0] > eval_errors[-1]

    # Let's examine predictions for different ranges and hull zones.
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    batch = torch.Tensor(batch_size, input_size).to(target_device)

    # Let's examine predictions for different dice pools and spent defense tokens.
    # Go through the following scenarios:
    # 1.1 An attack upon a ship with only 1 hull remaining
    # 1.2 The same dice pool but on a ship with full hull
    # 1.3 A dice pool with only blank dice
    # 1.4 A dice pool with only blanks when attacking at long range.

    # Create a state from resolve attack effects and an empty action.
    world_state = WorldState()
    world_state.round = 1
    ship_a = Ship(name="Ship A", template=ship_templates["All Defense Tokens"], upgrades=[], player_number=1)
    ship_b = Ship(name="Ship B", template=ship_templates["All Defense Tokens"], upgrades=[], player_number=2)
    world_state.addShip(ship_a, 0)
    world_state.addShip(ship_b, 1)
    pool_colors, pool_faces = ['black'] * 4, ['hit_crit'] * 4
    world_state.setPhase("ship phase", "attack - resolve attack effects")
    ship_b.set('damage', ship_b.get('hull') - 1)
    attack = AttackState('short', ship_a, 'left', ship_b, 'front', pool_colors, pool_faces)
    world_state.updateAttack(attack)
    action_encoding = torch.cat((Encodings.encodeWorldState(world_state),
                                 Encodings.encodeAction(world_state.sub_phase, None)))
    state_encoding = Encodings.encodeAttackState(world_state)
    batch[0] = torch.cat(
        (action_encoding.to(target_device), state_encoding.to(target_device)))

    # Same dice pool but the defender has full hull
    ship_b.set('damage', 0)
    attack = AttackState('short', ship_a, 'left', ship_b, 'front', pool_colors, pool_faces)
    world_state.updateAttack(attack)
    action_encoding = torch.cat((Encodings.encodeWorldState(world_state),
                                 Encodings.encodeAction(world_state.sub_phase, None)))
    state_encoding = Encodings.encodeAttackState(world_state)
    batch[1] = torch.cat(
        (action_encoding.to(target_device), state_encoding.to(target_device)))

    # Full hull and all blanks
    pool_colors, pool_faces = ['black'] * 4, ['blank'] * 4
    world_state.setPhase("ship phase", "attack - resolve attack effects")
    attack = AttackState('short', ship_a, 'left', ship_b, 'front', pool_colors, pool_faces)
    world_state.updateAttack(attack)
    state_encoding = Encodings.encodeAttackState(world_state)
    batch[2] = torch.cat(
        (action_encoding.to(target_device), state_encoding.to(target_device)))

    # Full hull, all blanks, firing at red range
    pool_colors, pool_faces = ['red'] * 2, ['blank'] * 2
    world_state.setPhase("ship phase", "attack - resolve attack effects")
    attack = AttackState('long', ship_a, 'left', ship_b, 'front', pool_colors, pool_faces)
    world_state.updateAttack(attack)
    state_encoding = Encodings.encodeAttackState(world_state)
    batch[3] = torch.cat(
        (action_encoding.to(target_device), state_encoding.to(target_device)))

    lifetime_out = network(batch)
    print("super cool attack effects round estimates are {}".format(lifetime_out))

    # The lifetimes should go up sequentially with the above scenarios.
    # However if the ship won't be destroyed the NN can't make an accurate relative number so be
    # lenient once lifetimes go above round 6. The first scenario should result in destruction
    # however.
    assert(lifetime_out[0].item() < 6)
    for i in range(batch.size(0) - 1):
        assert(lifetime_out[i].item() < lifetime_out[i+1].item() or 
                (lifetime_out[i].item() > 6. and lifetime_out[i+1].item() > 6.))


def test_defense_tokens_model(spend_defense_tokens_model):
    """Test basic network learning loop.

    Test lifetime predictions during the spend defense tokens phase.
    """
    network, errors, eval_errors = spend_defense_tokens_model
    network.eval()
    phase_name = "attack - spend defense tokens"
    world_size = Encodings.calculateWorldStateSize()
    action_size = Encodings.calculateActionSize(phase_name)
    attack_size = Encodings.calculateAttackSize()
    input_size = world_size + action_size + attack_size

    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    batch = torch.Tensor(batch_size, input_size).to(target_device)

    print("Eval errors for B were {}".format(eval_errors))
    #print("First and last errors in b are {} and {}".format(eval_errors[0], eval_errors[-1]))

    assert eval_errors[0] > eval_errors[-1]

    # Let's examine predictions for different dice pools and spent defense tokens.
    # Go through the following scenarios:
    # 1.1 An attack with more than enough damage to destroy the ship
    # 1.2 The same attack but a brace that would prevent destruction
    # 1.3 The same attack but a redirect that would prevent destruction
    # Result: 1.1 should have lower lifetime than 1.2 and 1.3
    # 2.1 An attack that can barely destroy the ship
    # 2.2 An attack that barely will not destroy the ship
    # Result: 2.1 should have lower lifetime than 2.2.
    # Ideally 1.1 and 2.1 would predict the current round.
    world_state = WorldState()
    world_state.round = 1
    ship_a = Ship(name="Ship A", template=ship_templates["All Defense Tokens"], upgrades=[], player_number=1)
    ship_b = Ship(name="Ship B", template=ship_templates["All Defense Tokens"], upgrades=[], player_number=2)
    world_state.addShip(ship_a, 0)
    world_state.addShip(ship_b, 1)
    pool_colors, pool_faces = ['black'] * 4, ['hit_crit'] * 4
    world_state.setPhase("ship phase", phase_name)
    # Set the front hull zone to 2 shields
    ship_b.get_range('shields')[ArmadaTypes.hull_zones.index('front')] = 2
    # Set the hull to 3 (by assigning damage to reduce the remaining hull to 3)
    ship_b.set('damage', ship_b.get('hull') - 3)
    attack = AttackState('short', ship_a, 'left', ship_b, 'front', pool_colors, pool_faces)
    world_state.updateAttack(attack)
    action_encoding = torch.cat((Encodings.encodeWorldState(world_state),
                                 Encodings.encodeAction(world_state.sub_phase, None)))
    state_encoding = Encodings.encodeAttackState(world_state)
    batch[0] = torch.cat(
        (action_encoding.to(target_device), state_encoding.to(target_device)))

    action = [("brace", (ArmadaTypes.green, None))]
    action_encoding = torch.cat((Encodings.encodeWorldState(world_state),
                                 Encodings.encodeAction(world_state.sub_phase, action)))
    state_encoding = Encodings.encodeAttackState(world_state)
    batch[1] = torch.cat(
        (action_encoding.to(target_device), state_encoding.to(target_device)))

    world_state = WorldState()
    world_state.round = 1
    ship_a = Ship(name="Ship A", template=ship_templates["All Defense Tokens"], upgrades=[], player_number=1)
    ship_b = Ship(name="Ship B", template=ship_templates["All Defense Tokens"], upgrades=[], player_number=2)
    world_state.addShip(ship_a, 0)
    world_state.addShip(ship_b, 1)
    pool_colors, pool_faces = ['black'] * 4, ['hit_crit'] * 2 + ['hit'] * 2
    world_state.setPhase("ship phase", phase_name)
    # Set the front hull zone to 2 shields
    ship_b.get_range('shields')[ArmadaTypes.hull_zones.index('front')] = 2
    # Set the hull to 3 (by assigning damage to reduce the remaining hull to 3)
    ship_b.set('damage', ship_b.get('hull') - 3)
    attack = AttackState('short', ship_a, 'left', ship_b, 'front', pool_colors, pool_faces)
    world_state.updateAttack(attack)

    action = [("redirect", (ArmadaTypes.green, [('left', 4)]))]
    action_encoding = torch.cat((Encodings.encodeWorldState(world_state),
                                 Encodings.encodeAction(world_state.sub_phase, action)))
    state_encoding = Encodings.encodeAttackState(world_state)
    batch[2] = torch.cat(
        (action_encoding.to(target_device), state_encoding.to(target_device)))

    round_status = network(batch[:3])
    print("super cool estimated rounds of destructions are {}".format(round_status[:3]))

    # Using no defense token results in destruction, the final round should be less
    assert(round_status[0].item() < round_status[1].item())
    assert(round_status[0].item() < round_status[2].item())


def test_policy_network(resolve_attack_effects_model, spend_defense_tokens_model):
    """Test basic network learning loop.

    Create a simple network to spend defense tokens. Verify that it extends lifetimes.
    Then train a network to spend accuracy icons. Verify that it reduces lifetimes.
    """

    # Test to actually train the action prediction networks.

    # TODO FIXME HERE It would make more sense to just move on to training behaviors.
    # Make a new test to start training actions that yield the highest lifetimes.
    # This would probably require some discounting of the future and multiple runs to get the actual
    # best outcomes, but we should be able to train a network the same way as in this test and
    # backpropagate through it to maximize some reward (either destroying the defender by choosing
    # accuracies well or prolonging its lifetime by spending tokens wisely).
    pass
