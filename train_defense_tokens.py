#
# Copyright Bernhard Firner, 2019
#
import random

from ship import (Ship)
from game_constants import (ArmadaPhases, ArmadaTypes)
from learning_agent import (LearningAgent)

from enum import Enum
import logging
import numpy
import os
import random
import torch

import argparse
import ship
import sys
import utility

from game_engine import handleAttack
from model import (ArmadaModel)
from simple_agent import (SimpleAgent)
from world_state import (WorldState)

parser = argparse.ArgumentParser(
    description='Learn how to spend defense tokens to maximize survival.',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-e',
    '--evaluate',
    action='store_true',
    default=False,
    required=False,
    help='Skip training, only evaluate. (not implemented)')
parser.add_argument(
    '--simruns',
    default=3000,
    required=False,
    type=int,
    help='Number of simulated battles to use for training.')
parser.add_argument(
    '--shipfile',
    default="data/test_ships.csv",
    required=False,
    type=str,
    help='csv file with ship templates.')
parser.add_argument(
    '-f',
    '--filename',
    default='defense_token_model.checkpoint',
    required=False,
    type=str,
    help='Checkpoint filename (for loading and saving).')
parser.add_argument(
    '--novelty',
    default=False,
    required=False,
    action='store_true',
    help="Attempt to use novelty-explorated instead of random. Does not currently work.")
args = parser.parse_args()


# Seed with time or a local source of randomness
random.seed()

#keys, ship_templates = utility.parseShips('data/armada-ship-stats.csv')
keys, ship_templates = utility.parseShips('data/test_ships.csv')

ship_names = [name for name in ship_templates.keys()]

# The goal:
#  We want a model that takes the attack state encoding as input and outputs an action that will
#  maximise the lifetime of the defender.
#    F(state) -> best action
#  We can train two parts to solve this problem. We want to train a model that maximizes the
#  lifetime of the defender but how can we get a target training label? If we already had a function
#  that mapped (state, action) pairs to the expected reward we could train with that:
#                                          ____________________
#          -----------------------------> |                    |-----> Lifetime
#          |    ___________               | Lifetime Predictor |
#  state --+-->| action    | -> action -> |____________________|
#              | predictor |
#              |___________|
#  We'll just always have the gradient push the network to predict an action with a longer lifetime.
#  Now we just need to also train this lifetime predictor. There are thus two networks, a "policy"
#  network the predicts an action and a "value" network that predicts the value of an action.
#  This isn't an optimal approach though. Deepmind trained their first go playing system in this
#  way, but later switched to an approach where only a single network is trained with exploration
#  using a minimum cost tree search.
#
#  We can try something similar. The lifetime at a node on the search tree is the average lifetimes
#  of multiple trials from that node.
#
# The plan:
# Choose a random scenario (range, ships)
# Loop:
#   1. Choose actions based upon policy network or a random action
#   2. Record the action and eventual reward (the lifetime) into a pool
#   3. Sample from the pool to train a prediction model. The prediction model should train for the
#   best possible result, or should predict a distribution. Can also train a policy net. 
#   4. (optional) Use the prediction network to train the policy model by following actions from
#      the policy model and recording the result. The reward is the comparison of the prediction
#      network to the actual outcome so the model will be trained to either outperform the
#      prediction or get close to it. This is an alternative to training the policy network
#      directly.
#
# _The Predictor_
# There is a lot of randomness in dice rolls, so our goal will be to train a density model. This
# means that we will train the network to estimate the parameters of the distribution that describes
# the lifetime. In plainer words, we will train the network to predict the average lifetime of the
# vessel and the undertainty of that prediction.  We are rolling three kinds of dice, so in reality
# this is the combination of three probability distributions, but to simplify things we will
# estimate it as a single normal.
# The loss will be the negative log of the probability of an outcome given a predicted mean and
# standard deviation. This works because as the probability approaches zero the negative log of it
# approaches infinity. So if the outputs the mean and sigma the loss will be:
# > d = torch.distributions(mean, sigma)
# > loss = -d.log_prob(result)

# Create a learning agent. This will initialize the model.
# TODO The learning agent should be the one to take random actions if it is not
# using novelty for exploration.
prediction_agent = LearningAgent(ArmadaModel(with_novelty=args.novelty))

# Load a previously trained model for additional training
# TODO FIXME HERE Reloading with the new novelty training stuff is not working
if os.path.isfile(args.filename):
    prediction_agent.model.load(args.filename)

optimizer = prediction_agent.model.get_optimizer("def_tokens")
if args.novelty:
    novelty_optimizer = prediction_agent.model.get_optimizer("def_tokens_novelty")
examples = []
prediction_params = []
batch_target = []
novelties = []
loss_fn = torch.nn.MSELoss()

class TrainState(Enum):
    POLICY=1
    PREDICTION=2
    NOVELTY=3

# Start with prediction training in the training loop
train_state = TrainState.PREDICTION

# Keep track of novelty for debugging purposes
running_novelty = 0.

# Random action probability
# TODO Once we begin training a policy network this should more towards 0 as training progresses.
random_likelihood = 1.0

def random_training(predict_tensor, target_tensor, optimizer):
    """Training based upon random actions.

    Trains the DNN to predict the target labels after taking random actions to explore the space.

    Args:
        predict_tensor (torch.Tensor) : The lifetime prediction tensor
        target_tensor (torch.Tensor)  : Target labels
        optimizer (torch.nn.Optimizer): Model optimizer
    """
    pass

def novelty_training(predict_tensor, target_tensor, novelties, running_novelty,
                     optimizer, novelty_optimizer, train_state):
    """Novelty-based training.
    
    Arguments:
        predict_tensor (torch.Tensor) : The lifetime prediction tensor
        target_tensor (torch.Tensor)  : Target labels
        novelties (torch.Tensor)      : Novelty tensor
        running_novelty (float)       : Running novelty (for debugging)
        optimizer (torch.nn.Optimizer): Model optimizer
        train_state (TrainState)      : Training state
    """
    # Alternate between training the lifetime prediction and novelty networks and
    # training the policy network.
    if TrainState.POLICY == train_state:
        # Policy network
        # TODO The policy network really should not be shown too many examples from
        # the same match up, only one example from a matchup should be shown during
        # a batch.

        # Traditional loss:
        # err = loss_fn(predict_tensor, target_tensor)
        # No gradient update for the predictor, just train the policy based upon the
        # eventual lifetime

        # Update the policy network
        optimizer.zero_grad()
        #loss = something
        loss.abs().sum().backward()
        optimizer.step()

        # Next loop train prediction
        train_state = TrainState.PREDICTION
    elif TrainState.PREDICTION == train_state:
        # Train the lifetime prediction network

        # Life time prediction
        # The output cannot be negative, run through a ReLU to clean that up
        f = torch.nn.ReLU()
        epsilon = 0.001
        predictions = f(predict_tensor[0]) + epsilon
        # Normal distribution:
        #normal = torch.distributions.normal.Normal(predict_tensor[0], predict_tensor[1])
        #loss = -normal.log_prob(target_tensor)
        # Poisson distribution (works well)
        poisson = torch.distributions.poisson.Poisson(predictions)
        loss = -poisson.log_prob(target_tensor)
        # Binomial distribution (works poorly)
        #binomial = torch.distributions.binomial.Binomial(predict_tensor[0], predict_tensor[1])
        #loss = -binomial.log_prob(target_tensor)
        # Geometric distribution (works poorly)
        #geometric = torch.distributions.geometric.Geometric(predict_tensor[0].abs())
        #loss = -geometric.log_prob(target_tensor)
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        # Next loop train novelty
        train_state = TrainState.NOVELTY

    elif TrainState.NOVELTY == train_state:
        # Train the lifetime prediction and novelty networks
        # TODO FIXME HERE Why are the novelties dimensionality 29? Isn't novelty 1D?
        novelties_tensor = torch.stack(tuple(novelties)).to(prediction_agent.device)
        # TODO FIXME HERE Novelty keeps going up forever! Argh!

        # Novelty prediction
        # Update policy network to push it into higher novelty. Also update the
        # novelty network to recognize this state.
        # Higher novelty trends towards 0 loss for the policy network.
        # The novelty knob sets an absolute maximum penalty and should probably
        # be adjusted over the course of training.
        novelty_knob = 0.01
        novelty_penalty = 1.0 / (novelties_tensor.sum() + novelty_knob)
        loss = novelty_penalty
        # The novelty penalty should be used to update the policy network
        # Lifetime prediction
        # Update the observer network using the novelty penalty
        novelty_optimizer.zero_grad()
        loss.sum().backward()
        #
        #novelties_tensor.sum().backward()

        novelty_optimizer.step()

        # Weighted average of running novelty (for logging)
        with torch.no_grad():
            running_novelty = 0.1 * novelties_tensor.sum() + 0.9 * running_novelty

        # Next loop train prediction (skipping policy for now)
        train_state = TrainState.PREDICTION


# Set up logging to track what happens during training.
logging.basicConfig(filename='training.log',level=logging.WARN)

print("Collecting examples.")
while len(examples) < args.simruns:
    attack_range = random.choice(ArmadaTypes.ranges)
    attack_hull = random.choice(ArmadaTypes.hull_zones)
    defend_hull = random.choice(ArmadaTypes.hull_zones)
    attacker_name = random.choice(ship_names)
    defender_name = random.choice(ship_names)

    # TODO FIXME HERE it looks like if a ship is re-used it does not get its hull and shields back!
    attacker = ship.Ship(name=attacker_name, template=ship_templates[attacker_name], upgrades=[], player_number=1)
    defender = ship.Ship(name=defender_name, template=ship_templates[defender_name], upgrades=[], player_number=2)

    # Make sure we are actually rolling dice
    colors, _ = attacker.roll(attack_hull, attack_range)
    if 0 < len(colors):
        logging.info("{} vs {} at range {}".format(attacker_name, defender_name, attack_range))
        world_state = WorldState()
        world_state.addShip(attacker, 0)
        world_state.addShip(defender, 1)
        num_rolls = 0
        # This will hold lists of the state action pairs for each roll
        state_actions = []
        while 0 < defender.hull():
            num_rolls += 1
            # Handle the attack and receive the updated world state
            # In this initial version of the code the prediction agent won't actually take any
            # actions but we need it to log the (attack_state, action) pairs
            prediction_agent.rememberStateActions()
            # TOOD FIXME HERE For random training sometimes the agents should be random agents.
            world_state = handleAttack(world_state=world_state, attacker=(attacker, attack_hull),
                                       defender=(defender, defend_hull), attack_range=attack_range,
                                       offensive_agent=prediction_agent,
                                       defensive_agent=prediction_agent)
            # Get the (state, action) pairs back
            state_action_pairs = prediction_agent.returnStateActions()
            state_actions.append(state_action_pairs)
            if args.novelty:
                logging.info("\t roll {}, estimate {}, novelty {}".format(num_rolls,
                    state_action_pairs[0][2][-2], state_action_pairs[0][3]))
            else:
                logging.info("\t roll {}, estimate {}".format(num_rolls,
                    state_action_pairs[0][2][-2]))
        # Pair the lifetimes with the (state, action) pairs and push them into the examples list
        for roll_idx, state_action_pairs in enumerate(state_actions):
            # Have the prediction terminate at 0 so subtract an additional 1
            lifetime = num_rolls - roll_idx - 1
            for state_action_pair in state_action_pairs:
                examples.append((state_action_pair[0], state_action_pair[1], state_action_pair[2], lifetime))
                prediction_params.append(state_action_pair[2][-2:].view(2, 1))
                batch_target.append(torch.tensor([[lifetime]], dtype=torch.float))
                if args.novelty:
                    novelties.append(state_action_pair[3])

                # Don't use the prediction part of the tensor for novelty, we only want to change
                # the policy

                if 0 == len(examples) % 500:
                    print("Running novelty is {}".format(running_novelty))

                if 0 == len(examples) % 1000:
                    print("{} examples collected.".format(len(examples)))

                if 0 == len(examples) % 32:
                    # Train on the last 32. This isn't a great way to do things since the samples
                    # in each batch will be correlated. This is just a first pass.
                    # Grab the last 2 output for the mean and variance prediction
                    # TODO Take mean and variance out of the policy network
                    predict_tensor = torch.cat(tuple(prediction_params), 1).to(prediction_agent.device)
                    target_tensor = torch.cat(tuple(batch_target), 1).to(prediction_agent.device)

                    if args.novelty:
                        # Incorporate the novelty into the loss of the model. We will use the inverse
                        # square root of the mean absolute novelty.
                        # A gradient exists from these projections back to the policy model through the
                        # policy's output vector
                        # Probably more complicated than is necessary.
                        novelty_training(predict_tensor, target_tensor, novelties, running_novelty,
                                         optimizer, novelty_optimizer, train_state)
                    else:
                        # Explore the space with random actions.
                        random_training(predict_tensor, target_tensor, optimizer)

                    # Clear for the next batch
                    prediction_params = []
                    batch_target = []
                    novelties = []

# Let's take a look at the last few predictions
abs_error = 0
for example in examples[-300:]:
    print("predicted {} actually {}".format(example[2][-2], example[3]))
    abs_error += abs(example[2][-2] - example[3])
    if 0 == example[3]:
        print("Final state was {}".format(example[0]))

print("average absolute error: {}".format(abs_error/300.0))

# Save the model
prediction_agent.model.save(args.filename)
