#
# Copyright Bernhard Firner, 2019
#
import random

from ship import (Ship)
from game_constants import (ArmadaPhases, ArmadaTypes)
from learning_agent import (LearningAgent)

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


# Seed with time or a local source of randomness
random.seed()

keys, ship_templates = utility.parseShips('data/armada-ship-stats.csv')

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
#   Choose actions based upon policy network + some randomness
#   Record the scenario into a pool
# Backprop on a random subset of the (state, action, reward, new state) tuples
# The goal is to train a network such that if it is given a state it predicts an action that
# maximizes the reword (in this case the lifetime of the defending ship). To train we take the
# network to produce an action based upon the current state
#
# Step 1:
# Just train a lifetime predictor. There is a lot of randomness in dice rolls of course, so our goal
# will be to train a density model. This means that we will train the network to estimate the
# parameters of the distribution that describes the lifetime. In plainer words, we will train the
# network to predict the average lifetime of the vessel and the undertainty of that prediction.
# We are rolling three kinds of dice, so in reality this is the combination of three probability
# distributions, but to simplify things we will estimate it as a single normal.
# The loss will be the negative log of the probability of an outcome given a predicted mean and
# standard deviation. This works because as the probability approaches zero the negative log of it
# approaches infinity. So if the outputs the mean and sigma the loss will be:
# > d = torch.distributions(mean, sigma)
# > loss = -d.log_prob(result)

# Create a learning agent. This will initial the model.
prediction_agent = LearningAgent(ArmadaModel())

# Load a previously trained model for additional training
if os.path.isfile("defense_token_model.checkpoint"):
    prediction_agent.model.load("defense_token_model.checkpoint")

optimizer = torch.optim.Adam(prediction_agent.model.parameters())
examples = []
batch_out = []
batch_target = []
loss_fn = torch.nn.MSELoss()

# Set up logging to track what happens during training.
logging.basicConfig(filename='training.log',level=logging.ERROR)

print("Collecting examples.")
while len(examples) < 1000:
    attack_range = random.choice(ArmadaTypes.ranges)
    attack_hull = random.choice(ArmadaTypes.hull_zones)
    defend_hull = random.choice(ArmadaTypes.hull_zones)
    attacker_name = random.choice(ship_names)
    defender_name = random.choice(ship_names)

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
            world_state = handleAttack(world_state=world_state, attacker=(attacker, attack_hull),
                                       defender=(defender, defend_hull), attack_range=attack_range,
                                       offensive_agent=prediction_agent,
                                       defensive_agent=prediction_agent)
            # Get the (state, action) pairs back
            state_action_pairs = prediction_agent.returnStateActions()
            state_actions.append(state_action_pairs)
        # Pair the lifetimes with the (state, action) pairs and push them into the examples list
        for roll_idx, state_action_pairs in enumerate(state_actions):
            # Have the prediction terminate at 0 so subtract an additional 1
            lifetime = num_rolls - roll_idx - 1
            for state_action_pair in state_action_pairs:
                examples.append((state_action_pair[0], state_action_pair[1], lifetime))
                batch_out.append(state_action_pair[1][-2:].view(2, 1))
                batch_target.append(torch.tensor([[lifetime]], dtype=torch.float))

                if 0 == len(examples) % 1000:
                    print("{} examples collected.".format(len(examples)))

                if 0 == len(examples) % 32:
                    # Train on the last 32. This isn't a great way to do things since the samples
                    # in each batch will be biased. This is just a first pass.
                    # Grab the last 2 output for the mean and variance prediction
                    out_tensor = torch.cat(tuple(batch_out), 1).to(prediction_agent.device)
                    target_tensor = torch.cat(tuple(batch_target), 1).to(prediction_agent.device)
                    # Traditional loss:
                    # err = loss_fn(out_tensor, target_tensor)
                    # Normal distribution:
                    #normal = torch.distributions.normal.Normal(out_tensor[0], out_tensor[1])
                    #loss = -normal.log_prob(target_tensor)
                    # Poisson distribution
                    poisson = torch.distributions.poisson.Poisson(out_tensor[0])
                    loss = -poisson.log_prob(target_tensor)
                    optimizer.zero_grad()
                    loss.abs().sum().backward()
                    optimizer.step()
                    # Clear for the next batch
                    batch_out = []
                    batch_target = []

# Let's take a look at the last few predictions
abs_error = 0
for example in examples[-300:]:
    print("predicted {} actually {}".format(example[1][-2], example[2]))
    abs_error += abs(example[1][-2] - example[2])

print("average absolute error: {}".format(abs_error/300.0))

# Save the model
prediction_agent.model.save("defense_token_model.checkpoint")
