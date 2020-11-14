#
# Copyright Bernhard Firner, 2019-2020
#
# Learning model that predicts actions based upon a one-hot encoding of the current world state.
# 

import itertools
import torch
from torch import (nn)

from armada_encodings import (Encodings)
from learning_agent import (LearningAgent)


class SeparatePhaseModel(nn.Module):
    """A class with a different model for each phase.

    A more sophisticated model would have shared weights between different phases, so this model
    should just be used for simple evaluations and experiments.
    """

    def __init__(self):
        # There is a different model for each phase.
        # Each model takes as input the world state and the attack state and output is the action

        world_size = Encodings.calculateWorldStateSize()
        attack_size = Encodings.calculateAttackSize()

        nn.Module.__init__(self)
        self.models = nn.ModuleDict()
        self.optimizers = {}

        # TODO FIXME Loop through all of the possible phases instead of just these two
        for phase_name in ["attack - resolve attack effects", "attack - spend defense tokens"]:
            action_size = Encodings.calculateActionSize(phase_name)
            self.models[phase_name] = self.init_fc_params(world_size + attack_size, action_size)
            self.optimizers[phase_name] = torch.optim.Adam(self.models[phase_name].parameters())

        self.sm = nn.Softmax()

    def get_optimizer(self, phase_name):
        """
        Gets an optimizer for one of the models.

        Args:
            phase_name (str): The name of the phase
        Returns:
            torch.optim
        """
        return self.optimizers[phase_name]

    def load(self, filename):
        """
        Loads parameters from an existing file with model weights.

        Args:
            filename (str): Path to a file with model weights
        Raises:
            RuntimeError: If the file does not exist or has weight that that don't match this model.
        """
        params = torch.load(filename)
        for phase_name in self.optimizers.keys():
            self.models[phase_name].load_state_dict(params[phase_name])
            self.optimizers[phase_name].load_state_dict(params[f"{phase_name}_optimizer"])

    def save(self, filename):
        """
        Saves parameters from the current models into the provided filename.

        Args:
            filename (str): Path to the file.
        """
        # TODO clean up
        states = {}
        for phase_name in self.optimizers.keys():
            states[phase_name] = self.models[phase_name].state_dict()
            states[f"{phase_name}_optimizer"] = self.optimizers[phase_name].state_dict()

        torch.save(states, filename)

    def forward(self, phase_name, encoding):
        """
        Forwards the encoding through the model.

        Args:
            phase_name (str): The name of the model (.e.g. "def_tokens")
            encoding (torch.tensor): An encoding of the current state
        Returns:
            action encoding
        """
        # Forward and return the results. The calling module will need to interpret them (e.g. by
        # rounding off or finding a most likely choice)
        x = self.models[phase_name].forward(encoding)
        return x

    def init_fc_params(self, input_size, output_size):
        """
        Initialize fully connected parameters for the given input and output sizes

        Args:
            filename (str): Path to a file with model weights
        Raises:
            RuntimeError: If the file does not exist or has weight that that don't match this model.
        """
        # Basic linear model
        layers = nn.Sequential(
                torch.nn.BatchNorm1d(input_size),
                nn.Linear(input_size, 2 * input_size),
                nn.ReLU(),
                nn.Linear(2 * input_size, 4 * output_size),
                nn.ReLU(),
                nn.Linear(4 * output_size, 2 * output_size),
                nn.ReLU(),
                nn.Linear(2 * output_size, output_size),
                )
        return layers

class ArmadaModel(nn.Module):

    def __init__(self, with_novelty=True):
        # An extra two outputs
        # are added in for the mean and variance estimation.

        nn.Module.__init__(self)
        self.with_novelty = with_novelty
        def_in = Encodings.calculateAttackSize()
        def_out = Encodings.calculateSpendDefenseTokensSize() + 2
        self.models = nn.ModuleDict()
        self.models["def_tokens"] = self.init_fc_params(def_in, def_out)
        self.optimizers = {
            "def_tokens": torch.optim.Adam(self.models["def_tokens"].parameters())
        }
        if self.with_novelty:
            # We will also create two more models to use for random distillation.  The first random
            # model will remain static and the second will be learn to predict the outputs of the
            # first. The difference between the two outputs will be used to estimate the novelty of
            # the current state. If the state is new then the second model will not be able to make
            # a good prediction of the first model's outputs.  In other words, the first model
            # projects the inputs into a new latent space. The ability of the second model to
            # predict the projection into the latent space should be correlated to how similar this
            # state is to ones we have previously visited.

            # The novelty network is a clone of the corresponding network
            self.models["def_tokens_novelty"] = nn.ModuleList([self.init_fc_params(def_in, def_out),
                                                              self.models["def_tokens"]])
            self.models["def_tokens_static"] = self.init_fc_params(def_in, def_out)
            self.optimizers["def_tokens_novelty"] = torch.optim.Adam(
                self.models["def_tokens_novelty"].parameters())

        self.sm = nn.Softmax()

    def get_optimizer(self, model_name):
        """
        Gets an optimizer for one of the models.

        Args:
            model_name (str): The name of the model (.e.g. "def_tokens")
        Returns:
            torch.optim
        """
        return self.optimizers[model_name]

    def load(self, filename):
        """
        Loads parameters from an existing file with model weights.

        Args:
            filename (str): Path to a file with model weights
        Raises:
            RuntimeError: If the file does not exist or has weight that that don't match this model.
        """
        params = torch.load(filename)
        self.models["def_tokens"].load_state_dict(params["def_tokens"])
        self.optimizers["def_tokens"].load_state_dict(params["def_tokens_optimizer"])
        if self.with_novelty:
            self.models["def_tokens_novelty"].load_state_dict(params["def_tokens_novelty"])
            self.optimizers["def_tokens_novelty"].load_state_dict(params["def_tokens_novelty_optimizer"])
            self.models["def_tokens_static"].load_state_dict(params["def_tokens_static"])

    def save(self, filename):
        """
        Saves parameters from the current models into the provided filename.

        Args:
            filename (str): Path to the file.
        """
        # TODO clean up
        if not self.with_novelty:
            torch.save({
                "def_tokens": self.models["def_tokens"].state_dict(),
                "def_tokens_optimizer": self.optimizers["def_tokens"].state_dict()
                },
                filename
            )
        else:
            torch.save({
                "def_tokens": self.models["def_tokens"].state_dict(),
                "def_tokens_optimizer": self.optimizers["def_tokens"].state_dict(),
                "def_tokens_novelty": self.models["def_tokens_novelty"].state_dict(),
                "def_tokens_novelty_optimizer": self.optimizers["def_tokens_novelty"].state_dict(),
                "def_tokens_static": self.models["def_tokens_static"].state_dict()
                },
                filename
            )

    def forward(self, model_name, encoding):
        """
        Forwards the encoding through the model.

        Args:
            model_name (str): The name of the model (.e.g. "def_tokens")
            encoding (torch.tensor): An encoding of the attack state.
        Returns:
            Tuple(best action, survival mean, survival variance)
        """
        # Forward and return the results. The calling module will need to interpret them (e.g. by
        # rounding off or finding a most likely choice)
        if not self.with_novelty:
            x = self.models[model_name].forward(encoding)
            return x
        else:
            x = self.models[model_name].forward(encoding)
            # Project the observed states
            latent_projection = self.models["def_tokens_static"].forward(encoding)
            # Check the novelty
            predicted_projection = self.models["def_tokens_novelty"][1].forward(encoding)

            novelty = (predicted_projection - latent_projection.detach()).abs()
            return x, novelty


    def init_fc_params(self, input_size, output_size):
        """
        Initialize fully connected parameters for the given input and output sizes

        Args:
            filename (str): Path to a file with model weights
        Raises:
            RuntimeError: If the file does not exist or has weight that that don't match this model.
        """
        # Basic three layer setup
        # If we wanted to guarantee that all outputs are correct then we could apply softmax to
        # different subsets of the output and then round the numeric outputs to guarantee only whole
        # numbers. As a first pass though, we will see if rounding suffices.
        layers = nn.Sequential(
                nn.Linear(input_size, 2 * input_size),
                #nn.BatchNorm1d(2 * input_size),
                #nn.ReLU(),
                nn.Linear(2 * input_size, 4 * output_size),
                #nn.BatchNorm1d(4 * output_size),
                nn.ReLU(),
                nn.Linear(4 * output_size, output_size),
                )
        return layers
