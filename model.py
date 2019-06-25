#
# Copyright Bernhard Firner, 2019
#
# Learning model that predicts actions based upon a one-hot encoding of the current world state.
# 

import torch
from torch import (nn)

from armada_encodings import (Encodings)
from learning_agent import (LearningAgent)

class ArmadaModel(nn.Module):

    def __init__(self):
        # An extra two outputs
        # are added in for the mean and variance estimation.

        nn.Module.__init__(self)
        def_in = Encodings.calculateAttackSize()
        def_out = Encodings.calculateSpendDefenseTokensSize() + 2
        self.models = nn.ModuleDict()
        self.models["def_tokens"] = self.init_fc_params(def_in, def_out)
        self.optimizers = {
                "def_tokens": torch.optim.Adam(self.models["def_tokens"].parameters())
                }
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

    def save(self, filename):
        """
        Saves parameters from the current models into the provided filename.

        Args:
            filename (str): Path to the file.
        """
        torch.save({
            "def_tokens": self.models["def_tokens"].state_dict(),
            "def_tokens_optimizer": self.optimizers["def_tokens"].state_dict()
            }, filename)

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
        x = self.models[model_name].forward(encoding)
        return x

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
                nn.BatchNorm1d(2 * input_size),
                nn.ReLU(),
                nn.Linear(2 * input_size, 4 * output_size),
                nn.BatchNorm1d(4 * output_size),
                nn.ReLU(),
                nn.Linear(4 * output_size, output_size),
                )
        return layers
