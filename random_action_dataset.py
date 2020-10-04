# Copyright Bernhard Firner, 2020

import random
import torch

from armada_encodings import Encodings
from game_constants import (ArmadaTypes)
from game_engine import handleAttack
from learning_components import (a_vs_b, collect_attack_batches, get_n_examples)
from random_agent import (RandomAgent)
from ship import Ship
from utility import parseShips
from world_state import (WorldState)


class RandomActionDataset(torch.utils.data.IterableDataset):
    """Dataset to create random actions."""


    def __init__(self, subphase, num_samples, batch_size=32, deterministic=False):
        """Dataset for random actions.

        Arguments:
            subphase      (str): Attack subphase to collect. TODO FIXME Cover more than just attacks
            num_samples   (int): Approximate number of samples to gather.
            deterministic(bool): Make each iteration produce the same results.
        """
        super(RandomActionDataset).__init__()
        self.subphase = subphase
        self.num_samples = num_samples
        self.batch_size = batch_size
        world_size = Encodings.calculateWorldStateSize()
        action_size = Encodings.calculateActionSize(self.subphase)
        attack_size = Encodings.calculateAttackSize()
        self.input_size = world_size + action_size + attack_size
        self.deterministic = deterministic

        # Variables for data generation
        self.randagent = RandomAgent()
        keys, ship_templates = parseShips('data/test_ships.csv')
     
        training_ships = ["All Defense Tokens", "All Defense Tokens",
                          "Imperial II-class Star Destroyer", "MC80 Command Cruiser",
                          "Assault Frigate Mark II A"]
        self.defenders = []
        self.attackers = []
        for name in training_ships:
            self.attackers.append(Ship(name=name, template=ship_templates[name],
                                  upgrades=[], player_number=1, device='cpu'))
        for name in training_ships:
            self.defenders.append(Ship(name=name, template=ship_templates[name],
                                  upgrades=[], player_number=2, device='cpu'))
        # We'll generate the samples in the iterator function so that it is done in parallel.
        # TODO FIXME If deterministic is true then generate everything a single time here to save time.


    def __iter__(self):
        """Get a data iterator."""
        worker_info = torch.utils.data.get_worker_info()
        desired_samples = self.num_samples
        # Different sampling behavior for single thread and multi-thread data loading
        if worker_info is not None:
            desired_samples = self.num_samples // worker_info.num_workers
            # The first worker will fetch any remainder
            if 1 == worker_info.id:
                desired_samples += self.num_samples % desired_samples
            if self.deterministic:
                torch.manual_seed(worker_info.id)
                random.seed(worker_info.id)
            else:
                random.seed(worker_info.seed)
        # Always use the same seed for each worker if deterministic mode is enabled.
        elif self.deterministic:
            torch.manual_seed(16)
            random.seed(16)
        #TODO generate a sample and yield it
        samples = get_n_examples(n_examples=desired_samples, ship_a=self.attackers,
                                 ship_b=self.defenders, agent=self.randagent)
        # This will collect just one example per scenario, which should ensure no bias in the
        # samples that would result in biased learning.
        num_collected = 0
        batch = torch.zeros(self.batch_size, self.input_size)
        labels = torch.zeros(self.batch_size, 1)
        for nsamples in collect_attack_batches(batch, labels, samples[num_collected:], self.subphase):
            num_collected += nsamples
            # Return if there is no more data.
            if 0 == num_collected:
                return
            yield((batch[:nsamples], labels[:nsamples]))
            # Return if the requested amount of data has already been generated.
            if num_collected >= len(samples):
                return
