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
        self.deterministic = deterministic

        # Variables for data generation
        self.randagent = RandomAgent()
        keys, ship_templates = parseShips('data/test_ships.csv')
     
        training_ships = ["All Defense Tokens", "All Defense Tokens",
                          "Imperial II-class Star Destroyer", "MC80 Command Cruiser",
                          "Assault Frigate Mark II A", "No Shield Ship", "One Shield Ship",
                          "Mega Die Ship"]
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
                desired_samples += self.num_samples - desired_samples * worker_info.num_workers
            if self.deterministic:
                torch.manual_seed(worker_info.id)
                random.seed(worker_info.id)
            else:
                random.seed(worker_info.seed)
        # Always use the same seed for each worker if deterministic mode is enabled.
        elif self.deterministic:
            torch.manual_seed(16)
            random.seed(16)
        # Generate the training data
        samples = get_n_examples(n_examples=desired_samples, ship_a=self.attackers,
                                 ship_b=self.defenders, agent=self.randagent)
        # This will collect just one example per scenario, which should ensure no bias in the
        # samples that would result in biased learning.
        num_collected = 0
        # TODO FIXME HERE Yield an iterator, not the samples
        return collect_attack_batches(batch_size=self.batch_size, attacks=samples, subphase=self.subphase)
