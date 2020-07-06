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


    def __init__(self, subphase, num_samples, deterministic=False):
        """Dataset for random actions.

        Arguments:
            subphase      (str): Attack subphase to collect. TODO FIXME Cover more than just attacks
            num_samples   (int): Approximate number of samples to gather.
            deterministic(bool): Make each iteration produce the same results.
        """
        super(RandomActionDataset).__init__()
        self.num_samples = num_samples
        self.subphase = subphase
        self.input_size = Encodings.calculateActionSize(self.subphase) + Encodings.calculateWorldStateSize()
        self.deterministic = deterministic

        # Variables for data generation
        self.randagent = RandomAgent()
        keys, ship_templates = parseShips('data/test_ships.csv')

        self.ship_a = Ship(name="Ship A", template=ship_templates["All Defense Tokens"],
                           upgrades=[], player_number=1, device='cpu')
        self.ship_b = Ship(name="Ship B", template=ship_templates["All Defense Tokens"],
                           upgrades=[], player_number=2, device='cpu')
        # We'll generate the samples in the iterator function so that it is done in parallel.
        # TODO FIXME If deterministic is true then generate everything a single time here to save time.


    def __iter__(self):
        """Get a data iterator."""
        worker_info = torch.utils.data.get_worker_info()
        iter_begin = 0
        iter_end = self.num_samples
        # Different sampling behavior for single thread and multi-thread data loading
        if worker_info is not None:
            iter_end = int(self.num_samples / worker_info.num_workers)
            if (worker_info.id + 1) <= self.num_samples % worker_info.num_workers:
                iter_end += 1
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
        # TODO FIXME HERE Modify get_n_examples to separate each attack into a subarray. Modify
        # collect_attack_batches to only sample from each attack once.
        samples = get_n_examples(n_examples=iter_end - iter_begin, ship_a=self.ship_a,
                                 ship_b=self.ship_b, agent=self.randagent)
        batch_size = 1
        num_samples = 0
        for sample in samples:
            num_samples += 1
            # This will collect just one example per scenario, which should ensure no bias in the
            # samples that would result in biased learning.
            num_collected = 0
            batch = torch.Tensor(batch_size, self.input_size)
            labels = torch.Tensor(batch_size, 1)
            for _ in collect_attack_batches(batch, labels, sample, self.subphase):
                num_collected += 1
                yield([batch[0], labels[0]])
