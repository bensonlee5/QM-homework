from abc import ABC, abstractmethod
import numpy as np
from itertools import cycle
from typing import Union
import pandas as pd
import random


class Baker(ABC):
    @abstractmethod
    def name(self):
        """
        Accessor for the name of the baker (e.g. - John or Jane)
        """
        pass

    @abstractmethod
    def choose_item_to_make(self, items_available: dict, **kwargs):
        """
        :param items_available: dictionary where keys are ids and values are tuples containing (time, reward)
        :return: key corresponding to item chosen to make
        """
        pass


class AffirmBaker(Baker):
    item_list = list()

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    def choose_item_to_make(self, items_available: dict, **kwargs) -> int:
        """
        Chooses item that maximizes reward per unit time

        :param items_available: dictionary of objects to process {key: (time, reward)}
        :param kwargs: additional keyword arguments (as you see fit)
        :return: key corresponding to item chosen to make
        """
        max_reward_per_time = -np.inf
        item_chosen = None

        for k, v in items_available.items():
            reward_per_time = float(v[1]) / float(v[0])

            if max_reward_per_time < reward_per_time:
                item_chosen = k
                max_reward_per_time = reward_per_time

        self.item_list.append(items_available[item_chosen])
        return item_chosen


class CandidateBaker(Baker):
    item_list = list()

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    def choose_item_to_make(self, items_available: dict, **kwargs) -> int:
        """
        Chooses item with highest reward.

        :param items_available: dictionary of objects to process {key: (time, reward)}
        :param kwargs: additional keyword arguments (as you see fit)
        :return: key corresponding to item chosen to make
        """
        max_reward = -np.inf
        item_chosen = None

        for k, v in items_available.items():
            if max_reward < v[1]:
                item_chosen = k
                max_reward = v[1]

        self.item_list.append(items_available[item_chosen])
        return item_chosen


class Bakery:
    items_per_day = 10
    items_available = dict()
    baker_scoreboard = dict()

    def __init__(self, baker: Union[Baker, list]):
        """
        Initializes scoreboard

        :param baker: Baker instance or list of Baker instances each containing a strategy
        """

        self.baker = [baker] if isinstance(baker, Baker) else baker
        for b in self.baker:
            self.baker_scoreboard[b.name] = 0

    def generate_batch(self):
        """
        Generate batch of potential bakery items to make
        """

        self.items_available.clear()

        time_to_bake = np.random.uniform(1, 4, size=self.items_per_day)
        reward = np.random.uniform(1, 10, size=self.items_per_day)

        self.items_available = dict(zip(range(self.items_per_day),
                                        list(zip(time_to_bake, reward))))

    def allocate_production(self):
        """
        Allocate choices to each baker based on their strategy
        """

        # Randomly pick baker order
        random.shuffle(self.baker)
        baker_cycle = cycle(self.baker)

        for _ in range(len(self.items_available)):
            baker_to_pick = next(baker_cycle)
            item_chosen = baker_to_pick.choose_item_to_make(self.items_available)
            del self.items_available[item_chosen]

    def run_bakery(self, time_to_run: int) -> dict:
        """
        Run production cycle

        :param time_to_run: how long to run production cycle for
        :return: returns dictionary containing scoreboard
        """

        for baker in self.baker:
            queue_idx = 0
            time_remaining = time_to_run

            while time_remaining > 0:
                next_item_to_bake = baker.item_list[queue_idx]
                if time_remaining > next_item_to_bake[0]:
                    time_remaining -= next_item_to_bake[0]
                    self.baker_scoreboard[baker.name] += next_item_to_bake[1]
                else:
                    break

                ++queue_idx

        return self.baker_scoreboard

    def print_score(self):
        """
        Prints final score in competition between all contestants
        """
        score = pd.DataFrame(data={'names': list(self.baker_scoreboard.keys()),
                                   'scores': list(self.baker_scoreboard.values())})
        score.sort_values(by='names', inplace=True)
        print(score)


def run_bakery():
    num_days = 10
    time_per_day = 20

    bot_aff = AffirmBaker("Affirm")
    bot_candidate = CandidateBaker("Candidate")

    bakery_ma = Bakery([bot_aff, bot_candidate])

    for _ in range(num_days):
        bakery_ma.generate_batch()
        bakery_ma.allocate_production()
        bakery_ma.run_bakery(time_per_day)

    bakery_ma.print_score()


if __name__ == "__main__":
    run_bakery()
