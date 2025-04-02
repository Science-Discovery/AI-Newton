import numpy as np
from typing import Tuple

cpuct = 5.0
alpha = 0.1             # 以一个较小的时间周期 decay
alpha_count = 0.001     # count 代表执行操作的次数，以一个较大的时间周期 decay
exp_ucb_discount = 0.1
exp_ucb_alpha = 1.0


class ExperimentUCB:
    def __init__(self, info: str = None):
        self.info: str | None = info
        self.value: float = 0.0
        self.count: float = 0.0

    def reset(self, for_value: float = 0.,
              for_count: float = 0.):
        self.value = for_value
        self.count = for_count

    def update(self, reward: float | None):
        if reward is None:
            self.count = self.count
        else:
            self.value = self.value * exp_ucb_discount + reward
            self.count = self.count + 1

    def ucb(self) -> float:
        return exp_ucb_alpha * self.value + np.sqrt(1.0 / (1 + self.count))

    def to_json(self) -> Tuple[float, float, str | None]:
        return (self.value, self.count, self.info)

    @staticmethod
    def from_json(data: Tuple[float, float, str | None]) -> "ExperimentUCB":
        obj = object.__new__(ExperimentUCB)
        value, count, info = data
        obj.value = value
        obj.count = count
        obj.info = info
        return obj
