import random

import numpy as np


# Generates X and Y dummy data based on a normal distribution
def normal_dataset(mean: float, stddev: float, size: int) -> np.array:
    mean_shift = random.randint(0, 100)
    stddev_shift = random.randint(0, 100)
    x = np.random.normal(mean, stddev, size)
    y = np.random.normal(mean + mean_shift, stddev + stddev_shift, size)
    return x, y
