import abc
from typing import Any, Type, TypeVar

import numpy as np
import numpy.typing as npt


class AbstractFeaturesGenerator(metaclass=abc.ABCMeta):
    # class modeling generators for user state
    def __init__(self, num_features: int) -> None:
        self.num_features = num_features
        pass

    @abc.abstractmethod
    def __call__(*args: Any, **kwds: Any) -> npt.NDArray[np.float64]:
        """Generate a user state

        Returns:
            npt.NDArray[np.float64]: user state
        """
        pass


class NormalUserFeaturesGenerator(AbstractFeaturesGenerator):
    """Normal distribution user state generator"""

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, num_features: int) -> npt.NDArray[np.float64]:
        return np.random.normal(self.mean, self.std, num_features)


class UniformFeaturesGenerator(AbstractFeaturesGenerator):
    """Uniform distribution user state generator"""

    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, num_features: int):
        return np.random.uniform(self.min_val, self.max_val, num_features).astype(
            np.float32
        )

    import numpy as np


class BinaryFeaturesGenerator:
    """Binary distribution user state generator"""

    def __init__(self):
        pass

    def __call__(self, num_features: int):
        if num_features <= 0:
            raise ValueError("num_features must be a positive integer")

        # Generate a random number of 1's between 0 and num_features
        num_ones = np.random.randint(0, num_features + 1)
        # num_ones = 3

        # Create a user vector with random 1's and 0's
        user_vector = np.zeros(num_features, dtype=np.float32)
        if num_ones > 0:
            # Randomly select the positions for 1's
            one_positions = np.random.choice(num_features, num_ones, replace=False)
            user_vector[one_positions] = 1

        return user_vector
