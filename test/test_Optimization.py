import unittest
from typing import Callable

import numpy as np
from numpy import ndarray

from classical.Optimization import GradientDescentOptimizer, cost_function_t, OptimizerGuess

type testFunction = Callable[[ndarray], float]


def rosenbrock(x: ndarray) -> float:
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def booth(x: ndarray) -> float:
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


def himmelblau(x: ndarray) -> float:
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


class test_Optimization(unittest.TestCase):
    @staticmethod
    def create_cost_function(func: testFunction) -> cost_function_t[None]:
        return lambda x: OptimizerGuess(x.copy(), func(x), None)

    def test_optimizer(self, optimizer: GradientDescentOptimizer):
        pass

    def test_gradient_descent(self):
        optimizer = GradientDescentOptimizer(
            cost_function=test_Optimization.create_cost_function(rosenbrock),
            cost_cutoff=0.1,
            initial_point=np.array([5, -2]),
        )

        pass
