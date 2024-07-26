import numpy as np
from numpy import ndarray
import unittest
from Optimization import GradientDescent
from typing import (Any, Callable, Dict, Optional, Union, Iterator)

type cost_function_t[Data] = Callable[[ndarray], OptimizerGuess[Data]]
type testFunction = Callable[[ndarray], float]


class OptimizerGuess[Data]:
    """
    A class representing an optimizer guess. It contains the guessed point, the cost function at the point, and some
    data for the guess, and can be partially ordered
    """

    def __init__(self, point: ndarray, cost: float, data: Data):
        self.point = point.copy()
        self.cost = cost
        self.data = data

    # Functions to allow comparison of guesses
    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return self.cost == other.cost and np.array_equal(self.point, other.point)

    # Following functions are implemented in terms of __lt__ and __eq__
    def __gt__(self, other):
        return other < self

    def __le__(self, other):
        return not (self > other)

    def __ge__(self, other):
        return not (self < other)

    def __ne__(self, other):
        return not (self == other)


def rosenbrock(x) -> float:
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


class test_Optimization(unittest.TestCase):
    @staticmethod
    def create_cost_function(func: testFunction) -> cost_function_t[None]:
        return lambda x: OptimizerGuess(x.copy(), func(x), None)

    def test_gradient_descent(self):
        s = 10
        ITERATION_COUNT = 200
        test_function = rosenbrock
        optimizer = GradientDescent(maxiter=ITERATION_COUNT, learning_rate=1.08)
        initial_point = np.random.uniform(-s, s, 2),
        x_opt, fx_opt, nfevs = optimizer.optimize(2, test_Optimization.create_cost_function(test_function),
                                                  initial_point)

        print(f"Found minimum {x_opt} at a value of {fx_opt} using {nfevs} evaluations.")
