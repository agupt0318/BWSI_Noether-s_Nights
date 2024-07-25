# Algorithms taken from Appendix C of the paper A Variational Quantum Attack for AES-like
# Symmetric Cryptography (Wang et al., 2022)
from typing import Callable

import numpy as np

from classical.util import bitstring_10, bits_to_string


class Optimizer:
    def __init__(self):
        self.guess_history = []
        self.cost_history = []
        self.key_history = []

    def step(self) -> bool:
        pass


class GradientDescent(Optimizer):
    def __init__(
            self,
            initial_guess: np.ndarray,
            cost_function: Callable[[np.ndarray], tuple[float, bitstring_10]],
            learning_rate: float,
            cost_cutoff: float
    ):
        super().__init__()

        self.guess = initial_guess
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.cost_cutoff = cost_cutoff

        self.adaptive_factor = 0
        self.num_dimensions = len(initial_guess)
        # noinspection PyTypeChecker
        self.best_guess: tuple[np.ndarray, float] = (None, float('inf'))

    def calculate_gradient_at_point(
            self,
            point: np.ndarray,
            cost_at_point: float
    ):
        epsilon = 0.01  # small step size for calculating the approximate gradient

        gradient = np.zeros(self.num_dimensions)
        for i in range(self.num_dimensions):
            # Calculate a point a small step away in the ith direction
            adjusted_point = point.copy()
            adjusted_point[i] += epsilon
            # Approximate the gradient in the ith direction using the difference quotient
            cost_at_adjusted_point, _ = self.cost_function(adjusted_point)
            gradient[i] = (cost_at_adjusted_point - cost_at_point) / epsilon

        return gradient

    def step(self) -> bool:

        self.adaptive_factor += 1

        cost_at_current_guess, key_for_current_guess = self.cost_function(self.guess)
        print(f'Current guess: {np.round(self.guess, 2)}, key: {bits_to_string(key_for_current_guess)}')

        self.guess_history.append(self.guess.copy())
        self.cost_history.append(cost_at_current_guess)
        self.key_history.append(key_for_current_guess)

        # If the cost is strictly less than the cutoff, stop
        if cost_at_current_guess < self.cost_cutoff:
            return True
        # Otherwise, update the current best guess
        elif cost_at_current_guess < self.best_guess[1]:
            self.best_guess = (self.guess.copy(), cost_at_current_guess)

        gradient = self.calculate_gradient_at_point(self.guess, cost_at_current_guess)

        # Calculate the adaptive step size
        step_size = self.learning_rate / abs(cost_at_current_guess + self.cost_cutoff) \
                    + np.log(self.adaptive_factor) / self.adaptive_factor * np.random.uniform(0, 1)

        # If the gradient is too low, generate a new random guess
        if sum(gradient ** 2) ** 0.5 < 0.8:
            self.guess = np.random.uniform(-1, 1, self.num_dimensions)
        # Otherwise Update the guess based on the gradient
        else:
            self.guess -= gradient * step_size

        return False


class N_M_Optimizer(Optimizer):
    def __init__(self, cost_function, guess, alpha, cost_cutoff):
        super().__init__()

        self.cost_function = cost_function
        self.guess = guess
        self.alpha = alpha
        self.cost_cutoff = cost_cutoff

        self.N = len(guess)  # N dimensions of guess
        self.points = [guess]  # list of x1--xN values

        self.calc_xi_component(0)

        self.times = self.N + 1

    def calc_xi_component(self, step: int):  # step should be 0 or 1
        xi = self.guess.copy()
        for i in range(self.N):
            if self.guess[i] == 0:
                xi[i] = 0.8
            else:
                xi[i] = self.guess[i] * self.alpha
            self.points[i + step] = xi

    def step(self) -> bool:
        self.points.sort(key=lambda x: self.cost_function(x))
        if self.cost_function(self.points[0]) <= self.cost_cutoff:
            return True
        if self.cost_function(self.points[-1]) - self.cost_function(self.points[1]) < 0.15:
            self.calc_xi_component(1)
            return False

        # calculate average of first N points m
        m = np.mean(self.points[:-1], axis=0)
        # calculate reflect point r
        r = 2 * m - self.points[-1]
        self.times += 1

        if self.cost_function(self.points[0]) <= self.cost_function(r) < self.cost_function(self.points[-2]):
            self.points[-1] = r
            return False

        if self.cost_function(r) < self.cost_function(self.points[0]):
            # calculate expand point s
            s = m + 2 * (m - self.points[-1])
            self.times += 1
            if self.cost_function(s) < self.cost_function(r):
                self.points[-1] = s
                return False
            else:
                self.points[-1] = r
                return False

        if self.cost_function(self.points[-2]) <= self.cost_function(r) < self.cost_function(self.points[-1]):
            c1 = m + (r - m) / 2
            self.times += 1
            if self.cost_function(c1) < self.cost_function(r):
                self.points[-1] = c1
                return False
            else:
                for i in range(1, self.N + 1):
                    self.points[i] = self.guess + (self.points[i] - self.guess) / 2.0
                self.times += self.N
                return False
        if self.cost_function(self.points[-1]) <= self.cost_function(r):
            c2 = m + (self.points[-1] - m) / 2.0
            self.times += 1
            if self.cost_function(c2) < self.cost_function(self.points[-1]):
                self.points[-1] = c2
                return False
            else:
                for i in range(1, self.N + 1):
                    self.points[i] = self.guess + (self.points[i] - self.guess) / 2.0
                self.times += self.N
                return False

        return False
