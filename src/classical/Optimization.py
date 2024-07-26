# Algorithms taken from Appendix C of the paper A Variational Quantum Attack for AES-like
# Symmetric Cryptography (Wang et al., 2022)
import abc
import math
from typing import Callable, Union

import numpy as np
from numpy import ndarray


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


type cost_function_t[Data] = Callable[[ndarray], OptimizerGuess[Data]]


class Optimizer[Data]:
    __metaclass__ = abc.ABCMeta

    def __init__(self, cost_function: cost_function_t[Data], cost_cutoff: float):
        """
        :param cost_function: The cost function to minimize. It should take in a point and return an OptimizerGuess
        :param cost_cutoff:   The cutoff for the cost below which we accept a guess and stop
        """

        # The cost function that we are to minimize
        self._cost_function = cost_function
        # The cutoff for the cost function below which we accept a guess and stop
        self._cost_cutoff = cost_cutoff

        # The current best point
        self._best_guess: Union[OptimizerGuess, None] = None
        # The history of guesses, used for debug purposes
        self._history: list[OptimizerGuess] = []
        # Whether the optimization algorithm has terminated
        self._finished = False

    def get_best_guess(self):
        return self._best_guess

    def get_history(self):
        return self._history

    def is_finished(self):
        return self._finished

    def step(self) -> bool:
        """
        Runs one step of the optimization algorithm
        :return: Whether the algorithm has terminated
        """

        # If we are already finished, we don't need to do anything
        if self._finished:
            return True

        # Get the next guess
        new_guess = self._next_guess()
        self._history.append(new_guess)

        # print(f'Current point: {np.round(new_guess.point, 2)}, data: {new_guess.data}')

        # Update the current best guess
        if self._best_guess is None or new_guess < self._best_guess:
            self._best_guess = new_guess

        # Check if the new guess satisfies the cost cutoff
        if new_guess.cost < self._cost_cutoff:
            self._finished = True

        return self._finished

    @abc.abstractmethod
    def _next_guess(self) -> OptimizerGuess:
        """
        This method should be overridden by subclasses and represents one step of iteration in the optimization
        algorithm. It should return a new guess, which is hopefully better than the last guess
        :return:
        """
        raise NotImplementedError("Subclasses must implement this method")

    def evaluate_point(self, point: ndarray) -> OptimizerGuess[Data]:
        """
        Evaluates the given point and returns an OptimizerGuess object representing the result
        """
        return self._cost_function(point)


class GradientDescentOptimizer[Data](Optimizer[Data]):
    def __init__(
            self,
            cost_function: cost_function_t[Data],
            cost_cutoff: float,
            initial_point: ndarray,
            learning_rate: float
    ):
        super().__init__(cost_function, cost_cutoff)

        # The current point in gradient descent
        self.current_point = initial_point
        # A constant controlling the step size in gradient descent
        self.learning_rate = learning_rate

        # The dimensionality of the search space
        self.dimensionality = len(initial_point)

    def _calculate_gradient_at_point(
            self,
            point: ndarray,
            cost_at_point: float
    ):
        epsilon = 0.01  # small step size for calculating the approximate gradient

        gradient = np.zeros(self.dimensionality)
        for i in range(self.dimensionality):
            # Calculate a point a small step away in the ith direction
            adjusted_point = point.copy()
            adjusted_point[i] += epsilon
            # Approximate the gradient in the ith direction using the difference quotient
            cost_at_adjusted_point = self.evaluate_point(adjusted_point).cost
            gradient[i] = (cost_at_adjusted_point - cost_at_point) / epsilon

        return gradient

    def _next_guess(self) -> OptimizerGuess:
        guess = self.evaluate_point(self.current_point)

        gradient = self._calculate_gradient_at_point(guess.point, guess.cost)

        try:
            # Calculate the adaptive step size
            step_size = self.learning_rate / abs(guess.cost - self._cost_cutoff)

            # If the gradient is too low, generate a new random guess
            if sum(gradient ** 2) ** 0.5 < 0.8:
                # print('Generated new random guess')
                self.current_point = np.random.uniform(-1, 1, self.dimensionality)
                self.adaptive_factor = 0
            # Otherwise Update the guess based on the gradient
            else:
                self.current_point -= gradient * step_size
        finally:
            return guess


# Nelder-Mead optimization algorithm, as described by Wikipedia
class NelderMeadOptimizer(Optimizer):
    def __init__(
            self,
            cost_function: cost_function_t,
            cost_cutoff: float,
            dimensionality: int,
            random_simplex_generator: Callable[[], list[ndarray]],
            alpha: float = 1,
            gamma: float = 2,  # This is the value used in the paper
            rho: float = 0.5,  # I think this is the value used in the paper
            sigma: float = .5  # TODO: look up the value used
    ):
        super().__init__(cost_function, cost_cutoff)

        # A function that returns a list of N + 1 points, where N is the dimensionality of the search space.
        self.random_simplex_generator = random_simplex_generator
        # The dimensionality of the search space
        self.dimensionality = dimensionality
        # Parameters for the Nelder-Mead optimization algorithm
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma

        # The candidate points and their costs
        self.guesses: list[OptimizerGuess] = self.generate_random_guesses()

        self.volume_history = []

    def generate_random_guesses(self) -> list[OptimizerGuess]:
        points = self.random_simplex_generator()

        # Make sure that points form a simplex
        assert len(points) == self.dimensionality + 1
        for i in points:
            assert len(i) == self.dimensionality

        return [self.evaluate_point(guess) for guess in points]

    def _get_centroid(self, guesses: list[OptimizerGuess]) -> ndarray:
        result = np.zeros(self.dimensionality)
        for guess in guesses:
            result += guess.point
        return result / len(guesses)

    def _next_guess(self) -> OptimizerGuess:
        # DEBUG: simplex volume
        vol = math.log(abs(np.linalg.det(
            np.array([[*g.point, 1] for g in self.guesses])
        )) + 0.000001)
        self.volume_history.append(vol)

        # In the Nelder-Mead optimization algorithm, the idea is to replace the worst guess with a better guess at each
        # step. Ideally, this should allow the simplex formed by the current list of guesses to converge to the minimum.

        # Sort the points x0 ... xn by ascending cost function value. Thus, self.points[0] is the best point (lowest
        # cost function) and self.points[-1] is the worst point (highest cost function)
        self.guesses.sort()
        best = self.guesses[0]
        worst = self.guesses[-1]
        second_worst = self.guesses[-2]

        if worst.cost - best.cost < 0.15:
            # print('Generated new random guess')
            self.guesses = self.generate_random_guesses()
            return best

        # Calculate the centroid (average, "center of mass") of all points other than the current worst point
        centroid_point = self._get_centroid(self.guesses[:-1])

        # Reflect the centroid across the current worst point.
        reflected = self.evaluate_point(centroid_point + self.alpha * (centroid_point - worst.point))

        # If the reflected point is the best point so far,
        if reflected < best:
            # We can consider a point even further in the direction of the reflected point
            expanded = self.evaluate_point(centroid_point + self.gamma * (reflected.point - centroid_point))
            # Replace the worst point with the expanded point or the reflected point, whichever one is better.
            self.guesses[-1] = min(expanded, reflected)

        # Otherwise, if the reflected point is better than the second-worst point (but not better than the best point),
        # replace the worst point with the reflected point
        elif reflected < second_worst:
            self.guesses[-1] = reflected
        else:
            # Take the best of the reflected point and the worst point, call it the contraction base
            contraction_base = min(reflected, self.guesses[-1])

            # Contract. This moves the centroid towards the contraction base
            contracted = self.evaluate_point(centroid_point + self.rho * (contraction_base.point - centroid_point))

            # If the contracted point
            if contracted < contraction_base:
                self.guesses[-1] = contracted
            else:
                # Shrink
                for index, point in enumerate([g.point for g in self.guesses[1:]]):
                    shrunk_point = best.point + self.sigma * (point - best.point)
                    self.guesses[index] = self.evaluate_point(shrunk_point)

        return best
