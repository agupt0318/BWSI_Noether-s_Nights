# psuedocode basis is from Appendix C of the paper A Variational Quantum Attack for AES-like
# Symmetric Cryptography (Wang et al., 2022)
from typing import Callable

import numpy as np


def calculate_gradient_at_point(
        point: np.ndarray,
        cost_at_point: float,
        cost_function: Callable[[np.ndarray], float],
        num_dimensions
):
    epsilon = 0.01

    gradient = np.zeros(num_dimensions)
    for i in range(num_dimensions):
        # Calculate a point a small step away in the ith direction
        adjusted_point = point.copy()
        adjusted_point[i] += epsilon
        # Approximate the gradient in the ith direction using the difference quotient
        cost_at_adjusted_point = cost_function(adjusted_point)
        gradient[i] = (cost_at_adjusted_point - cost_at_point) / epsilon

    return gradient


def gradient_descent(
        guess: np.ndarray,
        cost_function: Callable[[np.ndarray], float],
        learning_rate: float,
        cost_cutoff: float
) -> np.ndarray:
    f = 0
    num_dimensions = len(guess)

    # noinspection PyTypeChecker
    best_guess: tuple[np.ndarray, float] = (None, float('inf'))

    for ii in range(1024):
        f += num_dimensions + 1

        cost_at_current_guess = cost_function(guess)

        # If the cost is less than the cutoff, stop
        if cost_at_current_guess < cost_cutoff:
            return guess

        elif cost_at_current_guess < best_guess[1]:
            # Update the current best guess
            best_guess = (guess.copy(), cost_at_current_guess)

        gradient = calculate_gradient_at_point(guess, cost_at_current_guess, cost_function, num_dimensions)

        # Calculate the adaptive step size
        step_size = learning_rate / abs(cost_at_current_guess) \
                    + np.log(f) / f * np.random.uniform(0, 1)

        # Update the guess based on the gradient
        guess -= gradient * step_size

        # If the gradient is too low, generate a new random guess
        if gradient < 0.8:
            guess = np.random.uniform(-1, 1, num_dimensions)

    return best_guess[0]


def n_m_method(cost_function, guess, alpha, xerr):
    """
    generates the N other points (x1...xN)

    :param cost_function: implemented cost_function for computations
    :param guess: initial point
    :param alpha: amplification factor
    :param xerr: cut-off condition
    :return: new value of guess for iteration above in points
    """

    N = len(guess)  # N dimensions of guess
    points = [guess]  # list of x1--xN values

    # calculating xi component on guess
    def calc_xi_component():
        for i in range(N):
            xi = guess.copy()
            if guess[i] == 0:
                xi[i] = 0.8
            else:
                xi[i] = guess[i] * alpha
        return xi

    points.append(calc_xi_component())

    times = N + 1
    while times < 1024:
        points.sort(key=lambda x: cost_function(x))
        if cost_function(points[0]) <= xerr:
            break
        if cost_function(points[-1]) - cost_function(points[1]) < 0.15:
            points[i + 1] = calc_xi_component()
            continue

        # calculate average of first N points
        m = np.mean(points[:-1], axis=0)
        # calculate reflect point
        r = 2 * m - points[-1]
        times += 1

        if (cost_function(points[0]) <= cost_function(r) < cost_function(points[-2])):
            points[-1] = r
            continue

        if cost_function(r) < cost_function(points[0]):
            # calculate expand point
            s = m + 2 * (m - points[-1])
            times += 1
            if cost_function(s) < cost_function(r):
                points[-1] = s
                continue
            else:
                points[-1] = r
                continue

        if cost_function(points[-2]) <= cost_function(r) < cost_function(points[-1]):
            c1 = m + (r - m) / 2
            times += 1
            if cost_function(c1) < cost_function(r):
                points[-1] = c1
                continue
            else:
                for i in range(1, N + 1):
                    points[i] = guess + (points[i] - guess) / 2.0
                times += N
                continue
        if cost_function(points[-1]) <= cost_function(r):
            c2 = m + (points[-1] - m) / 2.0
            times += 1
            if cost_function(c2) < cost_function(points[-1]):
                points[-1] = c2
                continue
            else:
                for i in range(1, N + 1):
                    points[i] = guess + (points[i] - guess) / 2.0
                times += N
                continue
    return points[0]
