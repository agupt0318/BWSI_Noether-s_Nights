# Algorithms taken from Appendix C of the paper A Variational Quantum Attack for AES-like
# Symmetric Cryptography (Wang et al., 2022)
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from classical.util import bitstring_10, bits_to_string, hamming_distance


def calculate_gradient_at_point(
        point: np.ndarray,
        cost_at_point: float,
        cost_function: Callable[[np.ndarray], float],
        num_dimensions
):
    epsilon = 0.01  # small step size for calculating the approximate gradient

    gradient = np.zeros(num_dimensions)
    for i in range(num_dimensions):
        # Calculate a point a small step away in the ith direction
        adjusted_point = point.copy()
        adjusted_point[i] += epsilon
        # Approximate the gradient in the ith direction using the difference quotient
        cost_at_adjusted_point, _ = cost_function(adjusted_point)
        gradient[i] = (cost_at_adjusted_point - cost_at_point) / epsilon

    return gradient


def gradient_descent(
        known_key: bitstring_10,
        guess: np.ndarray,
        cost_function: Callable[[np.ndarray], tuple[float, bitstring_10]],
        learning_rate: float,
        cost_cutoff: float,
        num_iterations: int = 1024
) -> np.ndarray:
    adaptive_factor = 0
    num_dimensions = len(guess)

    # noinspection PyTypeChecker
    best_guess: tuple[np.ndarray, float] = (None, float('inf'))

    guess_history = []
    cost_history = []
    hamming_distance_history = []

    for _ in range(num_iterations):

        adaptive_factor += 1

        cost_at_current_guess, key_for_current_guess = cost_function(guess)
        print(f'Current guess: {np.round(guess, 2)}, key: {bits_to_string(key_for_current_guess)}')

        guess_history.append(guess.copy())
        cost_history.append(cost_at_current_guess)
        hamming_distance_history.append(hamming_distance(key_for_current_guess, known_key))

        # If the cost is strictly less than the cutoff, stop
        if cost_at_current_guess < cost_cutoff:
            return guess
        # Otherwise, update the current best guess
        elif cost_at_current_guess < best_guess[1]:
            best_guess = (guess.copy(), cost_at_current_guess)

        gradient = calculate_gradient_at_point(guess, cost_at_current_guess, cost_function, num_dimensions)

        # Calculate the adaptive step size
        step_size = learning_rate / abs(cost_at_current_guess + cost_cutoff) \
                    + np.log(adaptive_factor) / adaptive_factor * np.random.uniform(0, 1)

        # Update the guess based on the gradient
        guess -= gradient * step_size

        # If the gradient is too low, generate a new random guess
        if sum(gradient ** 2) ** 0.5 < 0.8:
            guess = np.random.uniform(-1, 1, num_dimensions)

    # Plot the guess history
    # line graph for hamming distance and hamiltonian vs iteration
    # line graph for ansatz params (10 lines) vs iteration
    fig, ax1 = plt.subplots()
    ax1.plot(cost_history, label='Cost', color='tab:blue')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(hamming_distance_history, label='Hamming Distance', color='tab:red')
    ax2.set_ylabel('Hamming Distance', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Cost and Hamming Distance Over Iterations')
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    fig.savefig('../../misc/cost-hamming.png')
    plt.show()

    fig, ax = plt.subplots()

    for i in range(10):
        guesses_at_i = [guess[i] for guess in guess_history]
        ax.plot(guesses_at_i, label=f'Index {i}')
        # guess is an array of 10 elements, should have 1 line per index
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Guess Value')
    ax.set_title('Guess History')
    ax.legend()
    fig.savefig('../../misc/ansatz.png')
    plt.show()


    return best_guess[0]


def n_m_method(cost_function, guess, alpha, cost_cutoff):
    """
    generates the N other points (x1...xN)

    :param cost_function: implemented cost_function for computations
    :param guess: initial point
    :param alpha: amplification factor
    :param cost_cutoff: cut-off condition
    :return: new value of guess for iteration above in points
    """

    N = len(guess)  # N dimensions of guess
    points = [guess]  # list of x1--xN values

    # calculating xi component
    def calc_xi_component(step: int):  # step should be 0 or 1
        xi = guess.copy()
        for i in range(N):
            if guess[i] == 0:
                xi[i] = 0.8
            else:
                xi[i] = guess[i] * alpha
            points[i + step] = xi

    calc_xi_component(0)

    times = N + 1
    while times < 1024:
        points.sort(key=lambda x: cost_function(x))
        if cost_function(points[0]) <= cost_cutoff:
            break
        if cost_function(points[-1]) - cost_function(points[1]) < 0.15:
            calc_xi_component(1)
            continue

        # calculate average of first N points m
        m = np.mean(points[:-1], axis=0)
        # calculate reflect point r
        r = 2 * m - points[-1]
        times += 1

        if cost_function(points[0]) <= cost_function(r) < cost_function(points[-2]):
            points[-1] = r
            continue

        if cost_function(r) < cost_function(points[0]):
            # calculate expand point s
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
