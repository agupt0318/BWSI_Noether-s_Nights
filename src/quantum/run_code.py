import random

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

from classical.CostFunction import construct_hamiltonian_for_ciphertext
from classical.Optimization import GradientDescentOptimizer, NelderMeadOptimizer
from classical.S_DES import encrypt_sdes
from classical.util import bits_to_string, hamming_distance, generate_random_key, generate_random_message
from quantum.VQE import VQE_crypto


def generate_random_simplex() -> list[ndarray]:
    return [
        np.array([random.random() for _ in range(10)], dtype=float) for _ in range(11)
    ]


def run():
    known_plaintext = generate_random_message()
    key = generate_random_key()

    known_ciphertext = encrypt_sdes(known_plaintext, key)

    print(f'Testing with key={bits_to_string(key)}')

    vqe_solver = VQE_crypto(
        known_plaintext,
        construct_hamiltonian_for_ciphertext(known_ciphertext),
        shots_per_estimate=100
    )

    # optimizer = GradientDescentOptimizer(
    #     cost_function=lambda x: vqe_solver.run(x),
    #     cost_cutoff=-9,
    #     initial_point=np.array([1] + ([0] * 9), dtype=float),
    #     learning_rate=.05
    # )

    optimizer = NelderMeadOptimizer(
        cost_function=lambda x: vqe_solver.run(x),
        cost_cutoff=-9,
        dimensionality=10,
        random_simplex_generator=generate_random_simplex,
    )

    for i in range(200):
        optimizer.step()
        if optimizer.finished:
            break

    cost_history = [i.cost for i in optimizer.history]
    guess_history = [i.point for i in optimizer.history]
    hamming_distance_history = [hamming_distance(i.key, key) for i in optimizer.history]

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
    # fig.savefig('../../misc/cost-hamming.png')
    plt.show()

    fig, ax = plt.subplots()

    for i in range(10):
        guesses_at_i = [guess[i] for guess in guess_history]
        ax.plot(guesses_at_i, label=f'θ{i}')
        # guess is an array of 10 elements, should have 1 line per index
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Guess Value')
    ax.set_title('Guess History')
    ax.legend()
    # fig.savefig('../../misc/ansatz.png')
    plt.show()


    fig, ax = plt.subplots()
    ax.plot(optimizer.volume_history)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Simplex volume')
    ax.set_title('Volume History')
    ax.legend()
    # fig.savefig('../../misc/ansatz.png')
    plt.show()


if __name__ == "__main__":
    run()
