import numpy as np
from matplotlib import pyplot as plt

from classical.Optimization import gradient_descent, GradientDescent
from classical.S_DES import encrypt_sdes
from classical.util import bits_to_string, hamming_distance
from quantum.VQE import VQE_crypto


def run():
    known_plaintext = (True, False, True, True, False, True, True, False)
    key = (False, True, False, True, False, True, False, True, False, True)
    known_ciphertext = encrypt_sdes(known_plaintext, key)

    print(f'Testing with key={bits_to_string(key)}')

    vqe_solver = VQE_crypto(known_plaintext, known_ciphertext)

    learning_rate = 1.08

    optimizer = GradientDescent(
        initial_guess=np.array([1] + ([0] * 9), dtype=float),
        cost_function=lambda x: vqe_solver.run(x),
        learning_rate=learning_rate,
        cost_cutoff=-9,
    )

    for i in range(200):
        if optimizer.step():
            # We can stop
            break

    cost_history = optimizer.cost_history
    guess_history = optimizer.guess_history
    hamming_distance_history = [hamming_distance(i, key) for i in optimizer.key_history]

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


if __name__ == "__main__":
    run()
