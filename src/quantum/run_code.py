from classical.CostFunction import *
from classical.Optimization import *
from classical.S_DES import *
from classical.util import *
from qiskit_based.CostFunctionQiskit import get_hamil_pauli_op
from quantum.VQE import *
from quantum.qsdes_ansatz import QuantumSDESAnsatz


def generate_random_simplex() -> list[ndarray]:
    return [np.random.uniform(0, math.tau, 10) for _ in range(11)]


def run():
    known_plaintext = generate_random_message()
    secret_key = generate_random_key()

    known_ciphertext = encrypt_sdes(known_plaintext, secret_key)

    print(
        f'Testing with key={bits_to_string(secret_key)}, message={bits_to_string(known_plaintext)}, ciphertext={bits_to_string(known_ciphertext)}')

    num_iterations = 200

    vqe_solver = VariationalQuantumEigensolver(
        QuantumSDESAnsatz(known_plaintext),
        # known_plaintext,
        hamiltonian_operator=get_hamil_pauli_op(known_ciphertext=known_ciphertext),
        shots_per_estimate=200
    )

    # optimizer: Optimizer[bitstring_10] = GradientDescentOptimizer(
    #     cost_function=lambda x: vqe_solver.evaluate_cost(x),
    #     cost_cutoff=-9,
    #     initial_point=np.array([1] + ([0] * 9), dtype=float),
    #     learning_rate=0.06,
    #     gradient_cutoff=0
    # )

    optimizer: Optimizer = NelderMeadOptimizer(
        cost_function=lambda x: vqe_solver.evaluate_cost(x),
        cost_cutoff=-9,
        dimensionality=10,
        random_simplex_generator=generate_random_simplex,
    )

    for i in range(num_iterations):
        optimizer.step()

        if optimizer.is_finished():
            print('Found solution by optimizer cutoff')
            break

    solution = optimizer.get_best_guess().data

    cost_history = [i.cost for i in optimizer.get_history()]
    guess_history = [i.point for i in optimizer.get_history()]

    # Plot the guess history
    fig, ax1 = plt.subplots()
    ax1.plot(cost_history, label='Cost', color='tab:blue')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('Cost and Hamming Distance Over Iterations')
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    plt.show()

    fig, ax = plt.subplots()

    for i in range(10):
        guesses_at_i = [guess[i] for guess in guess_history]
        ax.plot(guesses_at_i, label=f'θ{i}')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Guess Value')
    ax.set_title('Guess History')
    ax.legend()

    plt.show()

    if type(optimizer) is NelderMeadOptimizer and False:
        # noinspection PyTypeChecker
        optimizer: NelderMeadOptimizer[bitstring_10] = optimizer

        fig, ax = plt.subplots()
        ax.plot(optimizer.volume_history)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Simplex volume')
        ax.set_title('Volume History')
        ax.legend()

        plt.show()

    print(
        f'Final result: key={bits_to_string(solution)}, encrypted={bits_to_string(encrypt_sdes(known_plaintext, solution))}')


if __name__ == "__main__":
    run()
