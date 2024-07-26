from classical.CostFunction import *
from classical.Optimization import *
from classical.S_DES import encrypt_sdes
from classical.util import bits_to_string, hamming_distance, generate_random_key, generate_random_message
from quantum.VQE import VQE_crypto


def generate_random_simplex() -> list[ndarray]:
    return [np.random.uniform(0, math.tau, 10) for _ in range(11)]


def run():
    known_plaintext = generate_random_message()
    secret_key = generate_random_key()

    known_ciphertext = encrypt_sdes(known_plaintext, secret_key)

    print(f'Testing with key={
    bits_to_string(secret_key)
    }, message={
    bits_to_string(known_plaintext)
    }, ciphertext={
    bits_to_string(known_ciphertext)
    }')

    hamiltonian = construct_graph_hamiltonian_for_ciphertext(known_ciphertext)

    vqe_solver = VQE_crypto(
        known_plaintext,
        known_ciphertext,
        hamiltonian,
        shots_per_estimate=10,
        find_solution_by_lucky_measurement=False
    )
    #
    optimizer: Optimizer = GradientDescentOptimizer(
        cost_function=lambda x: vqe_solver.run(x),
        cost_cutoff=-9,
        initial_point=np.array([1] + ([0] * 9), dtype=float),
        learning_rate=1.08
    )

    # optimizer: Optimizer = AdaGradOptimizer(
    #     cost_function=lambda x: vqe_solver.run(x),
    #     cost_cutoff=-9,
    #     initial_point=np.array([1] + ([0] * 9), dtype=float),
    #     learning_rate=0.05,
    #     adagrad_factor=0.1
    # )

    # optimizer: Optimizer = NelderMeadOptimizer(
    #     cost_function=lambda x: vqe_solver.run(x),
    #     cost_cutoff=-9,
    #     dimensionality=10,
    #     random_simplex_generator=generate_random_simplex,
    # )

    for i in range(50):
        optimizer.step()

        if vqe_solver.solution is not None:
            print('Found solution by lucky measurement')
            break

        if optimizer.is_finished():
            print('Found solution by optimizer cutoff')
            break

    if vqe_solver.solution is None:
        solution = optimizer.get_best_guess().data
    else:
        solution = vqe_solver.solution

    cost_history = [i.cost for i in optimizer.get_history()]
    guess_history = [i.point for i in optimizer.get_history()]
    hamming_distance_history = [hamming_distance(i.data, secret_key) for i in optimizer.get_history()]

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

    plt.show()

    if type(optimizer) is NelderMeadOptimizer:
        optimizer: NelderMeadOptimizer = optimizer

        fig, ax = plt.subplots()
        ax.plot(optimizer.volume_history)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Simplex volume')
        ax.set_title('Volume History')
        ax.legend()

        plt.show()

    print(f'Final result: key={
    bits_to_string(solution)
    }, encrypted={
    bits_to_string(encrypt_sdes(known_plaintext, solution))
    }')


if __name__ == "__main__":
    run()
