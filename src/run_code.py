from qiskit_based.CostFunctionQiskit import *
from quantum.VQE import *
from quantum.qsdes_ansatz import QuantumSDESAnsatz


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

    num_iterations = 200

    estimator = HamiltonianEstimator(
        QuantumSDESAnsatz(known_plaintext),
        hamiltonian_operator=get_hamil_pauli_op(known_ciphertext=known_ciphertext),
        shots_per_estimate=200
    )

    optimizer = get_optimizer(
        optimizer_type='nelder-mead',
        cost_function=lambda x: estimator.evaluate_cost(x),
        cost_cutoff=-15,
    )

    for i in range(num_iterations):
        optimizer.step()

        if optimizer.is_finished():
            print('Found solution by optimizer cutoff')
            break

    cost_history = [i.cost for i in optimizer.get_history()]
    guess_history = [i.point for i in optimizer.get_history()]

    # Plot the guess history
    fig, ax1 = plt.subplots()
    ax1.plot(cost_history, label='Cost', color='tab:blue')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('Cost Function for Nelder Mead')
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


def get_optimizer(
        optimizer_type: str,
        cost_function: cost_function_t,
        cost_cutoff: -15,
) -> Optimizer[bitstring_10]:
    if optimizer_type == 'gradient-descent':
        return GradientDescentOptimizer(
            cost_function,
            cost_cutoff,

            initial_point=np.array([1] + ([0] * 9), dtype=float),
            learning_rate=0.06,
            gradient_cutoff=0
        )

    elif optimizer_type == 'nelder-mead':
        return NelderMeadOptimizer(
            cost_function,
            cost_cutoff,

            dimensionality=10,
            random_simplex_generator=lambda: [np.random.uniform(0, math.tau, 10) for _ in range(11)],
            range_cutoff=0
        )

    else:
        raise ValueError(f'Unknown optimizer type: {optimizer_type}')


if __name__ == "__main__":
    run()
