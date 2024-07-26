import numpy as np

from classical.CostFunction import construct_graph_hamiltonian_for_ciphertext
from classical.S_DES import encrypt_sdes
from classical.util import bits_to_string, generate_random_key, generate_random_message
from quantum.VQE import VQE_crypto


def gradient_descent(x0, function, r, xerr):
    # x0: initial point (minimum expectation of the Hamiltonian from N+1 points) -- int
    # function: the function
    # r: the learning rate -- float
    count = 0
    length = len(x0)
    for ii in range(1024):
        cost = function(x0)
        print(x0, cost)

        count += 1
        if cost < xerr:
            break
        gd = np.zeros(length)
        for i in range(length):
            x = x0.copy()
            x[i] += 0.01
            cost_prime = function(x)
            count += 1
            gd[i] = (cost_prime - cost) / 0.01
        # Generate a random number r0 in range [0, 1]
        r0 = np.random.uniform(0, 1)
        step_size = r / abs(cost - xerr) + np.log(count) / count * r0
        x0 -= step_size * gd

        if sum(gd ** 2) ** 0.5 < 0.8:
            x0 = np.random.uniform(-1, 1, length)
    return x0


def easom_function(x):
    return -np.cos(x[0]) * -np.cos(x[1]) * np.exp(-((x[0] - np.pi) ** 2 + (x[1] - np.pi) ** 2))

def rosenbrock(x) -> float:
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


if __name__ == "__main__":
    # Initial point for gradient descent
    initial_point = np.random.uniform(-1, 1, 2)

    optimized_point = gradient_descent(
        x0=initial_point,
        function=rosenbrock,
        r=0.5,
        xerr=-0.1
    )

    print(f'Optimized point: {optimized_point}')
