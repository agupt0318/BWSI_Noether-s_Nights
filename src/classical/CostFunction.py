from matplotlib import pyplot as plt

from classical.S_DES import bitstring_8, encrypt_sdes
from classical.regular_graph import generate_regular_graph_edges
from classical.util import generate_random_message, hamming_distance, to_bits, bitstring_10, generate_random_key
from quantum.util import Hamiltonian


# Define the Hamiltonian components
def compute_wij(Vi: bool, Vj: bool) -> float:
    return 1. if Vi != Vj else -1.


def compute_ti(Vi: bool) -> float:
    return 0.5 if Vi else -0.5


def construct_graph_hamiltonian_for_ciphertext(ciphertext: bitstring_8) -> Hamiltonian:
    edges = generate_regular_graph_edges(3, 8)
    return construct_hamiltonian_from_graph(ciphertext, edges)


# Construct the Hamiltonian for a given regular graph
def construct_hamiltonian_from_graph(V, edges) -> Hamiltonian:
    num_nodes = len(V)

    pair_terms: list[tuple[float, int, int]] = []
    single_terms: list[tuple[float, int]] = []

    # Add terms wij Zi Zj
    for (i, j, _) in edges:
        wij = compute_wij(V[i], V[j])
        pair_terms.append((wij, i, j))

    # Add single-qubit terms ti Zi
    for i in range(num_nodes):
        ti = compute_ti(V[i])
        single_terms.append((ti, i))

    def calculate_hamiltonian(measurement: list[float]) -> float:
        result = 0
        for (wij, i, j) in pair_terms:
            result += wij * measurement[i] * measurement[j]
        for (ti, i) in single_terms:
            result -= ti * measurement[i]

        return result

    return Hamiltonian(calculate_hamiltonian)


if __name__ == '__main__':
    # noinspection PyTypeChecker
    message: bitstring_8 = generate_random_message()
    true_key: bitstring_10 = generate_random_key()
    encrypted: bitstring_8 = encrypt_sdes(message, true_key)

    hamiltonian = construct_graph_hamiltonian_for_ciphertext(encrypted)

    keys = [to_bits(i, 10) for i in range(2 ** 10)]
    encrypted_with_keys = [encrypt_sdes(message, key) for key in keys]

    ciphertext_costs = [hamiltonian.calculate(list(e)) for e in encrypted_with_keys]
    ciphertext_hamming = [hamming_distance(encrypted, e) for e in encrypted_with_keys]
    key_hamming = [hamming_distance(true_key, i) for i in keys]

    fig, ax = plt.subplots()
    ax.scatter(ciphertext_costs, key_hamming)
    ax.set_xlabel('Ciphertext Cost')
    ax.set_ylabel('Key Hamming Distance')

    plt.title('Ciphertext Cost vs Key Hamming Distance')
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(ciphertext_hamming, key_hamming)
    ax.set_xlabel('Ciphertext Hamming Distance')
    ax.set_ylabel('Key Hamming Distance')

    plt.title('Ciphertext Hamming Distance vs Key Hamming Distance')
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(ciphertext_hamming, ciphertext_costs)
    ax.set_xlabel('Ciphertext Hamming Distance')
    ax.set_ylabel('Ciphertext Cost')

    plt.title('Ciphertext Hamming Distance vs Ciphertext Cost')
    plt.show()
