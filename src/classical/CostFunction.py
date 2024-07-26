from matplotlib import pyplot as plt

from classical.S_DES import bitstring_8
from classical.regular_graph import generate_regular_graph_edges
from classical.util import generate_random_message, hamming_distance, to_bits
from quantum.util import Hamiltonian


# Define the Hamiltonian components
def compute_wij(Vi: bool, Vj: bool) -> float:
    return 1. if Vi != Vj else -1.


def compute_ti(Vi: bool) -> float:
    return 0.5 if Vi else -0.5


def construct_graph_hamiltonian_for_ciphertext(ciphertext: bitstring_8) -> Hamiltonian:
    edges = generate_regular_graph_edges(3, 8)
    return construct_hamiltonian_from_graph(ciphertext, edges)


def construct_hamming_hamiltonian_for_ciphertext(ciphertext: bitstring_8) -> Hamiltonian:
    def calculate_hamiltonian(measurement: list[bool]) -> float:
        return hamming_distance(ciphertext, tuple(measurement)) * 7 - 16

    return Hamiltonian(calculate_hamiltonian)


# Construct the Hamiltonian for a given regular graph
def construct_hamiltonian_from_graph(V, edges) -> Hamiltonian:
    num_nodes = len(V)

    pair_terms: list[tuple[float, int, int]] = []
    single_terms: list[tuple[float, int]] = []

    # Add terms wij Zi Zj
    for (i, j) in edges:
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
            result += ti * measurement[i]

        return result

    return Hamiltonian(calculate_hamiltonian)


if __name__ == '__main__':
    # noinspection PyTypeChecker
    bits: bitstring_8 = generate_random_message()
    hamiltonian = construct_graph_hamiltonian_for_ciphertext(bits)

    messages = [to_bits(i, 8) for i in range(2 ** 8)]
    messages.sort()
    messages.sort(key=lambda i: hamming_distance(bits, i))

    costs = [hamiltonian.calculate(list(i)) for i in messages]
    hamming = [hamming_distance(bits, i) for i in messages]

    fig, ax1 = plt.subplots()
    ax1.scatter(costs, hamming)
    ax1.set_xlabel('Cost')
    ax1.set_ylabel('Hamming')

    plt.title('Cost vs Hamming Distance')
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    fig.savefig('../../misc/cost-hamming.png')
    plt.show()
