import numpy as np

from classical.CostFunction import construct_hamiltonian_from_graph
from classical.regular_graph import generate_regular_graph_edges


def energy_level_analysis(V, num_nodes=8):
    results = []
    for n in range(1, num_nodes):
        edges = generate_regular_graph_edges(n, num_nodes)
        H = construct_hamiltonian_from_graph(V, edges)
        eigenvalues = np.linalg.eigvalsh(H)
        ground_energy = np.min(eigenvalues)
        highest_energy = np.max(eigenvalues)
        first_excited_energy = sorted(eigenvalues)[1]
        ratio = (first_excited_energy - ground_energy) / (highest_energy - ground_energy)
        results.append((n, ground_energy, highest_energy, first_excited_energy, ratio))
    return results


if __name__ == '__main__':
    # Example usage
    V = [1, 0, 1, 1, 1, 0, 1, 0]  # Example ciphertext values
    results = energy_level_analysis(V)

    # Print results
    for (n, ground_energy, highest_energy, first_excited_energy, ratio) in results:
        print(f"{n}-regular: Ground energy={ground_energy}, Highest energy={highest_energy}, "
              f"First excited energy={first_excited_energy}, Ratio={ratio:.4f}")
