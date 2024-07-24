import numpy as np

from quantum.utils import Hamiltonian


# Define the Hamiltonian components
def compute_wij(Vi, Vj):
    return 1 if Vi != Vj else -1


def compute_ti(Vi):
    return 0.5 if Vi == 1 else -0.5


def make_edge_for_each_vertex_n_steps_away(edges, num_nodes, n):
    for i in range(num_nodes):
        u, v = i, (i + n) % num_nodes
        if (u, v) not in edges and (v, u) not in edges:
            edges.add((u, v))
    return edges


# Generate all possible edges for a given regularity
# Code was lovingly inspired by https://math.stackexchange.com/questions/142112/how-to-construct-a-k-regular-graph
def generate_regular_graph_edges(n, num_nodes):
    edges = set()
    if n == 1:
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 4)
    elif n == 2:
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 1)
    elif n == 3:
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 4)
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 3)
    elif n == 4:
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 1)
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 2)
    elif n == 5:
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 4)
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 3)
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 2)
    elif n == 6:
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 1)
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 2)
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 3)
    elif n == 7:
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 4)
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 3)
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 2)
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 1)
    return edges


# Construct the Hamiltonian for a given regular graph
def construct_hamiltonian(V, edges) -> Hamiltonian:
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


# Low priority

# Energy level analysis for different regularities
def energy_level_analysis(V, num_nodes=8):
    results = []
    for n in range(1, num_nodes):
        edges = generate_regular_graph_edges(n, num_nodes)
        H = construct_hamiltonian(V, edges)
        eigenvalues = np.linalg.eigvalsh(H)
        ground_energy = np.min(eigenvalues)
        highest_energy = np.max(eigenvalues)
        first_excited_energy = sorted(eigenvalues)[1]
        ratio = (first_excited_energy - ground_energy) / (highest_energy - ground_energy)
        results.append((n, ground_energy, highest_energy, first_excited_energy, ratio))
    return results


# Example usage
V = [1, 0, 1, 1, 1, 0, 1, 0]  # Example ciphertext values
results = energy_level_analysis(V)

# Print results
for (n, ground_energy, highest_energy, first_excited_energy, ratio) in results:
    print(f"{n}-regular: Ground energy={ground_energy}, Highest energy={highest_energy}, "
          f"First excited energy={first_excited_energy}, Ratio={ratio:.4f}")
