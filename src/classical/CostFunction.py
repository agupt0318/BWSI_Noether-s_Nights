import numpy as np


# Define the Pauli-Z operator
def pauli_z(state, i):
    state[i] = 1 - state[i]  # Toggle state, flips 180
    return state


# Define the Hamiltonian components
def compute_wij(Vi, Vj):
    return 1 if Vi != Vj else -1


def compute_ti(Vi):
    return 0.5 if Vi == 1 else -0.5

def make_edge_for_each_vertex_x_steps_away(edges, num_nodes, x):
    for i in range(num_nodes):
        u, v = i, (i + x) % num_nodes
        if (u, v) not in edges and (v, u) not in edges:
            edges.add((u, v))
    return edges

# Generate all possible edges for a given regularity
# Code was lovingly inspired by https://math.stackexchange.com/questions/142112/how-to-construct-a-k-regular-graph
def generate_regular_graph_edges(n, num_nodes):
    edges = set()
    if n == 1:
        edges = make_edge_for_each_vertex_x_steps_away(edges, num_nodes, 4)
    elif n == 2:
        edges = make_edge_for_each_vertex_x_steps_away(edges, num_nodes, 1)
    elif n == 3:
        edges = make_edge_for_each_vertex_x_steps_away(edges, num_nodes, 4)
        edges = make_edge_for_each_vertex_x_steps_away(edges, num_nodes, 3)
    elif n == 4:
        edges = make_edge_for_each_vertex_x_steps_away(edges, num_nodes, 1)
        edges = make_edge_for_each_vertex_x_steps_away(edges, num_nodes, 2)
    elif n == 5:
        edges = make_edge_for_each_vertex_x_steps_away(edges, num_nodes, 4)
        edges = make_edge_for_each_vertex_x_steps_away(edges, num_nodes, 3)
        edges = make_edge_for_each_vertex_x_steps_away(edges, num_nodes, 2)
    elif n == 6:
        edges = make_edge_for_each_vertex_x_steps_away(edges, num_nodes, 1)
        edges = make_edge_for_each_vertex_x_steps_away(edges, num_nodes, 2)
        edges = make_edge_for_each_vertex_x_steps_away(edges, num_nodes, 3)
    elif n == 7:
        edges = make_edge_for_each_vertex_x_steps_away(edges, num_nodes, 4)
        edges = make_edge_for_each_vertex_x_steps_away(edges, num_nodes, 3)
        edges = make_edge_for_each_vertex_x_steps_away(edges, num_nodes, 2)
        edges = make_edge_for_each_vertex_x_steps_away(edges, num_nodes, 1)
    return edges


# Construct the Hamiltonian for a given regular graph
def construct_hamiltonian(V, edges):
    num_nodes = len(V)
    H = np.zeros((2 ** num_nodes, 2 ** num_nodes))

    # Add terms wij Zi Zj
    for (i, j) in edges:
        wij = compute_wij(V[i], V[j])
        for state_idx in range(2 ** num_nodes):
            state = list(map(int, bin(state_idx)[2:].zfill(num_nodes)))
            new_state = pauli_z(state[:], i)
            new_state = pauli_z(new_state, j)
            new_state_idx = int(''.join(map(str, new_state)), 2)
            H[state_idx, new_state_idx] += wij

    # Add single-qubit terms ti Zi
    for i in range(num_nodes):
        ti = compute_ti(V[i])
        for state_idx in range(2 ** num_nodes):
            state = list(map(int, bin(state_idx)[2:].zfill(num_nodes)))
            new_state = pauli_z(state[:], i)
            new_state_idx = int(''.join(map(str, new_state)), 2)
            H[state_idx, new_state_idx] += ti

    return H


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
