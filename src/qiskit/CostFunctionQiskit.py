import rustworkx as rx
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import Aer
from qiskit_ibm_runtime import *
import logging
from classical.CostFunction import *
from classical.Optimization import *
from classical.S_DES import *
from classical.util import *
from quantum.VQE import *

# To run on hardware, select the backend with the fewest number of jobs in the queue

logger = logging.getLogger(__name__)
cost_history_dict = {"prev_vector": None, "iters": 0, "cost_history": [], }
backend = Aer.get_backend('aer_simulator')


# Define the Hamiltonian components
def compute_wij(Vi: bool, Vj: bool) -> float:
    return 1. if Vi != Vj else -1.


def compute_ti(Vi: bool) -> float:
    return 0.5 if Vi else -0.5


def generate_random_simplex() -> list[ndarray]:
    return [np.random.uniform(0, math.tau, 10) for _ in range(11)]


def build_max_cut_paulis(graph: rx.PyGraph, single_terms: list[tuple[int, float]]) -> list[tuple[str, float]]:
    """Convert the graph to Pauli list, including single-qubit terms."""
    pauli_list = []
    for edge in list(graph.edge_list()):
        paulis = ["I"] * graph.num_nodes()
        paulis[edge[0]], paulis[edge[1]] = "Z", "Z"
        weight = graph.get_edge_data(edge[0], edge[1])
        pauli_list.append(("".join(paulis)[::-1], weight))

    for index, coeff in single_terms:
        paulis = ["I"] * graph.num_nodes()
        paulis[index] = "Z"
        pauli_list.append(("".join(paulis)[::-1], coeff))

    return pauli_list


def construct_hamiltonian_from_graph(V, edges) -> tuple[list[tuple[int, int, float]], list[tuple[int, float]]]:
    num_nodes = len(V)
    pair_terms = []
    single_terms = []

    for (i, j, _) in edges:
        wij = compute_wij(V[i], V[j])
        pair_terms.append((i, j, wij))

    for i in range(num_nodes):
        ti = compute_ti(V[i])
        single_terms.append((i, ti))

    return pair_terms, single_terms


def get_isa_circuit(circuit: QuantumCircuit):
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    return pm.run(circuit)


def get_hamil_pauli_op(known_ciphertext):
    graph = rx.PyGraph()
    graph.add_nodes_from(np.arange(0, 10, 1))
    edge_list = generate_regular_graph_edges(3, 8)
    pair_terms, single_terms = construct_hamiltonian_from_graph(known_ciphertext, edge_list)
    graph.add_edges_from(pair_terms)

    max_cut_paulis = build_max_cut_paulis(graph, single_terms)
    cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
    return cost_hamiltonian


def cost(cost_hamiltonian, ansatz_parameters: np.ndarray, isa_circuit: QuantumCircuit):
    print("Cost Function Hamiltonian:", cost_hamiltonian)

    isa_observable = cost_hamiltonian.apply_layout(isa_circuit.layout)
    estimator = EstimatorV2(backend, options={"default_shots": int(1e4)})
    job = estimator.run([(isa_circuit, isa_observable, ansatz_parameters, None)])
    pub_result = job.result()[0]
    return pub_result.data.evs


def cost_func(known_ciphertext, isa_circuit, ansatz_parameters):
    H = get_hamil_pauli_op(known_ciphertext)
    return cost(H, ansatz_parameters, isa_circuit)


if __name__ == '__main__':
    np.random.seed(6)
    random.seed(6)

    known_plaintext = generate_random_message()
    secret_key = generate_random_key()
    known_ciphertext = encrypt_sdes(known_plaintext, secret_key)
    print(
        f'Testing with key={
            bits_to_string(secret_key)
        }, message={
            bits_to_string(known_plaintext)
        }, ciphertext={
            bits_to_string(known_ciphertext)
        }'
    )
    H = get_hamil_pauli_op(known_ciphertext)
    circuit = VQE_crypto(known_plaintext, known_ciphertext, Hamiltonian(lambda _: 0), 0)
    isa_circuit = get_isa_circuit(circuit)
    print(cost(H, np.array([1] + [0] * 9), isa_circuit))
