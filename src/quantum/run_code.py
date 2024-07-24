from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

from classical.CostFunction import construct_hamiltonian
from classical.S_DES import encrypt_sdes
from quantum.S_DES import QuantumSDES
from quantum.ansatz import A_ansatz_Y_Cz_model
from quantum.utils import write_classical_data, Hamiltonian


def cost_function(hamiltonian: Hamiltonian, circuit: QuantumCircuit):
    # circuit.assign_parameters()

    pass

def run():
    known_plaintext = (True, False, True, True, False, True, True, False)
    key = (False, True, False, True, False, True, False, True, False, True)
    known_ciphertext = encrypt_sdes(known_plaintext, key)

    hamiltonian = construct_hamiltonian(known_ciphertext, 3)  # 3 regular graph

    key_register = QuantumRegister(10)
    text_register = QuantumRegister(8)

    circuit = QuantumCircuit(key_register, text_register)

    circuit.compose(A_ansatz_Y_Cz_model(1, 0, 0, 0, 0, 0, 0, 0, 0, 0), key_register, inplace=True)

    write_classical_data(list(known_plaintext), circuit, target_qubits=list(text_register))

    circuit.barrier()
    circuit.compose(QuantumSDES(key=key_register, data=text_register))
    circuit.barrier()

    classical_register = ClassicalRegister(8)
    circuit.add_register(classical_register)
    circuit.measure(text_register, classical_register)

    measurements = AerSimulator().run(circuit, shots=20, memory=True).result().get_memory()
    # Calculate expected value of hamiltonian
    total = 0
    for measurement in measurements:
        measurement = [i == '1' for i in measurement]
        total += hamiltonian.calculate(measurement)
    expected_value_of_hamiltonian = total / len(measurements)

    return None


if __name__ == "__main__":
    run()
