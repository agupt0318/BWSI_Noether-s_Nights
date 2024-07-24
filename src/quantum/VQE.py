from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator

from classical.CostFunction import construct_hamiltonian
from classical.S_DES import bitstring_8
from quantum.S_DES import QuantumSDES
from quantum.ansatz import A_ansatz_Y_Cz_model
from quantum.utils import write_classical_data


class VQE_crypto(QuantumCircuit):
    def __init__(self, known_plaintext: bitstring_8, known_ciphertext: bitstring_8):
        self.simulator = AerSimulator()

        hamiltonian = self.hamiltonian = construct_hamiltonian(known_ciphertext, 3)

        key_register = self.key_register = QuantumRegister(10)
        text_register = self.text_register = QuantumRegister(8)

        super().__init__(key_register, text_register)

        ansatz_parameters = self.ansatz_parameters = [Parameter(f'ansatz_param_{i}') for i in range(10)]
        self.compose(A_ansatz_Y_Cz_model(ansatz_parameters), key_register, inplace=True)

        write_classical_data(list(known_plaintext), self, target_qubits=list(text_register))

        self.barrier()
        self.compose(QuantumSDES(key=key_register, data=text_register))
        self.barrier()

        classical_register = ClassicalRegister(8)
        self.add_register(classical_register)
        self.measure(text_register, classical_register)

    def compute_hamiltonian(self, ansatz_parameters: list[float]):
        self.assign_parameters(ansatz_parameters)

        measurements = self.simulator.run(self, shots=20, memory=True).result().get_memory()
        # Calculate expected value of hamiltonian
        total = 0
        for measurement in measurements:
            measurement = [i == '1' for i in measurement]
            total += self.hamiltonian.calculate(measurement)
        expected_value_of_hamiltonian = total / len(measurements)

        return expected_value_of_hamiltonian
