from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator

from classical.CostFunction import construct_hamiltonian_for_ciphertext
from classical.S_DES import bitstring_8
from classical.util import bitstring_10, bits_from_string
from quantum.S_DES import QuantumSDES
from quantum.ansatz import A_ansatz_Y_Cz_model
from quantum.util import write_classical_data


class VQE_crypto(QuantumCircuit):
    def __init__(self, known_plaintext: bitstring_8, known_ciphertext: bitstring_8):
        self.simulator = AerSimulator()

        self.hamiltonian = construct_hamiltonian_for_ciphertext(known_ciphertext)

        key_register = self.key_register = QuantumRegister(10)
        text_register = self.text_register = QuantumRegister(8)

        super().__init__(key_register, text_register)

        ansatz_parameters = self.ansatz_parameters = [Parameter(f'ansatz_param_{i}') for i in range(10)]
        self.compose(A_ansatz_Y_Cz_model(ansatz_parameters), key_register, inplace=True)

        write_classical_data(list(known_plaintext), self, target_qubits=list(text_register))

        self.barrier()
        self.compose(QuantumSDES(key=key_register, data=text_register))
        self.barrier()

        self.measure_all()

    def run(self, ansatz_parameters: list[float]) -> tuple[float, bitstring_10]:
        measurements = self.simulator.run(
            self.assign_parameters(ansatz_parameters),
            shots=20,
            memory=True
        ).result().get_memory()
        # Calculate expected value of hamiltonian

        total = 0
        keys_found: dict[str, int] = dict()
        for measurement in measurements:
            measured_key_str = measurement[:10]
            measured_data_bits = list(bits_from_string(measurement[10:]))
            total += self.hamiltonian.calculate(measured_data_bits)
            if measured_key_str not in keys_found:
                keys_found[measured_key_str] = 0
            keys_found[measured_key_str] += 1

        expected_value_of_hamiltonian = total / len(measurements)

        most_common_key = max(keys_found.keys(), key=lambda k: keys_found[k])

        # noinspection PyTypeChecker
        return expected_value_of_hamiltonian, bits_from_string(most_common_key)
