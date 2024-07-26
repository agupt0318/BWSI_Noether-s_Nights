from typing import Union

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator

from classical.S_DES import bitstring_8
from classical.util import bitstring_10, bits_from_string, bits_to_string
from quantum.ansatz import A_ansatz_Y_Cz_model
from quantum.quantum_sdes import QuantumSDES
from quantum.util import write_classical_data, Hamiltonian


class VQE_crypto(QuantumCircuit):
    def __init__(
            self,
            known_plaintext: bitstring_8,
            known_ciphertext: bitstring_8,
            hamiltonian: Hamiltonian,
            shots_per_estimate: int = 20,
    ):
        self.simulator = AerSimulator(method="statevector")
        self.hamiltonian = hamiltonian
        self.shots_per_estimate = shots_per_estimate
        self.known_ciphertext = known_ciphertext

        key_register = self.key_register = QuantumRegister(10)
        text_register = self.text_register = QuantumRegister(8)

        super().__init__(key_register, text_register)

        ansatz_parameters = self.ansatz_parameters = [Parameter(f'ansatz_param_{i}') for i in range(10)]
        self.compose(A_ansatz_Y_Cz_model(ansatz_parameters), key_register, inplace=True)

        write_classical_data(list(known_plaintext), self, target_qubits=list(text_register))

        self.barrier()
        self.compose(QuantumSDES(key=key_register, data=text_register), inplace=True)
        self.barrier()

        self.measure_all()

        self.solution: Union[None, bitstring_10] = None

    def run(self, ansatz_parameters: list[float]) -> tuple[float, bitstring_10]:
        measurements = self.simulator.run(
            self.assign_parameters(ansatz_parameters),
            shots=self.shots_per_estimate,
            memory=True
        ).result().get_memory()
        # Calculate expected value of hamiltonian

        total = 0
        keys_found: dict[str, int] = dict()
        for measurement in measurements:

            measured_key_str = bits_to_string(QuantumSDES.get_key_from_measurement(measurement))
            measured_data_bits = QuantumSDES.get_message_from_measurement(measurement)

            # if measured_data_bits == self.known_ciphertext:
            #     print(f'Found solution by lucky measurement: {measured_key_str}')
            #     self.solution = bits_from_string(measured_key_str)

            total += self.hamiltonian.calculate(list(measured_data_bits))
            if measured_key_str not in keys_found:
                keys_found[measured_key_str] = 0
            keys_found[measured_key_str] += 1

        expected_value_of_hamiltonian = total / len(measurements)

        most_common_key = max(keys_found.keys(), key=lambda k: keys_found[k])

        # noinspection PyTypeChecker
        return expected_value_of_hamiltonian, bits_from_string(most_common_key)
