from qiskit import QuantumCircuit, QuantumRegister

from classical.util import bitstring_8
from quantum.ansatz import Ansatz
from quantum.qsdes import QuantumSDES
from quantum.util import write_classical_data


class QuantumSDESAnsatz(QuantumCircuit):
    def __init__(self, known_plaintext: bitstring_8):
        # The ansatz we use for the key. The paper found that A-CZ is the most effective
        self.ansatz = Ansatz(10, 'A-CZ')

        # The registers used in the quantum circuit
        self.key_register = QuantumRegister(10)
        self.text_register = QuantumRegister(8)

        # Call QuantumCircuit's constructor with the registers we are using
        super().__init__(self.key_register, self.text_register)

        # Add the ansatz circuit
        self.compose(
            self.ansatz,
            qubits=self.key_register,
            inplace=True
        )

        # Write the plaintext
        write_classical_data(
            bits=list(known_plaintext),
            circuit=self,
            target_qubits=list(self.text_register)
        )

        # Add the S-DES circuit
        self.barrier()
        self.compose(
            QuantumSDES(
                key=self.key_register,
                data=self.text_register
            ),
            inplace=True
        )
        self.barrier()

        # Measure all
        self.measure_all()
