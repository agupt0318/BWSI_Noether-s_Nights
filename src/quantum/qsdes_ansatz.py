from qiskit import QuantumCircuit, QuantumRegister

from classical.util import bitstring_8
from quantum.QuantumSDES import QuantumSDES
from quantum.ansatz import A_ansatz_Y_Cz_model
from quantum.util import write_classical_data


class QuantumSDESAnsatz(QuantumCircuit):
    def __init__(self, known_plaintext: bitstring_8):
        self.ansatz = A_ansatz_Y_Cz_model()

        self.key_register = QuantumRegister(10)
        self.text_register = QuantumRegister(8)

        super().__init__(self.key_register, self.text_register)

        self.compose(
            self.ansatz,
            qubits=self.key_register,
            inplace=True
        )

        write_classical_data(
            bits=list(known_plaintext),
            circuit=self,
            target_qubits=list(self.text_register)
        )

        self.barrier()
        self.compose(
            QuantumSDES(
                key=self.key_register,
                data=self.text_register
            ),
            inplace=True
        )
        self.barrier()
