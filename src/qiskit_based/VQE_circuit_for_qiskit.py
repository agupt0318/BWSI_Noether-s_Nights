from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector

from classical.util import bitstring_8
from quantum.QuantumSDES import QuantumSDES
from quantum.ansatz import A_ansatz_Y_Cz_model
from quantum.util import write_classical_data


class VQE_circuit_qiskit(QuantumCircuit):
    def __init__(
            self,
            known_plaintext: bitstring_8,
    ):
        key_register = self.key_register = QuantumRegister(10)
        text_register = self.text_register = QuantumRegister(8)

        super().__init__(key_register, text_register)

        self.ansatz_parameters = ParameterVector("theta", length=10)
        self.compose(A_ansatz_Y_Cz_model([*self.ansatz_parameters]), key_register, inplace=True)

        write_classical_data(list(known_plaintext), self, target_qubits=list(text_register))

        self.barrier()
        self.compose(QuantumSDES(key=key_register, data=text_register), inplace=True)
        self.barrier()
