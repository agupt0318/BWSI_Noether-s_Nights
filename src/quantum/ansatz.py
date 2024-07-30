from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, Gate
from qiskit.circuit.library import CZGate, CXGate, CYGate


class Ansatz(QuantumCircuit):
    def __init__(self, num_qubits: int, ansatz_type: str):
        gate, circular = Ansatz.interpret_ansatz_type(ansatz_type)

        register = QuantumRegister(num_qubits)
        super().__init__(register)

        for i in range(num_qubits):
            self.h(register[i])
            self.ry(Parameter(f'theta_{i}'), register[i])

            if i == num_qubits - 1:
                if circular:
                    self.cz(register[i], register[0])
            else:
                self.append(
                    gate,
                    qargs=[register[i], register[i + 1]],
                    cargs=[],
                    copy=True
                )

    @staticmethod
    def interpret_ansatz_type(ansatz_type: str) -> tuple[Gate, bool]:
        circular = False

        if ansatz_type == 'A-CX':
            gate = CXGate()
            circular = True
        elif ansatz_type == 'A-CY':
            gate = CYGate()
            circular = True
        elif ansatz_type == 'A-CZ':
            gate = CZGate()
            circular = True
        elif ansatz_type == 'B-CX':
            gate = CXGate()
        elif ansatz_type == 'B-CY':
            gate = CYGate()
        elif ansatz_type == 'B-CZ':
            gate = CZGate()
        else:
            raise ValueError(f'Invalid ansatz type: {ansatz_type}')

        return gate, circular
