from typing import Callable

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit


def write_classical_data(bits: list[bool], circuit: QuantumCircuit, target_qubits: list[Qubit]):
    assert len(bits) == len(target_qubits)

    circuit.x([target_qubits[i] for i, bit in enumerate(bits) if bit])


class Hamiltonian:
    def __init__(self, calculation_function: Callable[[list[float]], float]):
        """
        :param calculation_function: A function taking the measurement in the Z basis and returning the value of the
                                     Hamiltonian
        """
        self.calculation_function = calculation_function

    def calculate(self, bits: list[bool]):
        values = [-1 if i else 1 for i in bits]
        return self.calculation_function(values)
