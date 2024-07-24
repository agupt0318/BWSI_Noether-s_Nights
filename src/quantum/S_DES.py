from typing import Sequence

from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Operation, CircuitInstruction, InstructionSet, Barrier
from qiskit.circuit.library import C4XGate, C3XGate
from qiskit.circuit.quantumcircuit import QubitSpecifier, ClbitSpecifier


class QuantumSDES(QuantumCircuit):
    def __init__(self, key: QuantumRegister, data: QuantumRegister, use_debug_barriers=False):
        self.use_debug_barriers = use_debug_barriers
        self.key_register = key
        self.data_register = data

        assert len(key) == 10
        assert len(data) == 8

        super().__init__(key, data)

        # Write key for SO box (K1)
        self.cx(key[0], data[6])
        self.cx(key[6], data[3])
        self.cx(key[8], data[7])
        self.cx(key[3], data[4])

        # Apply S0 box (K1)
        self.cx(data[4], data[0])
        self.append(C4XGate(), [data[3], data[4], data[6], data[7], data[0]])
        self.ccx(data[6], data[7], data[0])
        self.x(data[6])
        self.ccx(data[3], data[6], data[0])
        self.x(data[7])
        self.ccx(data[6], data[7], data[1])
        self.x(data[6])
        self.append(C4XGate(), [data[3], data[4], data[6], data[7], data[1]])
        self.x(data[3])
        self.ccx(data[3], data[6], data[1])
        self.x(data[4])
        self.ccx(data[4], data[6], data[1])
        self.x([data[7], data[4], data[3]])

        # Unwrite key for S0 box (K1)
        self.cx(key[0], data[6])
        self.cx(key[6], data[3])
        self.cx(key[8], data[7])
        self.cx(key[3], data[4])

        # Write key for S1 box (K1)
        self.cx(key[7], data[7])
        self.cx(key[2], data[4])
        self.cx(key[9], data[6])
        self.cx(key[5], data[3])

        # Apply S1 box (K1)
        self.cx(data[4], data[2])
        self.append(C3XGate(), [data[3], data[6], data[7], data[5]])
        self.x(data[3])
        self.ccx(data[3], data[7], data[5])
        self.ccx(data[3], data[7], data[2])
        self.x(data[4])
        self.append(C4XGate(), [data[3], data[4], data[6], data[7], data[2]])
        self.ccx(data[3], data[6], data[5])
        self.x([data[3], data[6]])
        self.ccx(data[3], data[6], data[2])
        self.x([data[7], data[4]])
        self.append(C3XGate(), [data[3], data[4], data[7], data[5]])
        self.x([data[6], data[7]])

        # Unwrite key for S1 box (K1)
        self.cx(key[7], data[7])
        self.cx(key[2], data[4])
        self.cx(key[9], data[6])
        self.cx(key[5], data[3])

        # Write key for S0 box (K2)
        self.cx(key[7], data[0])
        self.cx(key[2], data[1])
        self.cx(key[5], data[5])
        self.cx(key[4], data[2])

        # Apply S0 box (K2)
        self.cx(data[2], data[6])
        self.append(C4XGate(), [data[5], data[2], data[0], data[1], data[6]])
        self.ccx(data[0], data[5], data[6])
        self.x(data[0])
        self.ccx(data[0], data[1], data[6])
        self.x(data[5])
        self.ccx(data[0], data[5], data[3])
        self.x(data[0])
        self.append(C4XGate(), [data[5], data[2], data[0], data[1], data[3]])
        self.x(data[1])
        self.ccx(data[0], data[1], data[3])
        self.x(data[2])
        self.ccx(data[0], data[2], data[3])
        self.x([data[1], data[2], data[5]])

        # Unwrite key for S0 box (K2)
        self.cx(key[7], data[0])
        self.cx(key[2], data[1])
        self.cx(key[5], data[5])
        self.cx(key[4], data[2])

        # Write key for S1 box (K2)
        self.cx(key[9], data[5])
        self.cx(key[1], data[2])
        self.cx(key[8], data[0])
        self.cx(key[0], data[1])

        # Apply S1 box (K2)
        self.cx(data[2], data[4])
        self.append(C3XGate(), [data[1], data[0], data[5], data[7]])
        self.x(data[1])
        self.ccx(data[1], data[5], data[7])
        self.ccx(data[1], data[5], data[4])
        self.x(data[2])
        self.append(C4XGate(), [data[5], data[0], data[1], data[2], data[4]])
        self.ccx(data[0], data[1], data[7])
        self.x([data[0], data[1]])
        self.ccx(data[0], data[1], data[4])
        self.x([data[2], data[5]])
        self.append(C3XGate(), [data[5], data[1], data[2], data[7]])
        self.x([data[0], data[5]])

        # Unwrite key for S1 box (K2)
        self.cx(key[9], data[5])
        self.cx(key[1], data[2])
        self.cx(key[8], data[0])
        self.cx(key[0], data[1])

        # Swaps to put the bits in the right position
        self.swap(data[0], data[6])
        self.swap(data[1], data[3])
        self.swap(data[2], data[4])
        self.swap(data[5], data[7])

    # I override this method to add barriers before every instruction
    # for debugging purposes. Calling optimized() returns the circuit
    # with these barriers removed.

    # noinspection SpellCheckingInspection
    def append(
            self,
            instruction: Operation | CircuitInstruction,
            qargs: Sequence[QubitSpecifier] | None = None,
            cargs: Sequence[ClbitSpecifier] | None = None,
            *,
            copy: bool = True,
    ) -> InstructionSet:
        if self.use_debug_barriers:
            self._current_scope().append(CircuitInstruction(
                Barrier(18),
                [*self.key_register, *self.data_register],
                ()
            ))

        return super().append(instruction, qargs, cargs, copy=copy)

    def debug_mpl(self):
        self.draw(output='mpl', plot_barriers=False, fold=43)
        plt.show()

    def debug_print(self):
        print(self.draw(output='text', plot_barriers=False, fold=145, vertical_compression="high"))


if __name__ == '__main__':
    circuit = QuantumSDES(QuantumRegister(10), QuantumRegister(8), use_debug_barriers=True)
    circuit.debug_mpl()
