import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit import Qubit


def permute_in_place(reg: QuantumRegister, *permutation: int) -> QuantumCircuit:
    """
    Permutes some qubits in place, with reg[i] -> reg[permutation[i]]. reg and permutation should be of the same length
    :param reg: The quantum register to permute
    :param permutation: A list of integers describing the permutation
    :return: A quantum circuit
    """
    circuit = QuantumCircuit(reg)

    num_qubits = circuit.num_qubits

    assert num_qubits == len(permutation)

    arr = [permutation.index(i + 1) for i in range(len(permutation))]

    for i in reversed(range(num_qubits)):
        target_index = arr.index(i)
        if i != target_index:
            arr[i], arr[target_index] = arr[target_index], arr[i]
            circuit.swap(i, target_index)

    return circuit


# Can you do this one? Put in_reg[permutation[i]] into out_reg[i] for i
def permute_to_output(in_reg: QuantumRegister, out_reg: QuantumRegister, *permutation: int) -> QuantumCircuit:
    qc = QuantumCircuit(in_reg, out_reg)

    numInputQubits = len(circuit[0].num_qubits)

    for i in range(numInputQubits):
        qc.cx(i, i + 1)


# class QuantumS_DES(QuantumCircuit):
#     def __init__(self):
#         key_register = self.key_register = QuantumRegister(10, name='key_register')
#         data_register = self.data_register = QuantumRegister(8, name='data_register')
#
#         # I'm not sure how many we will need
#         ancilla_register = self.ancilla_register = AncillaRegister(8, name='ancilla_register')
#
#         super().__init__(key_register, data_register)
#
#         [x0, x1, x2, x3, x4, x5, x6, x7] = [i for i in data_register]
#         [k0, k1, k2, k3, k4, k5, k6, k7, k8, k9] = [i for i in key_register]
#
#         K1_left: list[Qubit] = [k1, k2, k3, k4, k5, k6, k7, k8, k9, k0]
#         K1_right: list[Qubit] = [k9, k0, k1, k2, k3, k4, k5, k6, k7, k8, k9]
#         K2_left: list[Qubit] = [k2, k3, k4, k5, k6, k7, k8, k9, k0, k1]
#         K2_right: list[Qubit] = [k8, k9, k0, k1, k2, k3, k4, k5, k6, k7]
#
#         # Encryption part
#
#         # x1' = x6 XOR k0 then put through S0 box then


if __name__ == '__main__':
    circuit = QuantumS_DES()
    circuit.draw('mpl')
    plt.show()
