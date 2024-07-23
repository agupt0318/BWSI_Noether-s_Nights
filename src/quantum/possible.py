from qiskit import QuantumCircuit, QuantumRegister


class QuantumS_DES(QuantumCircuit):
    def __init__(self):
        key_register = self.key_register = QuantumRegister(10, name='key_register')
        data_register = self.data_register = QuantumRegister(8, name='data_register')

        super().__init__(key_register, data_register)
        # Copied from circuit diagram


if __name__ == '__main__':
    circuit = QuantumS_DES()
    circuit.draw('mpl')
    # plt.show()
