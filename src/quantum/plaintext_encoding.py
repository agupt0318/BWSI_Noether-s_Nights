from qiskit import QuantumCircuit, QuantumRegister


def plaintext_circuit(plaintext):
    register = QuantumRegister(len(plaintext))
    circuit = QuantumCircuit(register)
    for i in range(len(plaintext)):
        if int(plaintext[i]) == 1:
            circuit.x(i)
    return circuit
