from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from classical.CostFunction import construct_hamiltonian
from quantum import S_DES
from quantum.ansatz import A_ansatz_Y_Cz_model
from quantum.plaintext_encoding import plaintext_circuit
from qiskit_aer import AerSimulator

def run():
    text = [1,0,1,1,0,1,1,0]
    key_register = QuantumRegister(10)
    text_register = QuantumRegister(8)
    circuit = QuantumCircuit(key_register, text_register) 

    circuit.compose(A_ansatz_Y_Cz_model(1,0,0,0,0,0,0,0,0,0), key_register, inplace=True)
    circuit.compose(plaintext_circuit(text), text_register, inplace=True)

    circuit.barrier()
    S_DES(circuit, key=key_register,data=text_register)
    circuit.barrier()

    classical_register = ClassicalRegister(8)
    circuit.add_register(classical_register)
    circuit.measure(text_register, classical_register)
    ciphertext = AerSimulator().run(circuit, shots=1, memory=True).result()

    H = construct_hamiltonian(ciphertext, 3) # 3 regular graph
    return None





if __name__ == "__main__":
    run()