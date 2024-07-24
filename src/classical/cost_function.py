import numpy as np
from quantum.ansatz import A_ansatz_Y_Cz_model
# code inspired by https://joshuagoings.com/2020/08/20/VQE/#make-that-hamiltonian

Pauli_Z = np.array([[ 1 + 0.0j, 0],
               [ 0,-1]])
X_gate = np.array([[0,1],
                   [1,0]])
I_gate = np.array([[1,0],
                   [0,1]])
parameter_num = 10
def create_hamiltonian(ciphertext):
    V = []
    for num in ciphertext:
        V.append(num)
    w01 = 1 if V[0] != V[1] else -1
    w06 = 1 if V[0] != V[6] else -1
    w07 = 1 if V[0] != V[7] else -1
    w13 = 1 if V[1] != V[3] else -1
    w17 = 1 if V[1] != V[7] else -1
    w24 = 1 if V[2] != V[4] else -1
    w25 = 1 if V[2] != V[5] else -1
    w27 = 1 if V[2] != V[7] else -1
    w34 = 1 if V[3] != V[4] else -1
    w36 = 1 if V[3] != V[6] else -1
    w45 = 1 if V[4] != V[5] else -1
    w56 = 1 if V[5] != V[6] else -1
    t = [] 
    for i in range(len(V)):
        t.append(0.5 if V[i] == 1 else -0.5)
    H = w01*np.kron(I_gate,I_gate,I_gate,I_gate,I_gate,I_gate,Pauli_Z,Pauli_Z) + w06*np.kron(I_gate,Pauli_Z,I_gate,I_gate,I_gate,I_gate,I_gate,Pauli_Z) + w07*np.kron(Pauli_Z,I_gate,I_gate,I_gate,I_gate,I_gate,I_gate,Pauli_Z)+w13*np.kron(I_gate,I_gate,I_gate,I_gate,Pauli_Z,I_gate,Pauli_Z,I_gate)+w17*np.kron(Pauli_Z,I_gate,I_gate,I_gate,I_gate,I_gate,Pauli_Z,I_gate)+w24*np.kron(I_gate,I_gate,I_gate,Pauli_Z,I_gate,Pauli_Z,I_gate,I_gate)+w25*np.kron(I_gate,I_gate,Pauli_Z,I_gate,I_gate,Pauli_Z,I_gate,I_gate)+w27*np.kron(Pauli_Z,I_gate,I_gate,I_gate,I_gate,Pauli_Z,I_gate,I_gate)+w34*np.kron(I_gate,I_gate,I_gate,Pauli_Z,Pauli_Z,I_gate,I_gate,I_gate)+w36*np.kron(I_gate,Pauli_Z,I_gate,I_gate,Pauli_Z,I_gate,I_gate,I_gate)+w45*np.kron(I_gate,I_gate,Pauli_Z,Pauli_Z,I_gate,I_gate,I_gate,I_gate)+w56*np.kron(I_gate,Pauli_Z,Pauli_Z,I_gate,I_gate,I_gate,I_gate,I_gate)
    H += t[0]*np.kron(I_gate,I_gate,I_gate,I_gate,I_gate,I_gate,I_gate,Pauli_Z)+t[1]*np.kron(I_gate,I_gate,I_gate,I_gate,I_gate,I_gate,Pauli_Z,I_gate)+t[2]*np.kron(I_gate,I_gate,I_gate,I_gate,I_gate,Pauli_Z,I_gate,I_gate)+t[3]*np.kron(I_gate,I_gate,I_gate,I_gate,Pauli_Z,I_gate,I_gate,I_gate)+t[4]*np.kron(I_gate,I_gate,I_gate,Pauli_Z,I_gate,I_gate,I_gate,I_gate)+t[5]*np.kron(I_gate,I_gate,Pauli_Z,I_gate,I_gate,I_gate,I_gate,I_gate)+t[6]*np.kron(I_gate,Pauli_Z,I_gate,I_gate,I_gate,I_gate,I_gate,I_gate)+t[7]*np.kron(Pauli_Z,I_gate,I_gate,I_gate,I_gate,I_gate,I_gate,I_gate)
    lowest_energy = np.linalg.eigvalsh(H)[0]
    return H,lowest_energy

def cost_function(theta_list,ansatz,Hmol,guess):
    circuit = ansatz(theta_list)
    guess = 

    



