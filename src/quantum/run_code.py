import numpy as np

from classical.Optimization import gradient_descent
from classical.S_DES import encrypt_sdes
from quantum.VQE import VQE_crypto


def run():
    known_plaintext = (True, False, True, True, False, True, True, False)
    key = (False, True, False, True, False, True, False, True, False, True)
    known_ciphertext = encrypt_sdes(known_plaintext, key)

    vqe_solver = VQE_crypto(known_plaintext, known_ciphertext)

    learning_rate = 1.08

    best_ansatz_parameters = gradient_descent(
        guess=np.array([1] + ([0] * 9)),
        cost_function=lambda x: vqe_solver.compute_hamiltonian(x),
        learning_rate=learning_rate,
        cost_cutoff=-9
    )




if __name__ == "__main__":
    run()
