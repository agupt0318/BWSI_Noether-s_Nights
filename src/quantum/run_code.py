import numpy as np

from classical.Optimization import gradient_descent
from classical.S_DES import encrypt_sdes
from classical.util import bits_to_string
from quantum.VQE import VQE_crypto


def run():
    known_plaintext = (True, False, True, True, False, True, True, False)
    key = (False, True, False, True, False, True, False, True, False, True)
    known_ciphertext = encrypt_sdes(known_plaintext, key)

    print(f'Testing with key={bits_to_string(key)}')

    vqe_solver = VQE_crypto(known_plaintext, known_ciphertext)

    learning_rate = 1.08

    best_ansatz_parameters = gradient_descent(
        known_key=key,
        guess=np.array([1] + ([0] * 9), dtype=float),
        cost_function=lambda x: vqe_solver.run(x),
        learning_rate=learning_rate,
        cost_cutoff=-9,
        num_iterations=20
    )


if __name__ == "__main__":
    run()
