import random
import unittest

import numpy as np
from qiskit.quantum_info import Statevector

from classical.S_DES import *
from quantum.possible import QuantumS_DES


# noinspection PyMethodMayBeStatic
class S_DES_Test(unittest.TestCase):
    def test_sdes(self):
        for _ in range(50):
            bits = tuple(to_bits(random.randint(0, 2 ** 8 - 1), 8))
            key = tuple(to_bits(random.randint(0, 2 ** 10 - 1), 10))

            encrypted = encrypt(bits, key)
            decrypted = decrypt(encrypted, key)

            self.assertEqual(bits, decrypted)

    def test_qsdes(self):
        qsdes_circuit = QuantumS_DES()

        bits = tuple(to_bits(random.randint(0, 2 ** 8 - 1), 8))
        key = tuple(to_bits(random.randint(0, 2 ** 10 - 1), 10))

        actual_encrypted = encrypt(bits, key)

        sv = create_statevector_from_amplitudes(
            18,
            {
                from_bits([
                    *bits[::-1], *key[::-1]
                ]): 1
            }
        )
        a = get_states_for_amplitudes(sv.evolve(qsdes_circuit))
        print(
            to_bits([*[*a.values()][0]][0], 18),
            actual_encrypted
        )


def normalize_arr(arr: list[complex]):
    """
    Normalizes an array of complex numbers. If the array is all zeroes, this throws a DivisionByZeroError.
    """
    factor = sum(abs(i) ** 2 for i in arr) ** 0.5  # Note that the builtin absolute value works on complex numbers yay!
    return [i / factor for i in arr]


def create_statevector_from_amplitudes(num_qubits: int, amplitudes: dict[int, complex]):
    """
    Creates a Statevector with the given number qubits from a dictionary mapping states to their corresponding
    (possibly non-normalized) complex amplitudes. If all values in amplitudes are zero, this throws a
    DivisionByZeroError.

    Example:
        create_superposition(3, {0b000: 1, 0b111: -1}) yields 1/√2 (|000> - |111>)
    """
    return Statevector(normalize_arr([amplitudes.get(i, 0) for i in range(2 ** num_qubits)]))


def get_states_for_amplitudes(statevector: Statevector, state_mapper=lambda x: x) -> dict[np.complex128, set[int]]:
    """
    Returns a dictionary mapping each unique amplitudes in the statevector to their corresponding states. States with
    zero amplitude are ignored.
    """
    final_state = statevector.data
    result = dict()

    for amplitude, state in zip(final_state, range(len(final_state))):
        amplitude = np.round(amplitude, 3)
        if amplitude == 0:
            continue

        if amplitude not in result:
            result[amplitude] = set()

        result[amplitude].add(state_mapper(state))

    return result
