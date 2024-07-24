import random
import unittest

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from classical.S_DES import *
from quantum.S_DES import QuantumS_DES


# noinspection PyMethodMayBeStatic
class S_DES_Test(unittest.TestCase):
    def generate_random_key(self) -> bitstring_10:
        # noinspection PyTypeChecker
        return tuple(to_bits(random.randint(0, 2 ** 10 - 1), 10))

    def generate_random_message(self) -> bitstring_8:
        # noinspection PyTypeChecker
        return tuple(to_bits(random.randint(0, 2 ** 8 - 1), 8))

    def test_sdes_reversible(self):
        for _ in range(100):
            message = self.generate_random_message()
            key = self.generate_random_key()

            encrypted_message = encrypt_sdes(message, key)
            decrypted_message = decrypt(encrypted_message, key)

            self.assertEqual(message, decrypted_message)

    def test_q_sdes(self):
        key_register = QuantumRegister(10, name='key')
        data_register = QuantumRegister(8, name='data')
        q_sdes = QuantumS_DES(key_register, data_register)

        # noinspection PyTypeChecker
        simulator = AerSimulator(method="statevector")

        for _ in range(25):
            # The plaintext in big-endian order
            message = self.generate_random_message()
            # The key in big-endian order
            key = self.generate_random_key()

            print(f'Testing message = {to_string(message)}, key = {to_string(key)}')

            actual_encrypted_message = encrypt_sdes(message, key)

            # Set bits for message
            register_prep_circuit = QuantumCircuit(q_sdes.key_register, q_sdes.data_register)
            register_prep_circuit.x([q_sdes.data_register[i] for i, bit in enumerate(message) if bit])
            register_prep_circuit.x([q_sdes.key_register[i] for i, bit in enumerate(key) if bit])
            register_prep_circuit.barrier()

            full_circuit = register_prep_circuit.compose(q_sdes)
            full_circuit.measure_all()

            simulation_result = simulator.run(full_circuit, shots=1, memory=True).result()
            quantum_encrypted_message = tuple(i == '1' for i in simulation_result.get_memory()[0][:8][::-1])

            self.assertEqual(actual_encrypted_message, quantum_encrypted_message)


def normalize_arr(arr: list[complex]):
    """
    Normalizes an array of complex numbers. If the array is all zeroes, this throws a DivisionByZeroError.
    """
    factor = sum(
        abs(i) ** 2 for i in arr) ** 0.5  # Note that the builtin absolute value works on complex numbers yay!
    return [i / factor for i in arr]


def to_string(bits: tuple[bool, ...]) -> str:
    return ''.join(map(lambda i: '1' if i else '0', bits))


def create_statevector_from_amplitudes(num_qubits: int, amplitudes: dict[int, complex]):
    """
    Creates a Statevector with the given number qubits from a dictionary mapping states to their corresponding
    (possibly non-normalized) complex amplitudes. If all values in amplitudes are zero, this throws a
    DivisionByZeroError.

    Example:
        create_superposition(3, {0b000: 1, 0b111: -1}) yields 1/√2 (|000> - |111>)
    """
    return Statevector(normalize_arr([amplitudes.get(i, 0) for i in range(2 ** num_qubits)]))


def get_states_for_amplitudes(statevector: Statevector, state_mapper=lambda x: x) -> dict[
    np.complex128, set[int]]:
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
