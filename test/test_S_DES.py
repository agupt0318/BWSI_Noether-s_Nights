import random
import unittest

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
