import random
import unittest

from classical.S_DES import *


# noinspection PyMethodMayBeStatic
class S_DES_Test(unittest.TestCase):
    def test_sdes(self):
        for _ in range(50):
            bits = tuple(to_bits(random.randint(0, 2 ** 8 - 1), 8))
            key = tuple(to_bits(random.randint(0, 2 ** 10 - 1), 10))

            encrypted = encrypt(bits, key)
            decrypted = decrypt(encrypted, key)

            self.assertEqual(bits, decrypted)
