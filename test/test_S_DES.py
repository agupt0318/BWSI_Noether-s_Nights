import unittest

from S_DES import *

class S_DES_Test(unittest.TestCase):
    def test_key_generation(self):
        initial_key = tuple(i + 1 for i in range(10))
        K1, K2 = generate_sub_keys(initial_key)

        print(K1, K2)
