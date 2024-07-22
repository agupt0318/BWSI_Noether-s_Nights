bitstring_4 = tuple[bool, bool, bool, bool]
bitstring_8 = tuple[bool, bool, bool, bool, bool, bool, bool, bool]
bitstring_10 = tuple[bool, bool, bool, bool, bool, bool, bool, bool, bool, bool]


def from_bits(bits: list[int]):
    """
    Converts a big-endian list of bits into an integer.
    """
    return int(''.join(map(lambda i: '1' if i else '0', bits)), 2)


def to_bits(num: int, num_bits: int):
    """
    Converts a number to a list of bits in big-endian
    """
    bits = list(map(lambda i: i == '1', '{0:b}'.format(num)))
    # Pad bits to the right length
    bits = [False] * (num_bits - len(bits)) + bits
    return bits


def permute(bits: tuple, *permutation: int) -> tuple:
    # noinspection PyTypeChecker
    return tuple(bits[i - 1] for i in permutation)


def P10(bits: bitstring_10) -> bitstring_10:
    # noinspection PyTypeChecker
    return permute(bits, 3, 5, 2, 7, 4, 10, 1, 9, 8, 6)


def P8(bits: bitstring_10) -> bitstring_8:
    # noinspection PyTypeChecker
    return permute(bits, 6, 3, 7, 4, 8, 5, 10, 9)


def P4(bits: bitstring_4) -> bitstring_4:
    # noinspection PyTypeChecker
    return permute(bits, 2, 4, 3, 1)


def shift1(bits: bitstring_10) -> bitstring_10:
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = bits
    return k2, k3, k4, k5, k1, k7, k8, k9, k10, k6


def shift2(bits: bitstring_10) -> bitstring_10:
    return shift1(shift1(bits))


def generate_sub_keys(key: bitstring_10) -> tuple[bitstring_8, bitstring_8]:
    return (
        P8(shift1(P10(key))),
        P8(shift2(shift1(P10(key))))
    )


def IP(bits: bitstring_8) -> bitstring_8:
    # noinspection PyTypeChecker
    return permute(bits, 2, 6, 3, 1, 4, 8, 5, 7)


def IP_inverse(bits: bitstring_8) -> bitstring_8:
    # noinspection PyTypeChecker
    return permute(bits, 4, 1, 3, 5, 7, 2, 8, 6)


def EP(bits: bitstring_4) -> bitstring_8:
    # noinspection PyTypeChecker
    return permute(bits, 4, 1, 2, 3, 2, 3, 4, 1)


def xor(a: tuple, b: tuple) -> tuple:
    return tuple(i != j for (i, j) in zip(a, b))


def apply_S_box(bits: bitstring_4, box: list[list[int]]) -> int:
    row_index = from_bits([bits[0], bits[3]])
    col_index = from_bits([bits[1], bits[2]])
    return box[row_index][col_index]


def split_8_to_4(bits: bitstring_8) -> tuple[bitstring_4, bitstring_4]:
    # noinspection PyTypeChecker
    left: bitstring_4 = tuple(bits[:4])
    # noinspection PyTypeChecker
    right: bitstring_4 = tuple(bits[4:])

    return left, right


def F(bits: bitstring_4, key: bitstring_8) -> bitstring_4:
    # noinspection PyTypeChecker
    xor_ed: bitstring_8 = xor(EP(bits), key)
    # noinspection PyTypeChecker
    left, right = split_8_to_4(xor_ed)

    S0 = [
        [1, 0, 3, 2],
        [3, 2, 1, 0],
        [0, 2, 1, 3],
        [3, 1, 3, 2]
    ]
    S1 = [
        [0, 1, 2, 3],
        [2, 0, 1, 3],
        [3, 0, 1, 0],
        [2, 1, 0, 3]
    ]

    # noinspection PyTypeChecker
    bits_after_S_box: bitstring_4 = tuple([
        *to_bits(apply_S_box(left, S0), 2),
        *to_bits(apply_S_box(right, S1), 2)
    ])

    return P4(bits_after_S_box)


def f_K(L: bitstring_4, R, key: bitstring_8) -> tuple[bitstring_4, bitstring_4]:
    # noinspection PyTypeChecker
    return


def encrypt(plaintext_bits: bitstring_8, key: bitstring_10) -> bitstring_8:
    K1, K2 = generate_sub_keys(key)

    L, R = split_8_to_4(IP(plaintext_bits))

    L, R = xor(L, F(R, K1)), R

    L, R = R, L

    L, R = xor(L, F(R, K2)), R

    # noinspection PyTypeChecker
    joined: bitstring_8 = tuple([*L, *R])

    ciphertext_bits = IP_inverse(joined)

    return ciphertext_bits


def decrypt(ciphertext_bits: bitstring_8, key: bitstring_10) -> bitstring_8:
    K1, K2 = generate_sub_keys(key)

    L, R = split_8_to_4(IP(ciphertext_bits))

    L, R = xor(L, F(R, K2)), R

    L, R = R, L

    L, R = xor(L, F(R, K1)), R

    # noinspection PyTypeChecker
    joined: bitstring_8 = tuple([*L, *R])

    plaintext_bits = IP_inverse(joined)

    return plaintext_bits
