import random

bitstring_4 = tuple[bool, bool, bool, bool]
bitstring_8 = tuple[bool, bool, bool, bool, bool, bool, bool, bool]
bitstring_10 = tuple[bool, bool, bool, bool, bool, bool, bool, bool, bool, bool]


def generate_random_key() -> bitstring_10:
    # noinspection PyTypeChecker
    return tuple(to_bits(random.randint(0, 2 ** 10 - 1), 10))


def generate_random_message() -> bitstring_8:
    # noinspection PyTypeChecker
    return tuple(to_bits(random.randint(0, 2 ** 8 - 1), 8))


def bits_to_string(bits: tuple[bool, ...]) -> str:
    return ''.join(map(lambda i: '1' if i else '0', bits))


def bits_from_string(string: str) -> tuple[bool, ...]:
    return tuple(i == '1' for i in string)


def from_bits(bits: list[int]):
    """
    Converts a big-endian list of bits into an integer.
    """
    return int(''.join(map(lambda i: '1' if i else '0', bits)), 2)


def to_bits(num: int, num_bits: int):
    """
    Converts a number to a list of bits in big-endian
    """
    bits = list(bits_from_string('{0:b}'.format(num)))
    # Pad bits to the right length
    bits = [False] * (num_bits - len(bits)) + bits
    return bits


def hamming_distance(a: tuple[bool, ...], b: tuple[bool, ...]) -> int:
    """
    Calculates the hamming distance (number of different bits) in two bitstrings
    """
    assert len(a) == len(b)
    return sum(0 if a == b else 1 for a, b in zip(a, b))
