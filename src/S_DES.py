bitstring_8 = tuple[bool, bool, bool, bool, bool, bool, bool, bool]
bitstring_10 = tuple[bool, bool, bool, bool, bool, bool, bool, bool, bool, bool]


def P10(bits: bitstring_10) -> bitstring_10:
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = bits
    return k3, k5, k2, k7, k4, k10, k1, k9, k8, k6


def P8(bits: bitstring_10) -> bitstring_8:
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = bits
    return k6, k3, k7, k4, k8, k5, k10, k9


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
