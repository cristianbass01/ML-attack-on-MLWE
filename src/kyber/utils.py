from sympy import isprime, totient
from typing import List

def xor_bytes(a: bytes, b: bytes) -> bytes:
    """
    XOR two byte arrays, assume that they are
    of the same length

    :param a: First byte array
    :param b: Second byte array
    :return: XOR of the two byte arrays
    """
    assert len(a) == len(b)
    return bytes(a ^ b for a, b in zip(a, b))


def select_bytes(a: bytes, b: bytes, cond: bool) -> bytes:
    """
    Select between the bytes a or b depending
    on whether cond is False or True

    :param bytes a: First byte array
    :param bytes b: Second byte array
    :param bool cond: Condition to select a or b
    :return: Selected byte array
    """
    assert len(a) == len(b)
    out = [0] * len(a)
    cw = -cond % 256
    for i in range(len(a)):
        out[i] = a[i] ^ (cw & (a[i] ^ b[i]))
    return bytes(out)

def get_divisors(n: int) -> List[int]:
    """Returns all divisors of n."""
    divisors = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
    return sorted(divisors)

def find_root_of_unity(n: int, q: int) -> int:
    """
    Finds a primitive nth root of unity modulo q.
    
    :param int n: The order of the root of unity
    :param int q: The modulus
    :return: A primitive nth root of unity modulo q
    :rtype: int
    """
    if not isprime(q):
        raise ValueError("q should be prime for best results.")
    
    if n > totient(q):
        raise ValueError("No valid nth root of unity exists.")

    divisors = get_divisors(n)[:-1]  # Exclude n itself
    for omega in range(2, q):
        if pow(omega, n, q) == 1:
            if all(pow(omega, d, q) != 1 for d in divisors):
                return omega
    return None  # No root found