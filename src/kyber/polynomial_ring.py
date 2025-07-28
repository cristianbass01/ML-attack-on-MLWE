from typing import Union
import random
from .utils import find_root_of_unity
from typing import Optional, List, Union
from math import sqrt
import numpy as np
import math 

class PolynomialRing:
    def __init__(self, n: int, q: int):
        self.n = n
        self.q = q
        self.root_of_unity = None
        self.ntt_zetas = None
        self.ntt_f = None
        self.is_ntt = False
        self.use_mods = False
        self.coeffs = [0] * self.n

    def copy(self):
        return self._poly(self.coeffs.copy(), self.is_ntt)
    
    def _poly(self, coeffs: List[int], is_ntt: Optional[bool] = False) -> "PolynomialRing":
        poly = PolynomialRing(self.n, self.q)

        # initialize the NTT values
        poly.root_of_unity = self.root_of_unity
        poly.ntt_zetas = self.ntt_zetas
        poly.ntt_f = self.ntt_f
        poly.is_ntt = is_ntt

        poly.use_mods = self.use_mods

        poly.coeffs = coeffs
        return poly
    
    def zero(self) -> "PolynomialRing":
        """
        Return the zero polynomial
        """
        return self._poly([0] * self.n, self.is_ntt)

    def _initialize_ntt(self) -> "PolynomialRing":
        """
        Initialize the NTT zetas and f values
        """
        if self.root_of_unity is None:
            root = find_root_of_unity(self.n, self.q)
            if root is None:
                raise ValueError(f"Root of unity not found for {self.n = }, {self.q = }")
            self.root_of_unity = root

        if self.ntt_zetas is None:
            self.ntt_zetas = [pow(self.root_of_unity, self._br(i, int(math.log2(self.n))-1), self.q) for i in range(self.n // 2)]

        if self.ntt_f is None:
            self.ntt_f = pow(self.n // 2, -1, self.q)
        return self

    def __call__(self, coefficients: Union[int, List[int]], is_ntt: Optional[bool] = False) -> "PolynomialRing":
        """
        Helper function which right pads with zeros
        to allow polynomial construction as
        f = R([1,1,1])
        """
        if isinstance(coefficients, int):
            coefficients = [coefficients]
        elif not isinstance(coefficients, list):
            raise TypeError(f"Polynomials should be constructed from a list of integers, of length at most n = {self.n}")
        
        l = len(coefficients)
        if l > self.n:
            raise ValueError(f"Coefficients describe polynomial of degree greater than maximum degree {self.n}")
        elif l < self.n:
            coefficients = coefficients + [0] * (self.n - l)
        
        return self._poly(coefficients, is_ntt)
        
    def is_zero(self) -> bool:
        """
        Return if polynomial is zero: f = 0
        """
        return all([c == 0 for c in self.coeffs])

    def is_constant(self) -> bool:
        """
        Return if polynomial is constant: f = c
        """
        return all(c == 0 for c in self.coeffs[1:])

    def gen(self) -> "PolynomialRing":
        """
        Return the generator `x` of the polynomial ring

        :return: Generator of the polynomial ring (not in NTT form)
        """
        coeffs = [0, 1] + [0] * (self.n - 2)
        return self._poly(coeffs, False)
        
    def _sparsify(self, input_bytes: bytes, coefficients: List[int], max_hamming: int) -> List[int]:     
        nonzeros = [i for i, coeff in enumerate(coefficients) if coeff != 0]

        if len(nonzeros) > max_hamming:
            # Too many nonzero coordinates, randomly zero some out.
            extra = len(nonzeros) - max_hamming
            if len(input_bytes) == 0:
                seed = random.randint(0, 2**32 - 1)
            else:
                seed = int.from_bytes(input_bytes, "little")

            random.seed(seed)
            idxs = random.sample(nonzeros, extra)
            for idx in idxs:
                coefficients[idx] = 0

        # Required Hamming weight is higher than nonzeros in secret.
        assert max_hamming >= len([coeff for coeff in coefficients if coeff != 0])

        return coefficients
    
    def hamming_weight(self) -> int:
        """
        Computes the Hamming weight (number of nonzero elements) in the coefficient list.

        :return: The number of nonzero elements in the list.
        """
        return len([1 for coeff in self.coeffs if coeff != 0])
    
    def uniform(self, input_bytes: Optional[bytes] = None) -> "PolynomialRing":
        """
        Compute a random element of the polynomial ring with coefficients in the
        canonical range: ``[0, q-1]``

        :param bytes input_bytes: Optional byte array to seed the random number generator.
        :return: Random polynomial in the ring (not in NTT form)
        """
        if input_bytes is not None:
            seed = int.from_bytes(input_bytes, "little")
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        coeffs = rng.integers(0, self.q, size=self.n).tolist()

        return self._poly(coeffs, False)
    
    def ntt_sample(self, input_bytes: bytes) -> "PolynomialRing":
        """
        Algorithm 1 (Parse)
        https://pq-crystals.org/kyber/data/kyber-specification-round3-20210804.pdf

        Algorithm 6 (Sample NTT)

        Parse: B^* -> R

        :param bytes input_bytes: Byte array of length 3n/2
        :param max_hamming: Maximum Hamming weight of the polynomial. If set to -1,
                            the polynomial will not be sparsified. Default is -1.
        :return: Polynomial in NTT form
        """
        self._initialize_ntt()
        
        bit_length = math.ceil(math.log2(self.q))
        total_bits = len(input_bytes) * 8
        bit_index = 0
        coefficients = []  # will accumulate accepted coefficients

        # While we have not yet collected self.n coefficients...
        while len(coefficients) < self.n and (bit_index + bit_length) <= total_bits:
            # Read the next 'bit_length' bits as a candidate integer.
            candidate = 0
            for i in range(bit_length):
                byte_index = (bit_index + i) // 8
                bit_offset = (bit_index + i) % 8
                # Note: bits are read in little-endian order.
                candidate |= (((input_bytes[byte_index] >> bit_offset) & 1) << i)
            bit_index += bit_length
            
            if candidate < self.q:
                coefficients.append(candidate)
        
        if len(coefficients) < self.n:
            raise ValueError("Not enough bits in input_bytes to sample a full polynomial.")

        return self._poly(coefficients, is_ntt=True)
    
    def cbd(self, input_bytes: bytes, eta: int, max_hamming: Optional[int] = -1) -> "PolynomialRing":
        """
        Algorithm 2 (Centered Binomial Distribution)
        https://pq-crystals.org/kyber/data/kyber-specification-round3-20210804.pdf

        Algorithm 6 (Sample Poly CBD)

        Expects a byte array of length (eta * deg / 4)
        For Kyber, this is 64 eta.

        :param bytes input_bytes: Byte array of length 64 eta
        :param int eta: Error distribution parameter
        :param int max_hamming: The desired Hamming weight of the polynomial. Default is -1 (so no sparsification).
        :return: Polynomial
        """
        assert (self.n * eta) // 4 == len(input_bytes)

        coefficients = [0] * self.n
        b_int = int.from_bytes(input_bytes, "little")
        mask = (1 << eta) - 1
        mask2 = (1 << 2 * eta) - 1
        for i in range(self.n):
            x = b_int & mask2
            a = (x & mask).bit_count()
            b = ((x >> eta) & mask).bit_count()
            b_int >>= 2 * eta
            coefficients[i] = self.mod_q(a - b)

        if max_hamming != -1:
            coefficients = self._sparsify(input_bytes, coefficients, max_hamming)

        return self._poly(coefficients, False)
    
    def binary(self, input_bytes: bytes, max_hamming: Optional[int] = -1) -> "PolynomialRing":
        """
        Samples a polynomial with coefficients from a binary distribution [0, 1].

        :param bytes input_bytes: Random input bytes.
        :param int max_hamming: If set, enforces the specified Hamming weight.
        :return: Polynomial with binary coefficients.
        """
        assert len(input_bytes) == (self.n // 8)

        coefficients = [0] * self.n
        b_int = int.from_bytes(input_bytes, "little")
        for i in range(self.n):
            x = b_int & 1
            b_int >>= 1
            coefficients[i] = x

        if max_hamming != -1:
            coefficients = self._sparsify(input_bytes, coefficients, max_hamming)

        return self._poly(coefficients, False)
    
    def ternary(self, input_bytes: bytes, max_hamming: Optional[int] = -1) -> "PolynomialRing":
        """
        Samples a polynomial with coefficients from a ternary distribution [-1, 0, 1].

        Ternary uses 2 bits per coefficient, but the value 11 is discarded, so more than 2 bits are needed.
        On average, 4 pairs are needed every 3 coefficients = self.n * 4 / 3 pairs total.
        We calculate then the std:
            -> var(X) = (1-p)/p^2 = 4/9
            -> std(X) = 2/3
            -> std(S) = 2/3 * sqrt(n) 

        :param bytes input_bytes: Random input bytes.
        :param int max_hamming: If set, enforces the specified Hamming weight.
        :return: Polynomial with binary coefficients.
        """
        assert len(input_bytes) >= (self.n // 3) # average needed

        coefficients = [0] * self.n
        b_int = int.from_bytes(input_bytes, "little")
        i = 0
        while i < self.n:
            if b_int == 0:
                raise ValueError(f"Input bytes exhausted before filling all coefficients: {i}/{self.n}")
            
            x = b_int & 3
            b_int >>= 2
            if x != 3:
                coefficients[i] = self.mod_q(x - 1) # [-1, 0, 1]
                i += 1

        if max_hamming != -1:
            coefficients = self._sparsify(input_bytes, coefficients, max_hamming)

        return self._poly(coefficients, False)
    
    def gaussian(self, input_bytes: bytes, std: float, max_hamming: Optional[int] = -1) -> "PolynomialRing":
        """
        Samples a polynomial with coefficients from a gaussian distribution.

        :param bytes input_bytes: Random input bytes.
        :param float std: Standard deviation of the Gaussian distribution.
        :param int max_hamming: If set, enforces the specified Hamming weight.
        :return: Polynomial with Gaussian-distributed coefficients.
        """
        seed = int.from_bytes(input_bytes, "little")
        rng = np.random.default_rng(seed)
        coefficients = rng.normal(0, std, size=self.n)
        coefficients = self.mod_q(coefficients.round().astype(int))

        # convert back to list of int
        coefficients = coefficients.tolist()

        if max_hamming != -1:
            coefficients = self._sparsify(input_bytes, coefficients, max_hamming)

        return self._poly(coefficients, False)
    
    def encode(self, d: int) -> bytes:
        """
        Encode (Inverse of Algorithm 3)
        """
        t = 0
        for i in range(self.n - 1):
            t |= self.coeffs[self.n - i - 1]
            t <<= d
        t |= self.coeffs[0]
        return t.to_bytes((self.n * d) // 8, "little")
    
    def decode(self, input_bytes: bytes, d: int, is_ntt: Optional[bool]=False) -> "PolynomialRing":
        """
        Decode (Algorithm 3)

        decode: B^32l -> R_q

        :param bytes input_bytes: The input bytes to decode.
        :param int d: The bit-width of each coefficient.
        :param bool is_ntt: Flag indicating if the polynomial is in NTT form. Default is False.
        :return: The decoded polynomial ring.
        """
        # Ensure the value d is set correctly
        if self.n * d != len(input_bytes) * 8:
            raise ValueError(f"input bytes must be a multiple of (polynomial degree) / 8, {self.n*d = }, {len(input_bytes)*8 = }")

        # Set the modulus
        if d == 12:
            m = self.q
        else:
            m = 1 << d

        coeffs = [0] * self.n
        b_int = int.from_bytes(input_bytes, "little")
        mask = (1 << d) - 1
        for i in range(self.n):
            coeffs[i] = self.mod(b_int & mask, m)
            b_int >>= d

        return self._poly(coeffs, is_ntt=is_ntt)

    def compress(self, d: int) -> "PolynomialRing":
        """
        Compress the polynomial by compressing each coefficient

        NOTE: This is lossy compression
        """
        self.coeffs = [self._compress_ele(c, d) for c in self.coeffs]
        return self
    
    def _compress_ele(self, x: int, d: int) -> int:
        """
        Compute round((2^d / q) * x) % 2^d
        """
        t = 1 << d
        y = (t * x + self.q // 2) // self.q
        return y % t

    def decompress(self, d: int) -> "PolynomialRing":
        """
        Decompress the polynomial by decompressing each coefficient

        NOTE: This as compression is lossy, we have
        x' = decompress(compress(x)), which x' != x, but is
        close in magnitude.
        """
        self.coeffs = [self._decompress_ele(c, d) for c in self.coeffs]
        return self

    def _decompress_ele(self, x: int, d: int) -> int:
        """
        Compute round((q / 2^d) * x)
        """
        t = 1 << (d - 1)
        return (self.q * x + t) >> d

    def to_ntt(self) -> "PolynomialRing":
        """
        Convert a polynomial to number-theoretic transform (NTT) form.
        The input is in standard order, the output is in bit-reversed order.
        """
        if self.is_ntt:
            raise TypeError("Polynomial is already in NTT form")

        self._initialize_ntt()
        k, l = 1, self.n // 2
        coeffs = self.coeffs.copy()
        while l >= 2:
            start = 0
            while start < self.n:
                zeta = self.ntt_zetas[k]
                k += 1
                for j in range(start, start + l):
                    t = zeta * coeffs[j + l]
                    coeffs[j + l] = coeffs[j] - t
                    coeffs[j] = coeffs[j] + t
                start = l + j + 1
            l >>= 1

        for j in range(self.n):
            coeffs[j] = self.mod_q(coeffs[j])

        return self._poly(coeffs, is_ntt=True)
    
    def from_ntt(self) -> "PolynomialRing":
        """
        Convert a polynomial from number-theoretic transform (NTT) form in place
        The input is in bit-reversed order, the output is in standard order.
        """
        if not self.is_ntt:
            raise TypeError("Polynomial must be in NTT form to convert to standard form")
            
        l, l_upper = 2, self.n // 2
        k = l_upper - 1
        coeffs = self.coeffs.copy()
        while l <= l_upper:
            start = 0
            while start < self.n:
                zeta = self.ntt_zetas[k]
                k = k - 1
                for j in range(start, start + l):
                    t = coeffs[j]
                    coeffs[j] = t + coeffs[j + l]
                    coeffs[j + l] = coeffs[j + l] - t
                    coeffs[j + l] = zeta * coeffs[j + l]
                start = j + l + 1
            l = l << 1

        for j in range(self.n):
            coeffs[j] = self.mod_q(coeffs[j] * self.ntt_f)

        return self._poly(coeffs, is_ntt=False)

    def reduce_coefficients(self):
        """
        Reduce all coefficients modulo q inplace
        """
        self.coeffs = self._reduce_coefficients(self.coeffs)
        return self

    def _reduce_coefficients(self, coeffs: List[int]) -> List[int]:
        """
        Return a list of coefficients reduced modulo q
        """
        return [self.mod_q(c) for c in coeffs]
    
    def reduce_symmetric_mod(self):
        """
        Reduce all coefficients using symmetric modulo q
        """
        self.use_mods = True
        self.coeffs = self._reduce_coefficients(self.coeffs)
        return self
    
    def reduce_mod(self):
        """
        Reduce all coefficients using modulo q
        """
        self.use_mods = False
        self.coeffs = self._reduce_coefficients(self.coeffs)
        return self

    def mod_q(self, coeff: int) -> int:
        """
        Reduce a single coefficient using the correct modulo q
        """
        reduced = coeff % self.q
        if self.use_mods and reduced > self.q // 2:
            reduced -= self.q
        
        return reduced
    
    def mod(self, coeff: int, q: int) -> int:
        """
        Reduce a single coefficient using the correct modulo q
        """
        reduced = coeff % q
        if self.use_mods and reduced > q // 2:
            reduced -= q
        return reduced

    
    @staticmethod
    def _br(i: int, k: int) -> int:
        """
        bit reversal of an unsigned k-bit integer
        """
        bin_i = bin(i & (2**k - 1))[2:].zfill(k)
        return int(bin_i[::-1], 2)
    
    def __neg__(self) -> "PolynomialRing":
        """
        Returns -f, by negating all coefficients
        """
        neg_coeffs = [self.mod_q(-x) for x in self.coeffs]
        return self._poly(neg_coeffs, self.is_ntt)
        
    def __add__(self, other) -> "PolynomialRing":
        if isinstance(other, type(self)):
            new_coeffs = [self.mod_q(x + y) for x, y in zip(self.coeffs, other.coeffs)]
        elif isinstance(other, int):
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] = self.mod_q(new_coeffs[0] + other)
        else:
            raise NotImplementedError(
                "Polynomials can only be added to each other"
            )
        return self._poly(new_coeffs, self.is_ntt)

    def __radd__(self, other) -> "PolynomialRing":
        return self.__add__(other)

    def __iadd__(self, other) -> "PolynomialRing":
        self = self + other
        return self

    def __sub__(self, other: Union["PolynomialRing", int]) -> "PolynomialRing":

        if isinstance(other, type(self)):
            new_coeffs = [self.mod_q(x - y) for x, y in zip(self.coeffs, other.coeffs)]
        elif isinstance(other, int):
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] = self.mod_q(new_coeffs[0] - other)
        else:
            raise NotImplementedError(
                "Polynomials can only be subtracted from each other"
            )
        return self._poly(new_coeffs, self.is_ntt)

    def __rsub__(self, other: Union["PolynomialRing", int]) -> "PolynomialRing":
        return -self.__sub__(other)

    def __isub__(self, other: Union["PolynomialRing", int]) -> "PolynomialRing":
        self = self - other
        return self
    
    def __mul__(self, other: Union["PolynomialRing", int]) -> "PolynomialRing":
        """
        Multiplication between two polynomials
        
        NOTE: 
            schoolbook multiplication is used for non-NTT polynomials
            NTT multiplication is used for NTT polynomials
            In case of mixed multiplication, the non-NTT polynomial is converted to NTT
        """
        if isinstance(other, type(self)):
            if self.is_ntt and other.is_ntt:
                new_coeffs = self._ntt_multiplication(other)
            elif not self.is_ntt and not other.is_ntt:
                new_coeffs = self._schoolbook_multiplication(other)
            else:
                if not self.is_ntt:
                    tmp_self = self.to_ntt()
                    new_coeffs = tmp_self._ntt_multiplication(other)
                elif not other.is_ntt:
                    tmp_other = other.to_ntt()
                    new_coeffs = self._ntt_multiplication(tmp_other)
        elif isinstance(other, int):
            new_coeffs = [self.mod_q(c * other) for c in self.coeffs]
        else:
            raise NotImplementedError(
                f"Polynomials can only be multiplied by each other, or scaled by integers, {type(other) = }, {type(self) = }"
            )
        return self._poly(new_coeffs, self.is_ntt)

    def __rmul__(self, other: Union["PolynomialRing", int]) -> "PolynomialRing":
        return self.__mul__(other)

    def __imul__(self, other: Union["PolynomialRing", int]) -> "PolynomialRing":
        self = self * other
        return self
    
    def _ntt_base_multiplication(self, a0: int, a1: int, b0: int, b1: int, zeta: int) -> tuple[int, int]:
        """
        Base case for ntt multiplication
        """
        r0 = self.mod_q(a0 * b0 + zeta * a1 * b1)
        r1 = self.mod_q(a1 * b0 + a0 * b1)
        return r0, r1

    def _ntt_multiplication(self, other: "PolynomialRing") -> list:
        """
        Number Theoretic Transform multiplication.
        """
        f_coeffs = self.coeffs
        g_coeffs = other.coeffs
        new_coeffs = []
        zetas = self.ntt_zetas
        for i in range(self.n // 4):
            pair = 4 * i
            f0, f1, f2, f3 = f_coeffs[pair], f_coeffs[pair + 1], f_coeffs[pair + 2], f_coeffs[pair + 3]
            g0, g1, g2, g3 = g_coeffs[pair], g_coeffs[pair + 1], g_coeffs[pair + 2], g_coeffs[pair + 3]
            zeta = zetas[self.n // 4 + i]
            r0, r1 = self._ntt_base_multiplication(f0, f1, g0, g1, zeta)
            r2, r3 = self._ntt_base_multiplication(f2, f3, g2, g3, -zeta)
            new_coeffs += [r0, r1, r2, r3]
        return new_coeffs
    
    def _schoolbook_multiplication(self, other: "PolynomialRing") -> list:
        """
        Naive implementation of polynomial multiplication
        suitible for all R_q = F_1[X]/(X^n + 1)
        """
        n = self.n
        a = self.coeffs
        b = other.coeffs
        new_coeffs = [0] * n
        for i in range(n):
            for j in range(0, n - i):
                new_coeffs[i + j] += a[i] * b[j]
        for j in range(1, n):
            for i in range(n - j, n):
                new_coeffs[i + j - n] -= a[i] * b[j]
        return self._reduce_coefficients(new_coeffs)

    def __pow__(self, n: int) -> "PolynomialRing":
        if not isinstance(n, int):
            raise TypeError("Exponentiation of a polynomial must be done using an integer.")

        # Deal with negative scalar multiplication
        if n < 0:
            raise ValueError("Negative powers are not supported for elements of a Polynomial Ring")
        
        f = self
        g = self._poly([1] + [0] * (self.n - 1), self.is_ntt)
        while n > 0:
            if n % 2 == 1:
                g = g * f
            f = f * f
            n = n // 2
        return g

    @staticmethod
    def same_ring(a: "PolynomialRing", b: "PolynomialRing") -> bool:
        return a.q == b.q and a.n == b.n
    
    def __eq__(self, other: Union["PolynomialRing", int]) -> bool:
        if isinstance(other, type(self)):
            if not PolynomialRing.same_ring(self, other):
                return False
            
            if self.is_ntt == other.is_ntt:
                return self.coeffs == other.coeffs
            else:
                if not self.is_ntt:
                    tmp_self = self.to_ntt()
                    return tmp_self.coeffs == other.coeffs
                if not other.is_ntt:
                    tmp_other = other.to_ntt()
                    return self.coeffs == tmp_other.coeffs
        elif isinstance(other, int):
            return self.is_constant() and self.mod_q(other) == self.coeffs[0]
        return False

    def __ne__(self, other: Union["PolynomialRing", int]) -> bool:
        return not self.__eq__(other)
    
    def __getitem__(self, idx):
        return self.coeffs[idx]

    def _schoolbook_representation(self) -> str:
        if self.is_zero():
            return "0"

        info = []
        for i, c in enumerate(self.coeffs):
            if c != 0:
                if i == 0:
                    info.append(f"{c}")
                elif i == 1:
                    if c == 1:
                        info.append("x")
                    else:
                        info.append(f"{c}*x")
                else:
                    if c == 1:
                        info.append(f"x^{i}")
                    else:
                        info.append(f"{c}*x^{i}")
        return " + ".join(info)

    def _ntt_representation(self) -> str:
        """
        Return the NTT representation of the polynomial
        a + b*z, c + d*z ... 
        """
        if self.is_zero():
            return "0"

        info = []
        poly = []
        for i, c in enumerate(self.coeffs):    
            if i % 2 == 0 and c != 0:
                    poly.append(f"{c}")
            else:
                if c != 0:
                    poly.append(f"{c}*x")
                
                if len(poly) == 0:
                    poly.append("0")

                info.append(" + ".join(poly))
                poly = []

        return ", ".join(info)     
    
    def __repr__(self):
        if self.is_ntt:
            return self._ntt_representation()
        else:
            return self._schoolbook_representation()

    def __str__(self):
        return self.__repr__()
    
    def __len__(self):
        return self.n
    
    def size(self) -> int:
        """
        Compute the size of the polynomial as 
        size(f) = max(|f[0]|, |f[1]|, ..., |f[n-1]|)
        in symmetric modulo q
        """
        if self.use_mods:
            return max([abs(c) for c in self.coeffs])
        else:
            size = 0 
            for c in self.coeffs:
                if c > self.q // 2:
                    size = self.q - c if self.q - c > size else size
                else:
                    size = c if c > size else size
            return size
        
    def is_small(self, bound: int) -> bool:
        """
        Check if the polynomial is small
        """
        return self.size() <= bound
    
    def to_list(self) -> List[int]:
        """
        Return the coefficients of the polynomial
        """
        return self.coeffs
    
    def from_list(self, coeffs: List[int], is_ntt: Optional[bool] = False) -> "PolynomialRing":
        """
        Set the coefficients of the polynomial
        """
        if len(coeffs) != self.n:
            raise ValueError(f"Polynomial must have {self.n} coefficients, but got {len(coeffs)}")
        
        poly = self._poly(coeffs, is_ntt)
        poly.reduce_coefficients()
        return poly
    
    @staticmethod
    def mod_dist(x: int, y: int, q: int) -> int:
        """
        Compute the modular distance between two integers
        """
        diff = (x - y) % q
        return min(diff, q - diff)

    @staticmethod
    def distance(a: "PolynomialRing", b: "PolynomialRing", norm: str = "L1") -> float:
        """
        Compute the distance between two polynomials using the specified norm.

        :param a: First polynomial.
        :param b: Second polynomial.
        :param norm: Distance metric ('L1', 'L2', 'Linf').
        :return: Distance between polynomials.
        """
        if not isinstance(a, PolynomialRing) or not isinstance(b, PolynomialRing):
            raise TypeError(f"Polynomials must be of type PolynomialRing to compute distance, but got {type(a) = }, {type(b) = }")
        if not PolynomialRing.same_ring(a, b):
            raise ValueError(f"Polynomials must be in the same ring to compute distance, but got {a.q = }, {b.q = }, {a.n = }, {b.n = }")

        # Compute distance based on the selected norm
        q = a.q
        if norm == "L1":
            return sum(PolynomialRing.mod_dist(x, y, q) for x, y in zip(a.coeffs, b.coeffs))
        elif norm == "L2":
            return sqrt(sum(PolynomialRing.mod_dist(x, y, q) ** 2 for x, y in zip(a.coeffs, b.coeffs)))
        elif norm == "Linf":
            return max(PolynomialRing.mod_dist(x, y, q) for x, y in zip(a.coeffs, b.coeffs))
        else:
            raise ValueError(f"Invalid norm: {norm}")

