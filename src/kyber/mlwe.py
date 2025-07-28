from .module import Module
import os
from hashlib import sha3_256, sha3_512, shake_128, shake_256
from typing import Optional
import math

class MLWE:
    def __init__(self, params: dict):
        """
        Initialise the ML-KEM with specified lattice parameters.

        :param dict params: Dictionary containing the lattice parameters:
            
            - n (int): Polynomial size, typically a power of 2.
            
            - q (int): Modulus, a prime number defining the ring.
            
            - k (int): Security parameter, determines key size (matrix k x k and vectors of size k). Default 1.
            
            - eta (int): Error distribution parameter, controls noise level. Default 2.
            
            - gaussian_std (int): Standard deviation for Gaussian error distribution. Default 2.
            
            - secret_type (str): Type of secret distribution, one of "cbd", "binary", "ternary", "gaussian". Default "cbd".

            - hw (int): Hamming weight for vector generation. Default -1 (no limit).
            
            - error_type (str): Type of error distribution, one of "cbd", "binary", "ternary", "gaussian". Default "cbd".
            
            - seed (int): Seed for deterministic randomness. Default os.urandom.
        """
        # ml-kem params
        self.n = params["n"]
        self.q = params["q"]
        self.k = params.get("k", 1)

        self.eta = params.get("eta", 2)
        self.gaussian_std = params.get("gaussian_std", 2)

        self.secret_type = params.get("secret_type", "cbd")
        self.error_type = params.get("error_type", "cbd")
        self.hw = params.get("hw", -1)

        self.M = Module(self.n, self.q)
        self.R = self.M.ring

        # Use system randomness by default, for deterministic randomness
        # use the method `set_drbg_seed()`
        if "seed" in params and params["seed"] is not None:
            self.set_drbg_seed(params["seed"].to_bytes(48, 'big'))
        else:
            self.random_bytes = os.urandom

    def set_drbg_seed(self, seed: bytes):
        """
        Change entropy source to a DRBG and seed it with provided value.

        Setting the seed switches the entropy source from :func:`os.urandom()`
        to an AES256 CTR DRBG.

        Used for both deterministic versions of ML-KEM as well as testing
        alignment with the KAT vectors

        NOTE:
          currently requires pycryptodome for AES impl.

        :param bytes seed: random bytes to seed the DRBG with
        """
        try:
            from .aes256_ctr_drbg import AES256_CTR_DRBG

            self._drbg = AES256_CTR_DRBG(seed)
            self.random_bytes = self._drbg.random_bytes
        except ImportError as e:  # pragma: no cover
            print(f"Error importing AES from pycryptodome: {e = }")
            raise Warning(
                "Cannot set DRBG seed due to missing dependencies, try installing requirements: pip -r install requirements"
            )
    
    def reset_seed(self, seed: Optional[int] = None):
        if seed is not None:
            self.set_drbg_seed(seed.to_bytes(48, 'big'))
        else:
            self.random_bytes = os.urandom

    def get_random_bytes(self) -> bytes:
        """
        Get random bytes from the entropy source.

        :param int size: Number of random bytes to return
        :return: Random bytes
        """
        return self.random_bytes(32)

    @staticmethod
    def _xof(bytes32: bytes, i: bytes, j: bytes) -> bytes:
        """
        eXtendable-Output Function (XOF) described in 4.9 of FIPS 203 (page 19)

        :param bytes bytes32: 32 byte array
        :param bytes i: single byte
        :param bytes j: single byte
        :return: 840 bytes of output

        NOTE:
          We use hashlib's ``shake_128`` implementation, which does not support
          an easy XOF interface, so we take the "easy" option and request a
          fixed number of 840 bytes (5 invocations of Keccak), rather than
          creating a byte stream.

          If your code crashes because of too few bytes, you can get dinner at:
          Casa de Chá da Boa Nova
          https://cryptojedi.org/papers/terminate-20230516.pdf
        """
        input_bytes = bytes32 + i + j
        if len(input_bytes) != 34:
            raise ValueError(
                "Input bytes should be one 32 byte array and 2 single bytes."
            )
        return shake_128(input_bytes).digest(840)

    def _prf(self, output_size: int, s: bytes, b: bytes) -> bytes:
        """
        Pseudorandom function described in 4.3 of FIPS 203 (page 18)

        :param int output_size: Number of bytes to output
        :param bytes s: 32 byte array
        :param bytes b: single byte
        :return: output_size bytes of output
        """
        input_bytes = s + b
        if len(input_bytes) != 33:
            raise ValueError(
                "Input bytes should be one 32 byte array and one single byte."
            )
        return shake_256(input_bytes).digest(output_size)

    @staticmethod
    def _H(s: bytes) -> bytes:
        """
        Hash function described in 4.4 of FIPS 203 (page 18)

        :param bytes s: 32 byte array
        :return: 32 bytes of output
        """
        return sha3_256(s).digest()

    @staticmethod
    def _J(s: bytes) -> bytes:
        """
        Hash function described in 4.4 of FIPS 203 (page 18)

        :param bytes s: 32 byte array
        :return: 32 bytes of output
        """
        return shake_256(s).digest(32)

    @staticmethod
    def _G(s: bytes) -> tuple[bytes, bytes]:
        """
        Hash function described in 4.5 of FIPS 203 (page 18)

        :param bytes s: 32 byte array
        :return: 2x 32 bytes of output
        """
        h = sha3_512(s).digest()
        return h[:32], h[32:]

    def _get_prf_output_length_and_generation_fn(self, type: str) -> tuple[int, callable]:
        if type == "cbd":
            output_dim_prf = (self.n * self.eta) // 4
            generation_fn = lambda input_bytes, max_hamming: self.R.cbd(input_bytes, self.eta, max_hamming=max_hamming)
        elif type == "binary":
            output_dim_prf = self.n // 8
            generation_fn = self.R.binary
        elif type == "ternary":
            mean_pairs = self.n * 4 / 3
            std_pairs = math.sqrt(4 * self.n / 9)
            z_score = 10  # Very high probability of success
            bit_pairs_needed = int(mean_pairs + z_score * std_pairs)
            output_dim_prf = bit_pairs_needed // 4
            generation_fn = self.R.ternary
        elif type == "gaussian":
            output_dim_prf = self.n // 4
            generation_fn = lambda input_bytes, max_hamming: self.R.gaussian(input_bytes, self.gaussian_std, max_hamming=max_hamming)
        elif type == "uniform":
            output_dim_prf = self.n // 4
            generation_fn = lambda input_bytes, max_hamming: self.R.uniform(input_bytes)
        elif type == "ntt":
            output_dim_prf = 3 * self.n / 2
            generation_fn = lambda input_bytes, max_hamming: self.R.ntt_sample(input_bytes)
        else:
            raise ValueError(f"Unknown type {type}")
        
        return output_dim_prf, generation_fn
    
    def _generate_matrix(self, rho: bytes, transpose: Optional[bool] = False, max_hamming: Optional[int] = -1, type: Optional[str] = "ntt") -> Module:
        """
        Helper function which generates a element of size
        k x k from a seed `rho`.

        When `transpose` is set to True, the matrix A is
        built as the transpose.

        :param bytes rho: 32 byte array
        :param bool transpose: flag to transpose the matrix
        :return: Matrix of the public key (generally not in NTT form)
        """
        # Output of XOF should be enough for every distribution
        _, generation_fn = self._get_prf_output_length_and_generation_fn(type)

        A_data = [[0 for _ in range(self.k)] for _ in range(self.k)]
        for i in range(self.k):
            for j in range(self.k):
                xof_bytes = self._xof(rho, bytes([j]), bytes([i]))
                A_data[i][j] = generation_fn(xof_bytes, max_hamming)
        A_hat = self.M(A_data, transpose=transpose)
        return A_hat

    def generate_A_hat(self, d: bytes) -> Module:
        """
        Helper function which generates a element of size
        k x k from a seed `d`.

        :param bytes d: 32 byte array
        :return: Matrix of the public key (in NTT form)
        """
        rho, _ = self._G(d + bytes([self.k]))
        return self._generate_matrix(rho, type="ntt")
    
    def generate_A(self, d: bytes) -> Module:
        """
        Helper function which generates a element of size
        k x n from a seed `d`.

        :param bytes d: 32 byte array
        :return: Matrix of the public key (not in NTT form)
        """
        rho, _ = self._G(d + bytes([self.k]))
        return self._generate_matrix(rho, type="uniform")
    
    def _generate_vector(self, sigma: bytes, N: int, max_hamming: Optional[int] = -1, type: Optional[str]="cbd") -> tuple[Module, int]:
        """
        Helper function which generates a element in the
        module from the specified distribution.

        :param bytes sigma: 32 byte array
        :param int N: Counter
        :return: Vector (generally not in NTT form, unless called with type="ntt") and updated counter
        """
        output_dim_prf, generation_fn = self._get_prf_output_length_and_generation_fn(type)

        elements = [0 for _ in range(self.k)]
        for i in range(self.k):
            prf_output = self._prf(output_dim_prf, sigma, bytes([N]))
            elements[i] = generation_fn(prf_output, max_hamming)
            N += 1
        v = self.M.vector(elements)
        return v, N

    def generate_secret(self, d: bytes) -> Module:
        """
        Helper function which generates a element in the
        module.

        :param bytes d: 32 byte array
        :return: Vector of the secret (not in NTT form)
        """
        _, sigma = self._G(d + bytes([self.k]))
        return self._generate_vector(sigma, 0, max_hamming=self.hw, type=self.secret_type)[0]

    def generate_error(self, d: bytes) -> Module:
        """
        Helper function which generates a element in the
        module.

        :param bytes d: 32 byte array
        :return: Vector of the error (not in NTT form)
        """
        _, sigma = self._G(d + bytes([self.k]))
        return self._generate_vector(sigma, self.k, type=self.error_type)[0]

    def generate_polynomial(self, sigma: bytes, N: int, max_hamming: Optional[int] = -1, type: Optional[str]="cbd") -> tuple[Module, int]:
        """
        Helper function which generates a element in the
        polynomial ring from the specified distribution.

        :param bytes sigma: 32 byte array
        :param int eta: Error distribution parameter
        :param int N: Counter
        :return: Polynomial (generally not in NTT form, unless called with type="ntt") and updated counter
        """
        output_length, generation_fn = self._get_prf_output_length_and_generation_fn(type)

        prf_output = self._prf(output_length, sigma, bytes([N]))
        p = generation_fn(prf_output, max_hamming)
        return p, N + 1

    def generate_hat(self, d: bytes) -> tuple[Module, Module, Module, Module]:
        """
        Generate the matrices and vectors used in the cryptographic protocol.

        This function generates the matrix A_hat, the secret vector s_hat, 
        the error vector e_hat, and the public value B_hat using the provided 
        seed `d`.

        :param bytes d: 32 byte array used as a seed for generating matrices and vectors.
        :return: Tuple containing the matrix A_hat, the secret vector s_hat, 
             the error vector e_hat, and the public value B_hat all in NTT form.
        :rtype: tuple(Module, Module, Module, Module)
        """
        # Expand 32 + 1 bytes to two 32-byte seeds. Note that the
        # inclusion of the lattice parameter here is for domain
        # separation between different parameter sets
        rho, sigma = self._G(d + bytes([self.k]))

        # Generate A_hat from seed rho
        A_hat = self._generate_matrix(rho)

        # Set counter for PRF
        N = 0

        # Generate the error vector s ∈ R^k
        s, N = self._generate_vector(sigma, N, max_hamming=self.hw, type=self.secret_type)

        # Generate the error vector e ∈ R^k
        e, N = self._generate_vector(sigma, N, type=self.error_type)

        # Compute public value (in NTT form)
        s_hat = s.to_ntt()
        e_hat = e.to_ntt()
        B_hat = A_hat @ s_hat + e_hat

        return A_hat, s_hat, e_hat, B_hat
    
    def generate(self, d: bytes) -> tuple[Module, Module, Module, Module]:
        """
        Generate the matrices and vectors used in the cryptographic protocol.

        This function generates the matrix A, the secret vector s, 
        the error vector e, and the public value B using the provided 
        seed `d`.

        :param bytes d: 32 byte array used as a seed for generating matrices and vectors.
        :return: Tuple containing the matrix A, the secret vector s, 
             the error vector e, and the public value B (not in NTT form).
        :rtype: tuple(Module, Module, Module, Module)
        """
        # Expand 32 + 1 bytes to two 32-byte seeds. Note that the
        # inclusion of the lattice parameter here is for domain
        # separation between different parameter sets
        rho, sigma = self._G(d + bytes([self.k]))

        # Generate A from seed rho
        A = self._generate_matrix(rho, type="uniform")

        # Set counter for PRF
        N = 0

        # Generate the secret vector s ∈ R^k
        s, N = self._generate_vector(sigma, N, max_hamming=self.hw, type=self.secret_type)

        # Generate the error vector e ∈ R^k
        e, N = self._generate_vector(sigma, N, type=self.error_type)

        # Compute public value
        B = A @ s + e

        return A, s, e, B
    
    def generate_A_B_hat(self, s_hat: Module, d: bytes):
        """
        Generates the matrices A_hat and B_hat used in the cryptographic protocol
        starting from the secret vector in NTT form.
        
        :param Module s_hat: The secret module vector.
        :param bytes d: A byte string used as a seed for generating matrices.
        :return: A tuple containing the matrix A_hat and the matrix B_hat (in NTT form)
        :rtype: tuple(Module, Module)
        """
        # Expand 32 + 1 bytes to two 32-byte seeds. Note that the
        # inclusion of the lattice parameter here is for domain
        # separation between different parameter sets
        rho, sigma = self._G(d + bytes([self.k]))

        # Generate A_hat from seed rho
        A_hat = self._generate_matrix(rho)

        # Set counter for PRF
        N = self.k

        # Generate the error vector e ∈ R^k
        e, N = self._generate_vector(sigma, N, type=self.error_type)

        # Compute public value (in NTT form)
        e_hat = e.to_ntt()
        B_hat = A_hat @ s_hat + e_hat

        return A_hat, B_hat
    
    def generate_A_B(self, s: Module, d: bytes):
        """
        Generates the matrices A and B used in the cryptographic protocol
        starting from the secret vector.
        
        :param Module s: The secret module vector (not in NTT form)
        :param bytes d: A byte string used as a seed for generating matrices.
        :return: A tuple containing the matrix A and the matrix B (not in NTT form)
        :rtype: tuple(Module, Module)
        """
        # Expand 32 + 1 bytes to two 32-byte seeds. Note that the
        # inclusion of the lattice parameter here is for domain
        # separation between different parameter sets
        rho, sigma = self._G(d + bytes([self.k]))

        # Generate A_hat from seed rho
        A = self._generate_matrix(rho, type="uniform")

        # Generate the error vector e ∈ R^k
        e, _ = self._generate_vector(sigma, self.k, type=self.error_type)

        # Compute public value
        B = A @ s + e

        return A, B
    
    def generate_B(self, A: Module, s: Module, d: bytes):
        """
        Generates the matrix B used in the cryptographic protocol
        starting from the public matrix A and the secret vector s.
        
        :param Module A: The public matrix (not in NTT form)
        :param Module s: The secret module vector (not in NTT form)
        :param bytes d: A byte string used as a seed for generating matrices.
        :return: The matrix B (not in NTT form)
        :rtype: Module
        """
        _, sigma = self._G(d + bytes([self.k]))

        # Generate the error vector e ∈ R^k
        e, _ = self._generate_vector(sigma, self.k, type=self.error_type)

        # Compute public value
        B = A @ s + e

        return B