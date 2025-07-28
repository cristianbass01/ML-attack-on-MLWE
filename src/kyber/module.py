from .polynomial_ring import PolynomialRing
import copy
from typing import List, Tuple, Optional, Union

class Module:
    def __init__(self, n: int, q: int) -> None:
        """
        Initialise a module over the ring ``ring``.
        """
        self.ring = PolynomialRing(n, q)
        self.matrix = None
        self._transpose = False

    def copy(self) -> "Module":
        """
        Return a copy of the module
        """
        return self._module(copy.deepcopy(self.matrix), self._transpose)
    
    def _module(self, matrix: List[List[PolynomialRing]], transpose: Optional[bool] = False) -> "Module":
        """
        Return a module with the given matrix
        """
        module = Module(self.ring.n, self.ring.q)

        # Initialize ring parameters
        module.ring.root_of_unity = self.ring.root_of_unity
        module.ring.ntt_zetas = self.ring.ntt_zetas
        module.ring.ntt_f = self.ring.ntt_f

        # Set the matrix and transpose
        module.matrix = matrix
        module._transpose = transpose
        return module
    
    def zeros(self, n: int, m: int) -> "Module":
        """
        Return a module of zeros

        :param int n: the number of rows in the matrix
        :param int m: the number of columns in the matrix
        :return: a module of zeros with dimension `n times m
        """
        matrix = [[self.ring.zero() for _ in range(m)] for _ in range(n)]
        return self._module(matrix)

    def random_element(self, m: int, n: int) -> "Module":
        """
        Generate a random element of the module of dimension m x n

        :param int m: the number of rows in the matrix
        :param int m: the number of columns in tge matrix
        :return: an element of the module with dimension `m times n`
        """
        matrix = [[self.ring.uniform() for _ in range(n)] for _ in range(m)]
        return self._module(matrix)

    def __call__(self, elements: Union[List[PolynomialRing], List[List[PolynomialRing]]], transpose: Optional[bool] = False) -> "Module":
        """
        Construct a module given a list of elements of the module's ring
        
        :param elements: a list of elements of the ring
        :param transpose: whether to transpose the matrix
        :return: a module of the ring
        """
        if not isinstance(elements, list):
            raise TypeError("elements of a module are matrices, built from elements of the base ring")

        if isinstance(elements[0], list):
            for element_list in elements:
                if not all(isinstance(aij, type(self.ring)) for aij in element_list):
                    raise TypeError(f"All elements of the matrix must be elements of the ring: {self.ring}")
            elements = elements
        elif isinstance(elements[0], type(self.ring)):
            if not all(isinstance(aij, type(self.ring)) for aij in elements):
                raise TypeError(f"All elements of the matrix must be elements of the ring: {self.ring}")
            elements = [elements]
        else:
            raise TypeError("elements of a module are matrices, built from elements of the base ring")

        if not len(set(map(len, elements))) == 1:
            raise ValueError("Matrix is not rectangular")
        
        return self._module(elements, transpose)

    def vector(self, elements: List[PolynomialRing]) -> "Module":
        """
        Construct a vector given a list of elements of the module's ring

        :param list: a list of elements of the ring
        :return: a vector of the module
        """
        return self([elements], transpose=True)
    
    def decode_vector(self, input_bytes: int, k: int, d: int, is_ntt: Optional[bool] = False) -> "Module":
        """
        Decode bytes into a a vector of polynomial elements.

        Each element is assumed to be encoded as a polynomial with ``d``-bit
        coefficients (hence a polynomial is encoded into ``ring.n * d`` bits).

        A vector of length ``k`` then has ``ring.n * d * k`` bits.

        :param input_bytes: the bytes to decode
        :param k: the number of elements in the vector
        :param d: the number of bits used for each coefficient
        :param is_ntt: whether the polynomial is in NTT form
        :return: a vector of polynomial elements
        """
        # Ensure the input bytes are the correct length to create k elements with
        # d bits used for each coefficient
        if self.ring.n * d * k != len(input_bytes) * 8:
            raise ValueError("Byte length is the wrong length for given k, d values")

        # Bytes needed to decode a polynomial
        n = self.ring.n * d // 8

        # Encode each chunk of bytes as a polynomial and create the vector
        elements = [
            self.ring.decode(input_bytes[i : i + n], d, is_ntt=is_ntt)
            for i in range(0, len(input_bytes), n)
        ]

        return self.vector(elements)
    
    @property
    def shape(self) -> tuple:
        """
        Return the dimensions of the matrix with m rows
        and n columns

        :return: the dimension of the matrix ``(m, n)``
        :rtype: tuple(int, int)
        """
        if not self._transpose:
            return len(self.matrix), len(self.matrix[0])
        else:
            return len(self.matrix[0]), len(self.matrix)
        
    def dim(self) -> tuple:
        """
        Return the dimensions of the matrix with m rows
        and n columns

        :return: the dimension of the matrix ``(m, n)``
        :rtype: tuple(int, int)
        """
        return self.shape
    
    def __len__(self) -> int:
        """
        Return the number of rows in the matrix
        """
        return self.shape[0] if not self._transpose else self.shape[1]
    
    def size(self) -> int:
        """
        Return the size of the Module that is computed
        from the sizes of the PolynomialRing
        size(Module) = max(size(PolynomialRing), size(PolynomialRing) ... )
        """
        return max([x.size() for row in self.matrix for x in row])
    
    def is_small(self, bound: int) -> bool:
        """
        Check if the module is small
        """
        return self.size() <= bound

    
    def encode(self, d: int) -> bytes:
        """
        Encode every element of a matrix into bytes and concatenate
        
        :param int d: the number of bits to encode each coefficient
        :return: the bytes representing the matrix
        """
        output = b""
        for row in self.matrix:
            for ele in row:
                output += ele.encode(d)
        return output
    
    def compress(self, d: int) -> "Module":
        """
        Compress every element of the matrix to have at most ``d`` bits

        This is a lossy compression

        :param int d: the number of bits to compress each coefficient to
        :return: the compressed matrix
        """
        for row in self.matrix:
            for ele in row:
                ele.compress(d)
        return self

    def decompress(self, d: int) -> "Module":
        """
        Perform (lossy) decompression of the polynomial assuming it has been
        compressed to have at most ``d`` bits.

        :param int d: the number of bits to decompress each coefficient to
        :return: the decompressed polynomial
        """
        for row in self.matrix:
            for ele in row:
                ele.decompress(d)
        return self

    def to_ntt(self) -> "Module":
        """
        Convert every element of the matrix into NTT form

        :return: the matrix with all elements in NTT form
        """
        matrix = [[x.to_ntt() for x in row] for row in self.matrix]
        return self._module( matrix, self._transpose)
    
    def from_ntt(self) -> "Module":
        """
        Convert every element of the matrix from NTT form

        :return: the matrix with all elements in NTT form
        """
        matrix = [[x.from_ntt() for x in row] for row in self.matrix]
        return self._module( matrix, self._transpose)
    
    def _check_dimensions(self) -> bool:
        """
        Ensure that the matrix is rectangular
        """
        return len(set(map(len, self.matrix))) == 1

    def transpose(self) -> "Module":
        """
        Return a matrix with the rows and columns of swapped
        """
        return self._module( self.matrix, not self._transpose)

    def transpose_self(self) -> None:
        """
        Swap the rows and columns of the matrix in place
        """
        self._transpose = not self._transpose
        return

    T = property(transpose) 

    def reduce_coefficients(self) -> "Module":
        """
        Reduce every element in the polynomial
        using the modulus of the PolynomialRing
        """
        for row in self.matrix:
            for ele in row:
                ele.reduce_coefficients()
        return self
    
    def reduce_symmetric_mod(self) -> "Module":
        """
        Reduce every element in the polynomial
        using the symmetric modulus of the PolynomialRing
        """
        for row in self.matrix:
            for ele in row:
                ele.reduce_symmetric_mod()
        return self
    
    def reduce_mod(self) -> "Module":
        """
        Reduce every element in the polynomial
        using the modulus of the PolynomialRing
        """
        for row in self.matrix:
            for ele in row:
                ele.reduce_mod()
        return self

    def __getitem__(self, idx: Tuple[int, int]) -> PolynomialRing:
        """
        matrix[i, j] returns the element on row i, column j

        :param idx: the index of the element to access
        :type idx: tuple(int, int)
        :return: the element at the index
        :rtype: PolynomialRing
        """
        assert (
            isinstance(idx, tuple) and len(idx) == 2
        ), "Can't access individual rows"
        
        if not self._transpose:
            return self.matrix[idx[0]][idx[1]]
        else:
            return self.matrix[idx[1]][idx[0]]

    def __eq__(self, other: "Module") -> bool:
        if not isinstance(other, type(self)):
            return False
        
        if self.shape != other.shape:
            return False
        m, n = self.shape
        return all([self[i, j] == other[i, j] for i in range(m) for j in range(n)])

    def __neg__(self) -> "Module":
        """
        Returns -self, by negating all elements
        """
        m, n = self.shape
        elements = [[-self[i, j] for j in range(n)] for i in range(m)]
        return self._module(elements, self._transpose)
        

    def __add__(self, other: "Module") -> "Module":
        if not isinstance(other, type(self)):
            raise TypeError("Can only add matrices to other matrices")
        if self.ring != other.ring:
            raise TypeError("Matrices must have the same base ring")
        if self.shape != other.shape:
            raise ValueError("Matrices are not of the same dimensions")

        m, n = self.shape
        elements = [[self[i, j] + other[i, j] for j in range(n)] for i in range(m)]
        return self._module(elements)

    def __iadd__(self, other: "Module") -> "Module":
        self = self + other
        return self

    def __sub__(self, other: "Module") -> "Module":
        if not isinstance(other, type(self)):
            raise TypeError("Can only add matrices to other matrices")
        if self.ring != other.ring:
            raise TypeError("Matrices must have the same base ring")
        if self.shape != other.shape:
            raise ValueError("Matrices are not of the same dimensions")

        m, n = self.shape
        elements = [[self[i, j] - other[i, j] for j in range(n)] for i in range(m)]
        return self._module(elements)

    def __isub__(self, other: "Module") -> "Module":
        self = self - other
        return self

    def __matmul__(self, other: "Module") -> "Module":
        """
        Denoted A @ B
        """
        if not isinstance(other, type(self)):
            raise TypeError("Can only multiply matrcies with other matrices")
        if self.ring != other.ring:
            raise TypeError("Matrices must have the same base ring")

        m, n = self.shape
        n_, l = other.shape
        if not n == n_:
            raise ValueError("Matrices are of incompatible dimensions")
        elements = [[sum(self[i, k] * other[k, j] for k in range(n)) for j in range(l)] for i in range(m)]

        return self._module(elements)

    def dot(self, other: "Module") -> PolynomialRing:
        """
        Compute the inner product of two vectors
        """
        if not isinstance(other, type(self)):
            raise TypeError("Can only perform dot product with other matrices")
        res = self.T @ other
        assert res.dim() == (1, 1)
        return res[0, 0]

    def __repr__(self) -> str:
        if self.matrix is None:
            return f"Module over the commutative ring: {self.ring}"

        m, n = self.shape

        if m == 1:
            return str(self.matrix[0])

        max_col_width = [
            max(len(str(self[i, j])) for i in range(m)) for j in range(n)
        ]
        info = "]\n[".join(
            [
                ", ".join(
                    [
                        f"{str(self[i, j]):>{max_col_width[j]}}"
                        for j in range(n)
                    ]
                )
                for i in range(m)
            ]
        )
        return f"[{info}]"
    
    def __str__(self) -> str:
        return self.__repr__()

    def to_list(self) -> List[List[List[int]]]:
        """
        Convert the matrix to a list
        """
        m, n = self.shape
        elements = [[self[i, j].to_list() for j in range(n)] for i in range(m)]
        return elements
    
    def from_list(self, elements: Union[List[List[List[int]]], List[List[int]]], transpose: Optional[bool] = False, is_ntt: Optional[bool] = False) -> "Module":
        """
        Construct a module from a list of elements
        """
        if not isinstance(elements, list) or not isinstance(elements[0], list):
            raise TypeError("elements of a module are matrices, built from elements of the base ring")
        
        if not isinstance(elements[0][0], list):
            elements = [elements]

        m, n = len(elements), len(elements[0])
        matrix = [[self.ring.from_list(elements[i][j], is_ntt=is_ntt) for j in range(n)] for i in range(m)]
        return self._module(matrix, transpose)

    @staticmethod
    def distance(a: "Module", b: "Module", norm: str = "L1") -> float:
        """
        Compute the distance between two modules by computing the distance of corresponding polynomials.

        :param other: The other module.
        :param norm: Distance metric ('L1', 'L2', 'Linf').
        :return: The distance between the two modules.
        """
        if not isinstance(a, type(b)):
            raise TypeError("Can only compute distance between modules of the same type")
        if a.ring.n != b.ring.n:
            raise ValueError(f"Modules must have the same ring size, but got {a.ring.n} and {b.ring.n}")
        if a.ring.q != b.ring.q:
            raise ValueError(f"Modules must have the same ring modulus, but got {a.ring.q} and {b.ring.q}")
        if a.shape != b.shape:
            raise ValueError(f"Modules must have the same shape, but got {a.shape} and {b.shape}")

        total_distance = 0
        for p1, p2 in zip(a.matrix, b.matrix):
            for f1, f2 in zip(p1, p2):
                total_distance += PolynomialRing.distance(f1, f2, norm=norm)
        return total_distance