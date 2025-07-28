from kyber.mlwe import MLWE

import numpy as np

from scipy.linalg import circulant


def neg_circ(a):
    """
    Generates a negative circulant matrix from the input vector a.
    """
    n = len(a)
    A = circulant(a)
    tri = np.triu_indices(n, 1)
    A[tri] *= -1
    return A

def get_lwe(params):
    # Generate the MLWE matrices
    mlwe = MLWE(params)
    random_bytes = mlwe.get_random_bytes()
    A, s, e, B = mlwe.generate(random_bytes)
    
    # Transform A:
    A_lwe = transform_matrix_lwe(A)

    # Transform s:
    s_lwe = transform_vector_lwe(s)

    # Transform B:
    b_lwe = transform_vector_lwe(B)

    # Transform e:
    e_lwe = transform_vector_lwe(e)

    return A_lwe, s_lwe, b_lwe, e_lwe

def transform_matrix_lwe(matrix : list):
    k = len(matrix)
    l = len(matrix[0])
    n = len(matrix[0][0])

    # Transform A:
    matrix_lwe = np.zeros((k*n, l*n))
    for i in range(k): # rows
        for j in range(l): # columns
            a = matrix[i][j]
            neg_circ_a = neg_circ(a)
            matrix_lwe[i*n:(i+1)*n, j*n:(j+1)*n] = neg_circ_a
        
    return matrix_lwe.squeeze()

def transform_vector_lwe(vector : list):
    return np.array(vector).flatten()



def reverse_neg_circ(matrix):
    """
    Given a negative circulant matrix, return the original 1D vector a.
    This assumes the input matrix is a negative circulant matrix with upper triangular negated.
    """
    return matrix[:, 0]  # First column of the circulant matrix is the original vector

def reverse_transform_matrix_lwe(matrix_lwe, k, l, n):
    """
    Reverse the transform_matrix_lwe operation.
    Converts a (k*n, l*n) matrix back into a list of shape (k, l, n).
    """
    A = []
    for i in range(k):
        row = []
        for j in range(l):
            block = matrix_lwe[i*n:(i+1)*n, j*n:(j+1)*n]
            a = reverse_neg_circ(block)
            row.append(a.tolist())
        A.append(row)
    return A