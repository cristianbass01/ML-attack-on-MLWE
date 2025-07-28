import torch
import math

###############################################
#             COMPLEX EMBEDDING               #
###############################################

def fourier_int_to_complex(x, q):
    """
    Maps integer tensor x (mod q) to a complex number using the Fourier transform.
    Uses the mapping: x -> exp(2*pi*i * x / q)
    """
    return torch.exp(2j * math.pi * x / q)

def fourier_complex_to_int(x, q):
    """
    Maps complex tensor x to an integer tensor (mod q) using the Fourier transform.
    Uses the mapping: x -> log(x) * q / (2*pi)
    """
    return torch.imag(torch.log(x)) * q / (2 * math.pi)


#################################################
#             FOURIER NTT SETUP                 #
#################################################

def br(i: int, k: int) -> int:
    """
    bit reversal of an unsigned k-bit integer
    """
    bin_i = bin(i & (2**k - 1))[2:].zfill(k)
    return int(bin_i[::-1], 2)

def fourier_ntt(tensor: torch.Tensor, ntt_zetas: list, dim: int = -1) -> torch.Tensor:
    coeffs = tensor.clone()
    n = coeffs.shape[dim]
    
    k, l = 1, n // 2
    while l >= 2:
        start = 0
        while start < n:
            zeta = ntt_zetas[k]
            k += 1
            for j in range(start, start + l):
                t = coeffs.index_select(
                    dim, 
                    torch.tensor([j + l])
                )  # Zeta exponentiation
                t = t ** zeta
                
                coeffs.index_copy_(
                    dim,
                    torch.tensor([j + l]),
                    coeffs.index_select(dim, torch.tensor([j])) / (t + 1e-9)
                )  # Element-wise division with epsilon

                coeffs.index_copy_(
                    dim,
                    torch.tensor([j]), 
                    coeffs.index_select(dim, torch.tensor([j])) * t
                )  # Element-wise multiplication

            start = l + j + 1
        l >>= 1
        
    return coeffs

def fourier_intt(tensor: torch.Tensor, ntt_zetas: list, ntt_f: int, dim: int = -1) -> torch.Tensor:
    coeffs = tensor.clone()
    n = coeffs.shape[dim]

    l, l_upper = 2, n // 2
    k = l_upper - 1
    while l <= l_upper:
        start = 0
        while start < n:
            zeta = ntt_zetas[k]
            k -= 1
            for j in range(start, start + l):
                t = coeffs.index_select(dim, torch.tensor([j]))  # Ensure correct shape
                
                coeffs.index_copy_(
                    dim,
                    torch.tensor([j]),
                    coeffs.index_select(dim, torch.tensor([j + l])) * t
                )  # Element-wise multiplication

                coeffs.index_copy_(
                    dim,
                    torch.tensor([j + l]),
                    coeffs.index_select(dim, torch.tensor([j + l])) / (t + 1e-9)
                )  # Element-wise division with epsilon

                coeffs.index_copy_(
                    dim,
                    torch.tensor([j + l]),
                    coeffs.index_select(dim, torch.tensor([j + l])) ** zeta
                )  # Exponentiation

            start = j + l + 1
        l <<= 1

    return coeffs ** ntt_f


###################################################
#           FOURIER NTT MULTIPLICATION            #
###################################################

def fourier_ntt_base_mul(a0: int, a1: int, b0: int, b1: int, zeta: int) -> tuple[int, int]:
    """
    Base case for ntt multiplication
    """
    r0 = (a0 ** b0) * ((a1 ** b1) ** zeta)
    r1 = (a1 ** b0) * (a0 ** b1)
    return r0, r1

def fourier_ntt_mul(a: torch.Tensor, b: torch.Tensor, ntt_zetas: list) -> torch.Tensor:
  """
  Number Theoretic Transform multiplication.
  """
  n = a.shape[-1]
  
  new_coeffs = torch.zeros_like(a, dtype=torch.complex128)
  for i in range(n // 4):
    pair = 4 * i
    f0, f1, f2, f3 = a[..., pair], a[..., pair + 1], a[..., pair + 2], a[..., pair + 3]
    g0, g1, g2, g3 = b[..., pair], b[..., pair + 1], b[..., pair + 2], b[..., pair + 3]
    zeta = ntt_zetas[n // 4 + i]
    r0, r1 = fourier_ntt_base_mul(f0, f1, g0, g1, zeta)
    r2, r3 = fourier_ntt_base_mul(f2, f3, g2, g3, -zeta)
    new_coeffs[..., pair], new_coeffs[..., pair + 1] = r0, r1
    new_coeffs[..., pair + 2], new_coeffs[..., pair + 3] = r2, r3
  return new_coeffs

def fourier_matmul(matrix: torch.Tensor, vector: torch.Tensor, ntt_zetas: list) -> torch.Tensor:
  m, n = matrix.shape[-3], matrix.shape[-2]
  n_, l = vector.shape[-3], vector.shape[-2]
  assert n == n_, "Matrix and vector shapes do not match"
  assert matrix.shape[-1] == vector.shape[-1], "Polynomial sizes do not match"

  pol_size = matrix.shape[-1]

  elements = torch.zeros_like(vector, dtype=torch.complex128)
  for i in range(m):
    for j in range(l):
      product = torch.ones(pol_size, dtype=torch.complex128)
      for k in range(n):
        product *= fourier_ntt_mul(vector[k, j], matrix[i, k], ntt_zetas)
      elements[i, j] = product
  return elements