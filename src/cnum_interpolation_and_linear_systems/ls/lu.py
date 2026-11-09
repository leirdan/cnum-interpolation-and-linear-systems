import numpy as np
from .utils import forward_substitution
from .utils import back_substitution

def lu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Soluciona o sistema linear Ax = b a partir da decomposição LU de A.
    """
    l, u = _decomp(a)
    y = forward_substitution(l, b)
    x = back_substitution(u, y)
    return x

def _decomp(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompõe a matriz A em uma matriz triangular inferior L e uma matriz triangular superior U.
    """
    a = np.asarray(a, dtype=float)

    n = a.shape[0]
    lower = np.eye(n, dtype=float)
    upper = a.copy()

    for k in range(n):
        for i in range(k + 1, n):
            if (upper[k, k] == 0):
                raise ValueError("Pivô nulo encontrado, logo, matriz L ou U não é invertível e não é possível de fatorar A.")

            multiplicador = upper[i, k] / upper[k, k]
            lower[i, k] = multiplicador 
            upper[i, k:] -= multiplicador * upper[k, k:]

    return lower, upper
    