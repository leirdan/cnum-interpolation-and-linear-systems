import numpy as np
from .utils import back_substitution

def gauss_elim(a : np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Soluciona o sistema linear Ax = b usando eliminação de Gauss.
    """

    aum = np.hstack((a, b.reshape(-1, 1)))  
    k = len(b)

    for i in range(k):
        pivot_row = _get_pivot_idx(aum, i)

        # Troca de linhas
        if pivot_row != i:
            aum[[i, pivot_row]] = aum[[pivot_row, i]]

        pivot = aum[i, i]

        # Atualização das linhas com base no multiplicador
        for j in range(i + 1, k):
            m = aum[j, i] / pivot
            aum[j, i:] = aum[j, i:] - m * aum[i, i:]

    x = back_substitution(aum)
    return x


def _get_pivot_idx(m: np.ndarray, i: int) -> int:
    """
    Retorna o índice da linha com o maior valor absoluto na coluna i a partir da linha i.
    """
    return np.argmax(np.abs(m[i:, i])) + i

