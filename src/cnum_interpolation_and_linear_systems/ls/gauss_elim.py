import numpy as np

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

    x = _back_substitution(aum)
    return x


def _get_pivot_idx(m: np.ndarray, i: int) -> int:
    """
    Retorna o índice da linha com o maior valor absoluto na coluna i a partir da linha i.
    """
    return np.argmax(np.abs(m[i:, i])) + i


def _back_substitution(aum: np.ndarray) -> np.ndarray:
    """
    Realiza a retrosubstituição para encontrar a solução do sistema linear.
    """
    k = aum.shape[0]
    x = np.zeros(k)

    # Aplica a fórmula geral da retrosubstituição pra cada variável x_i
    for i in range(k - 1, -1, -1):
        x[i] = (aum[i, -1] - np.dot(aum[i, i + 1:k], x[i + 1:k])) / aum[i, i]

    return x