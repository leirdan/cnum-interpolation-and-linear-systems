import numpy as np

def forward_substitution(l: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Realiza a substituição para frente para encontrar a solução do sistema linear Ly = b.
    Idealmente usada no algoritmo da fatoração LU.
    """
    n = l.shape[0]
    y = np.zeros(n)

    for i in range(n):
        y[i] = (b[i] - np.dot(l[i, :i], y[:i])) / l[i, i]

    return y 

def back_substitution(a: np.ndarray, b : np.ndarray = None) -> np.ndarray:
    """
    Realiza a retrosubstituição para encontrar a solução do sistema linear.
    Pode receber (a, b) separadamente ou a matriz aumentada (aum).
    """
    if b is not None:
        aum = np.hstack((a, b.reshape(-1, 1)))
    else:
        aum = a

    k = aum.shape[0]
    x = np.zeros(k)

    # Aplica a fórmula geral da retrosubstituição pra cada variável x_i
    for i in range(k - 1, -1, -1):
        x[i] = (aum[i, -1] - np.dot(aum[i, i + 1:k], x[i + 1:k])) / aum[i, i]

    return x