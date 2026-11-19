import numpy as np
from .utils import rel_dist, abs_dist

def gauss_seidel(a: np.ndarray, b: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Soluciona o sistema linear Ax = b usando o método iterativo de Gauss-Seidel.
    """

    # Evitar erros de divisão por zero.
    if (np.any(np.isclose(a.diagonal(), 0))):
        raise ValueError("A matriz A possui elementos nulos na diagonal, logo, não é possível aplicar o método de Gauss-Seidel.")

    # Retiramos a diagonal de A para evitar verificações como i != j,
    # possibilitando usar o operador @ para o produto escalar de cada linha de forma facilitada.
    a_without_diag = a.copy()
    np.fill_diagonal(a_without_diag, 0)

    # Computamos antecipadamente a diagonal de A para evitar recalcular a cada iteração.
    a_diagonal = a.diagonal()

    # X_0 começa como [b_i / a_ii] para cada elemento.
    x_prev = np.asarray(b, dtype=np.float64) / a_diagonal
    n = len(b)

    while 1:
        # Inicializamos x_next com os valores da iteração anterior para que, ao longo do loop,
        # os índices j < i já contenham os valores recém-atualizados e os índices j > i
        # mantenham os valores de x_prev.
        x_next = x_prev.copy()

        # Calcula o próximo x^k elemento a elemento usando a fórmula de Gauss-Seidel.
        for i in range(n):
            x_next[i] = (b[i] - (a_without_diag[i] @ x_next)) / a_diagonal[i]


        # Critério de parada dist relativa e dist absoluta
        if (abs_dist(x_next, x_prev) < epsilon and rel_dist(x_next, x_next) < epsilon):
            break

        if (np.any(np.isnan(x_next)) or np.any(np.isinf(x_next))):
            raise ValueError("O método de Gauss-Seidel divergiu. Verifique a matriz A e o vetor b.")

        # Caso não tenha atingido o critério de convergência, atualiza x^k-1 para a próxima iteração.
        x_prev = x_next.copy()

    return x_next