import numpy as np
from .utils import rel_dist

def gauss_jacobi(a: np.ndarray, b: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Soluciona o sistema linear Ax = b usando o método iterativo de Jacobi.
    """

    # Evitar erros de divisão por zero.
    if (np.any(np.isclose(a.diagonal(), 0))):
        raise ValueError("A matriz A possui elementos nulos na diagonal, logo, não é possível aplicar o método de Jacobi.")

    # Retiramos a diagonal de A para evitar verificações como i != j,
    # possibilitando usar o operador @ para multiplicação de matrizes de forma facilitada.
    a_without_diag = a.copy()
    np.fill_diagonal(a_without_diag, 0)

    # Computamos antecipadamente a diagonal de A para evitar recalcular a cada iteração.
    a_diagonal = a.diagonal()

    # X_0 começa como [b_i / a_ii] para cada elemento e X_1 começa como um vetor de zeros.
    x_prev = np.asarray(b, dtype=np.float64) / a.diagonal() 
    x_next = np.zeros_like(b, dtype=np.float64)

    while 1:
      # Calcula o próximo x^k usando a fórmula de Jacobi tradicional.
      x_next = (b - (a_without_diag @ x_prev)) / a_diagonal
      print(f"Iteração {x_next=}")
      if (np.abs(rel_dist(x_next, x_prev)) < epsilon):
        break

      if (np.any(np.isnan(x_next)) or np.any(np.isinf(x_next))):
        raise ValueError("O método de Jacobi divergiu. Verifique a matriz A e o vetor b.")

      # Caso não tenha atingido o critério de convergência, atualiza x^k-1 para a próxima iteração.
      x_prev = x_next.copy()

    return x_next
