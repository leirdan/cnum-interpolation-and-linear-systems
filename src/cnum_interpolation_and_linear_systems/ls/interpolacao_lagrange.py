import numpy as np

from .interpolacao_sist_linear import validar_pontos


def polinomio_lagrange_k(xs, k, x):
    """
    Calcula L_k(x) = prod_{j != k} (x - x_j) / prod_{j != k} (x_k - x_j).
    """
    numerador = 1.0
    denominador = 1.0
    for j in range(len(xs)):
        if j != k:
            numerador = numerador * (x - xs[j])
            denominador = denominador * (xs[k] - xs[j])
    return numerador / denominador


def lagrange(xs, ys, x):
    """
    Calcula P_n(x) = sum_k y_k L_k(x).
    x pode ser um número, uma lista ou um np.ndarray.
    """
    xs, ys = validar_pontos(xs, ys)
    x = np.asarray(x, dtype=float)
    soma = 0.0
    for k in range(len(xs)):
        soma = soma + ys[k] * polinomio_lagrange_k(xs, k, x)
    return soma
