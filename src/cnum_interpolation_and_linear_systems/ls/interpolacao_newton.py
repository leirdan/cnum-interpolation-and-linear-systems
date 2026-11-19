import numpy as np

from .interpolacao_sist_linear import validar_pontos

def coeficientes_newton(xs, ys):
    """
    Monta a tabela de diferenças divididas e devolve [d_0, d_1, ..., d_n],
    onde d_i = f[x_0, x_1, ..., x_i] (topo da coluna de ordem i).

    Ordem 0:  f[x_j] = y_j
    Ordem i:  f[x_j, ..., x_{j+i}] = (f[x_{j+1}, ..., x_{j+i}] - f[x_j, ..., x_{j+i-1}]) / (x_{j+i} - x_j)
    """
    xs, ys = validar_pontos(xs, ys)
    n = len(xs)
    coluna = list(ys)
    d = [coluna[0]]
    for ordem in range(1, n):
        nova_coluna = []
        for j in range(n - ordem):
            nova_coluna.append((coluna[j + 1] - coluna[j]) / (xs[j + ordem] - xs[j]))
        coluna = nova_coluna
        d.append(coluna[0])
    return np.array(d)


def newton(xs, ys, x):
    """
    Calcula P_n(x) = d_0 + d_1 (x - x_0) + d_2 (x - x_0)(x - x_1) + ...
                   + d_n (x - x_0)(x - x_1)...(x - x_{n-1}).
    x pode ser um número, uma lista ou um np.ndarray.
    """
    xs, ys = validar_pontos(xs, ys)
    x = np.asarray(x, dtype=float)
    d = coeficientes_newton(xs, ys)
    soma = 0.0
    produto = 1.0
    for i in range(len(d)):
        soma = soma + d[i] * produto
        produto = produto * (x - xs[i])
    return soma
