"""
Interpolação polinomial via SISTEMA LINEAR (matriz de Vandermonde).

Procura P_n(x) = a_0 + a_1 x + ... + a_n x^n tal que P_n(x_i) = y_i (i = 0..n).
Cada ponto gera uma equação, o que dá o sistema V a = y, com V[i][j] = x_i^j
(matriz de Vandermonde). O sistema é resolvido por Eliminação de Gauss com
pivoteamento parcial (gauss_elim.py, Questão 1).
"""
import numpy as np

from .gauss_elim import gauss_elim


def validar_pontos(xs, ys):
    """
    Valida e converte o tabelamento (xs, ys) para interpolação.

    Entrada: xs, ys -> listas ou arrays com n+1 valores cada.
    Saída:   (xs, ys) como np.ndarray 1D de float.
    Erro:    ValueError se não houver pontos, se os tamanhos forem diferentes
             ou se houver x repetido (o polinômio único exige xs distintos).
    """
    xs = np.asarray(xs, dtype=float).ravel()
    ys = np.asarray(ys, dtype=float).ravel()
    if xs.size == 0:
        raise ValueError("O tabelamento precisa de pelo menos um ponto.")
    if xs.size != ys.size:
        raise ValueError("O tabelamento precisa de xs e ys com o mesmo tamanho.")
    if np.any(np.diff(np.sort(xs)) == 0):
        raise ValueError("O tabelamento precisa de xs distintos.")
    return xs, ys


def matriz_vandermonde(xs):
    """
    Gera a matriz de Vandermonde V, com V[i, j] = xs[i] ** j.

    Entrada: xs -> np.ndarray com x_0, ..., x_n.
    Saída:   matriz (n+1) x (n+1) de float.
    Cada coluna é a anterior multiplicada por xs (evita recalcular potências).
    """
    m = len(xs)
    v = np.empty((m, m), dtype=float)
    v[:, 0] = 1.0
    for j in range(1, m):
        v[:, j] = v[:, j - 1] * xs
    return v


def interpolacao_sist_linear(xs, ys):
    """
    Interpolação polinomial resolvendo o sistema V a = y
    (Eliminação de Gauss com pivoteamento parcial, ls/gauss_elim.py).

    Entrada: xs, ys -> tabelamento com n+1 pontos (xs distintos).
    Saída:   np.ndarray [a_0, a_1, ..., a_n], em ordem crescente de potência:
             P_n(x) = a_0 + a_1 x + ... + a_n x^n.
    """
    xs, ys = validar_pontos(xs, ys)
    v = matriz_vandermonde(xs)
    return gauss_elim(v, ys)


def avaliar_polinomio(coefs, x):
    """
    Avalia P(x) = a_0 + a_1 x + ... + a_n x^n pelo método de Horner.

    Entrada: coefs -> [a_0, ..., a_n] (saída de interpolacao_sist_linear);
             x -> número ou np.ndarray.
    Saída:   P(x), no mesmo formato de x.
    """
    x = np.asarray(x, dtype=float)
    p = np.full_like(x, coefs[-1], dtype=float)
    for a_k in coefs[-2::-1]:  # a_{n-1}, ..., a_0
        p = p * x + a_k
    return p if p.ndim else float(p)
