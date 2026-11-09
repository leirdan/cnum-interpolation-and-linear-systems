import numpy as np
from .ls.gauss_elim import gauss_elim
from .ls.lu import lu

def main() -> None:
    print("Hello from cnum-interpolation-and-linear-systems!")

    a = np.array([[2, 1, -1],
                    [-3, -1, 2],
                    [-2, 1, 2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)

    print("Matrizes iniciais:")
    print(a)
    print(b)

    print("Solução via eliminação de Gauss:") 
    print(gauss_elim(a, b))

    print("Solução via fatoração LU:")
    print(lu(a, b))

