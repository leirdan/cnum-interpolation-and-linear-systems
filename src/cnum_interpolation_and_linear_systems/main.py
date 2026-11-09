import numpy as np
from .ls.gauss_elim import gauss_elim

def main() -> None:
    print("Hello from cnum-interpolation-and-linear-systems!")

    a = np.array([[2, 1, -1],
                    [-3, -1, 2],
                    [-2, 1, 2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)

    print("Solução:") 
    print(gauss_elim(a, b))
