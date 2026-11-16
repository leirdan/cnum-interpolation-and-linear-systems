from pathlib import Path

from .ls.gauss_elim import gauss_elim
from .ls.lu import lu
from .ls.gauss_jacobi import gauss_jacobi
from .pipeline import Metodo, rodar_pipeline
from .experimentos_interpolacao import questao_2_2, questao_2_3, salvar_tabelas

# ── Configurações Questão 1 ──────────────────────────────────────────────────────────────
SUFIXO_MATRIZ_B = "_matriz_b"

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
PASTA_A = RAIZ_PROJETO / "matrizes_alvo"
PASTA_B = RAIZ_PROJETO / "matrizes_b"  # onde o b gerado de cada matriz é salvo (.mtx)

# b original: floats aleatórios entre [B_MIN, B_MAX].
# A semente fixa deixa o b reprodutível (mesmo b para a mesma matriz, em qualquer execução).
B_MIN, B_MAX = -50.0, 50.0
SEED = 67

# Para adicionar um novo método: importe a função e acrescente uma entrada aqui.
# Todo método deve ter a assinatura (a: np.ndarray, b: np.ndarray) -> np.ndarray.
METODOS: dict[str, Metodo] = {
    "eliminacao_gauss": gauss_elim,
    "eliminacao_lu": lu,
    "jacobi": gauss_jacobi,
}

# ── Configurações Questão 2.2 ────────────────────────────────────────────────────────────
# f(x) = e^x em [1, 3]
PASTA_Q2 = RAIZ_PROJETO / "resultados_questao2"  # onde as tabelas da questão 2 são salvas (.md)
N_MALHA = 2001  # pontos usados para medir o erro máximo de P_n(x) em todo o intervalo (2.2 e 2.3)

A_22, B_22 = 1.0, 3.0
GRAUS_22 = [3, 4, 5, 10]
# pontos de comparação: nenhum deles é nó para os graus testados
PONTOS_TESTE_22 = [1.1, 1.3, 1.7, 1.9, 2.1, 2.3, 2.7, 2.9]

# ── Configurações Questão 2.3 ────────────────────────────────────────────────────────────
# f(x) = 1/(1 + 25x^2) em [0, 4], pelo sistema linear
A_23, B_23 = 0.0, 4.0
GRAUS_23 = [4, 10, 15]
# pontos de comparação: nenhum deles é nó para os graus testados
PONTOS_TESTE_23 = [0.1, 0.2, 0.3, 0.5, 0.7, 1.5, 2.5, 3.5, 3.9]


def questao_1() -> None:
    """
    Questão 1: resolve Ax = b para cada matriz de PASTA_A com cada método de METODOS
    (b aleatório em [B_MIN, B_MAX], salvo em PASTA_B) e imprime o resíduo máximo.
    """
    resultados = rodar_pipeline(PASTA_A, PASTA_B, METODOS, B_MIN, B_MAX, SEED, SUFIXO_MATRIZ_B)

    print("------------Resultados questão 1------------")
    print(f"\n{len(resultados)} resultado(s).")
    for r in resultados:
        print(f"- {r['metodo']} | {r['matriz']} | resíduo máx={r['residuo_max']:.3e}")


def questao_2() -> None:
    """
    Questão 2 (itens 2.2 e 2.3): imprime as tabelas de comparação no terminal e
    salva todas em PASTA_Q2/tabelas_questao2.md. Ver experimentos_interpolacao.py.
    """
    # ------------------------- Questão 2.2 ---------------------------
    tabelas_md = questao_2_2(A_22, B_22, GRAUS_22, PONTOS_TESTE_22, N_MALHA)

    # ------------------------- Questão 2.3 ---------------------------
    tabelas_md += questao_2_3(A_23, B_23, GRAUS_23, PONTOS_TESTE_23, N_MALHA)

    salvar_tabelas(PASTA_Q2, tabelas_md)


def main() -> None:

    # ------------------------- Questão 1 -----------------------------
    questao_1()

    # ------------------------- Questão 2 ------------------------------
    print()
    questao_2()


if __name__ == "__main__":
    main()