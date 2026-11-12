from pathlib import Path

from .ls.gauss_elim import gauss_elim
from .pipeline import Metodo, rodar_pipeline

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
}


def main() -> None:

    # ------------------------- Questão 1 -----------------------------
    resultadosQ1 = rodar_pipeline(PASTA_A, PASTA_B, METODOS, B_MIN, B_MAX, SEED, SUFIXO_MATRIZ_B)

    print("------------Resultados questão 1------------")
    print(f"\n{len(resultadosQ1)} resultado(s).")
    for r in resultadosQ1:
        print(f"- {r['metodo']} | {r['matriz']} | resíduo máx={r['residuo_max']:.3e}")

    # ------------------------- Questão 2 ------------------------------
    # todo
if __name__ == "__main__":
    main()