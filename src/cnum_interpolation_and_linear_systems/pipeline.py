from pathlib import Path
from typing import Callable

import numpy as np
import scipy.io
import scipy.sparse

Metodo = Callable[[np.ndarray, np.ndarray], np.ndarray]


# ── Leitura ────────────────────────────────────────────────────────────────────
def _para_denso(m) -> np.ndarray:
    """mmread devolve esparsa (formato coordinate) ou ndarray (formato array)."""
    if scipy.sparse.issparse(m):
        m = m.toarray()
    return np.asarray(m, dtype=float)


def listar_matrizes(pasta_a: Path, sufixo_ignorado: str) -> list[Path]:
    """Todos os .mtx da pasta (recursivamente), ignorando os que terminam com `sufixo_ignorado`."""
    return sorted(
        f for f in pasta_a.rglob("*.mtx")
        if not f.stem.endswith(sufixo_ignorado)
    )


def gerar_b(n: int, b_min: float, b_max: float, seed: int | None) -> np.ndarray:
    """Vetor b original: n floats aleatórios uniformes em [b_min, b_max]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(b_min, b_max, size=n)


def carregar_sistema(
    caminho_a: Path, b_min: float, b_max: float, seed: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Lê A de `caminho_a` e gera o vetor b original aleatório com a ordem de A."""
    a = _para_denso(scipy.io.mmread(caminho_a, spmatrix=False))
    return a, gerar_b(a.shape[0], b_min, b_max, seed)


def salvar_b(b: np.ndarray, pasta_b: Path, nome_matriz: str, sufixo: str) -> Path:
    """Salva o vetor b (como coluna n x 1) em `pasta_b/<nome_matriz><sufixo>.mtx`."""
    pasta_b.mkdir(parents=True, exist_ok=True)
    caminho = pasta_b / f"{nome_matriz}{sufixo}.mtx"
    scipy.io.mmwrite(caminho, b.reshape(-1, 1))
    return caminho


# ── Execução ───────────────────────────────────────────────────────────────────
def residuo_maximo(a: np.ndarray, b: np.ndarray, xs: np.ndarray) -> float:
    """Maior valor em módulo de |b_original - b_resultado|, com b_resultado = A @ xs."""
    return float(np.max(np.abs(b - a @ xs)))


def rodar_pipeline(
    pasta_a: Path,
    pasta_b: Path,
    metodos: dict[str, Metodo],
    b_min: float,
    b_max: float,
    seed: int | None,
    sufixo_b: str,
) -> list[dict]:
    """
    Roda cada método em cada matriz e salva o b gerado de cada matriz em `pasta_b`
    (<nome>{sufixo_b}.mtx). Cada resultado é um dicionário com:
      - "metodo":        nome do método
      - "matriz":        nome da matriz
      - "xs":            vetor solução
      - "residuo_max":   max(|b_original - A @ xs|)
    """
    resultados: list[dict] = []

    arquivos = listar_matrizes(pasta_a, sufixo_b)
    if not arquivos:
        print(f"Nenhum .mtx encontrado em {pasta_a}")
        return resultados

    for caminho in arquivos:
        nome = caminho.stem
        a, b = carregar_sistema(caminho, b_min, b_max, seed)

        if a.shape[0] != a.shape[1]:
            print(f"[{nome}] ignorada: A {a.shape} não é quadrada.")
            continue

        salvar_b(b, pasta_b, nome, sufixo_b)

        for nome_metodo, metodo in metodos.items():
            try:
                xs = metodo(a, b)
            except Exception as e:  # um método falhar não deve derrubar a pipeline
                print(f"[{nome}] {nome_metodo} falhou: {e}")
                continue

            residuo = residuo_maximo(a, b, xs)
            resultados.append({
                "metodo": nome_metodo,
                "matriz": nome,
                "xs": xs,
                "residuo_max": residuo,
            })
    return resultados