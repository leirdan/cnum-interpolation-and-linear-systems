"""
MatrixMarket Downloader + Filtro de Dominância Diagonal
========================================================

Fusão do antigo `matrixmarket_downloader_from_searchbynamepage.py` com o
antigo `newesttest.py`. Dois passos numa única execução:

1) DOWNLOAD: varre https://math.nist.gov/MatrixMarket/matrices.html (lista
   completa de matrizes), baixa cada .mtx.gz para a pasta de staging (--output).

2) FILTRO: descompacta os .gz, descarta matrizes que não sejam quadradas nem
   diagonalmente dominantes (estritamente, por linhas OU colunas) e move as
   aprovadas para a pasta de destino (--alvo, padrão: ../matrizes_alvo),
   gerando junto o vetor lado direito `<nome>_matriz_b.mtx` (coluna de 1s)
   na pasta de vetores (--alvo-b, padrão: ../matrizes_b).

Os benchmarks Java leem A de `matrizes_alvo` e b de `matrizes_b` — é só
colocar o par `<nome>.mtx` + `<nome>_matriz_b.mtx` com o mesmo nome base
nessas duas pastas para escolher o que medir.

Uso:
    python matrixmarket_downloader_from_searchbynamepage.py                  # baixa e filtra
    python matrixmarket_downloader_from_searchbynamepage.py --skip-download  # filtra só o já baixado
    python matrixmarket_downloader_from_searchbynamepage.py --skip-filter    # só baixa
"""

import argparse
import glob
import gzip
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import requests
import scipy.io
import scipy.sparse
from bs4 import BeautifulSoup

# ── Configurações ──────────────────────────────────────────────────────────────
BASE_URL = "https://math.nist.gov"
INDEX_URL = "https://math.nist.gov/MatrixMarket/matrices.html"
HEADERS = {"User-Agent": "MatrixMarket-Scraper/1.0 (educational use)"}

SCRIPT_DIR = Path(__file__).resolve().parent
PASTA_STAGING_PADRAO = SCRIPT_DIR / "matrizes_baixadas"
PASTA_ALVO_PADRAO = SCRIPT_DIR.parent / "matrizes_alvo"
PASTA_ALVO_B_PADRAO = SCRIPT_DIR.parent / "matrizes_b"

SUFIXO_MATRIZ_B = "_matriz_b"


# ── Helpers de download ────────────────────────────────────────────────────────

def get_soup(url: str, session: requests.Session) -> BeautifulSoup:
    resp = session.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def find_all_matrix_links(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """
    Coleta todos os links de matrizes da página matrices.html.
    Ex: <a href="/MatrixMarket/data/Harwell-Boeing/acoust/young1c.html">YOUNG1C</a>
    Retorna lista de (nome, url_absoluta).
    """
    matrices = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Links de matrizes: estão dentro de /MatrixMarket/data/ e terminam em .html
        if "/MatrixMarket/data/" in href and href.endswith(".html"):
            url = urljoin(BASE_URL, href)
            if url not in seen:
                seen.add(url)
                name = a.get_text(strip=True)
                matrices.append((name, url))
    return matrices


def find_mtx_gz_link(soup: BeautifulSoup) -> str | None:
    """
    Na página de uma matriz, busca o link para o arquivo .mtx.gz.
    Ex: <a href="/pub/MatrixMarket2/Harwell-Boeing/acoust/young1c.mtx.gz">
    """
    for a in soup.find_all("a", href=True):
        if a["href"].endswith(".mtx.gz"):
            return urljoin(BASE_URL, a["href"])
    return None


def dest_path(mtx_url: str, output_dir: Path) -> Path:
    """
    Deriva o caminho de destino a partir da URL do .mtx.gz.
    Ex: .../Harwell-Boeing/acoust/young1c.mtx.gz
        → output_dir/acoust/young1c.mtx.gz
    """
    parts = mtx_url.split("/")
    filename = parts[-1]           # young1c.mtx.gz
    set_name = parts[-2]           # acoust
    return output_dir / set_name / filename


def download_file(url: str, dest: Path, session: requests.Session) -> bool:
    if dest.exists():
        print(f"    ↩  Já existe: {dest.name}")
        return True
    try:
        with session.get(url, headers=HEADERS, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
        print(f"    ✔  {dest.name}  ({dest.stat().st_size:,} bytes)")
        return True
    except requests.RequestException as exc:
        print(f"    ✘  Falha: {exc}")
        return False


def baixar(output_dir: Path, delay: float, limit: int | None, verbose: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    stats = {"total": 0, "downloaded": 0, "skipped": 0, "no_mtx": 0, "failed": 0}

    # ── 1. Coleta todos os links da página índice ──────────────────────────────
    print(f"Abrindo lista completa: {INDEX_URL}\n")
    index_soup = get_soup(INDEX_URL, session)

    matrices = find_all_matrix_links(index_soup)
    if not matrices:
        print("Nenhuma matriz encontrada. Verifique o HTML da página.")
        sys.exit(1)

    total = len(matrices) if limit is None else min(len(matrices), limit)
    print(f"{len(matrices)} matrizes encontradas. Processando {total}.\n")
    print("-" * 55)

    # ── 2. Para cada matriz: acessa a página e baixa o .mtx.gz ────────────────
    for idx, (name, mat_url) in enumerate(matrices, 1):
        if limit is not None and stats["total"] >= limit:
            break

        stats["total"] += 1
        print(f"[{idx}/{total}] {name}")
        if verbose:
            print(f"    {mat_url}")

        # Acessa a página da matriz
        try:
            mat_soup = get_soup(mat_url, session)
        except requests.RequestException as exc:
            print(f"    ✘  Página inacessível: {exc}")
            stats["failed"] += 1
            time.sleep(delay)
            continue

        # Busca link .mtx.gz
        mtx_url = find_mtx_gz_link(mat_soup)
        if not mtx_url:
            print("    ⚠  .mtx.gz não encontrado nesta página.")
            stats["no_mtx"] += 1
            time.sleep(delay)
            continue

        if verbose:
            print(f"    {mtx_url}")

        # Download
        dest = dest_path(mtx_url, output_dir)
        if dest.exists():
            stats["skipped"] += 1
            print(f"    ↩  Já existe: {dest.name}")
        elif download_file(mtx_url, dest, session):
            stats["downloaded"] += 1
        else:
            stats["failed"] += 1

        time.sleep(delay)

    # ── Resumo ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("RESUMO DO DOWNLOAD")
    print(f"  Processadas     : {stats['total']}")
    print(f"  Baixadas        : {stats['downloaded']}")
    print(f"  Já existiam     : {stats['skipped']}")
    print(f"  Sem .mtx.gz     : {stats['no_mtx']}")
    print(f"  Falhas          : {stats['failed']}")
    print(f"  Destino (staging): {output_dir.resolve()}")
    print("=" * 55)


# ── Filtro de dominância diagonal (fusão do antigo newesttest.py) ─────────────

def check_diagonal_dominance(mat, strict=True):
    """
    Verifica se a matriz é Diagonalmente Dominante pelo:
    1. Critério das Linhas:  |a_ii| > sum_{j != i} |a_ij|
    2. Critério das Colunas: |a_jj| > sum_{i != j} |a_ij|

    Retorna (bool, mensagem)
    """
    # 1. Teste pelo Critério das Linhas (usa formato CSR)
    csr = mat.tocsr() if not scipy.sparse.isspmatrix_csr(mat) else mat
    diag = np.abs(csr.diagonal())

    if np.any(diag == 0):
        return False, "Possui elementos zero na diagonal principal"

    # Soma dos valores absolutos por linha ignorando a diagonal
    row_sums = np.array(np.abs(csr).sum(axis=1)).flatten() - diag

    is_row_dd = np.all(diag > row_sums) if strict else np.all(diag >= row_sums)
    if is_row_dd:
        return True, "Aprovada: Dominante por Linhas (Critério das Linhas)"

    # 2. Teste pelo Critério das Colunas (usa formato CSC)
    csc = mat.tocsc() if not scipy.sparse.isspmatrix_csc(mat) else mat
    col_sums = np.array(np.abs(csc).sum(axis=0)).flatten() - diag

    is_col_dd = np.all(diag > col_sums) if strict else np.all(diag >= col_sums)
    if is_col_dd:
        return True, "Aprovada: Dominante por Colunas (Critério das Colunas)"

    return False, "Reprovada: Não é diagonalmente dominante nem por linhas nem por colunas"


def extrair_gz(staging_dir: Path) -> int:
    """Descompacta todos os .mtx.gz do staging (remove o .gz original)."""
    extraidos = 0
    for gz_path in sorted(staging_dir.rglob("*.mtx.gz")):
        mtx_path = Path(str(gz_path)[:-3])  # remove o '.gz'
        try:
            with gzip.open(gz_path, "rb") as f_in, open(mtx_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            gz_path.unlink()
            extraidos += 1
        except Exception as e:
            print(f"Erro ao extrair {gz_path}: {e}")
    return extraidos


def filtrar(staging_dir: Path, alvo_dir: Path, alvo_b_dir: Path, manter_rejeitadas: bool) -> None:
    """
    FASE 2: triagem por dominância diagonal.
    Aprovadas → movidas para alvo_dir + vetor <nome>_matriz_b.mtx em alvo_b_dir.
    Reprovadas → apagadas (ou preservadas em staging/rejeitadas/ com --manter-rejeitadas).
    """
    alvo_dir.mkdir(parents=True, exist_ok=True)
    alvo_b_dir.mkdir(parents=True, exist_ok=True)
    rejeitadas_dir = staging_dir / "rejeitadas"

    pendentes = [
        f for f in sorted(staging_dir.rglob("*.mtx"))
        if not f.name.endswith(f"{SUFIXO_MATRIZ_B}.mtx")
        and alvo_dir not in f.parents
        and rejeitadas_dir not in f.parents
    ]

    print(f"\n>>> INICIANDO ANÁLISE DE DOMINÂNCIA DIAGONAL")
    print(f"Total de matrizes para analisar: {len(pendentes)}")
    print("-" * 75)

    aprovadas = 0
    rejeitadas = 0
    existentes = 0

    for mtx_path in pendentes:
        nome_arquivo = mtx_path.name
        nome_base = mtx_path.stem
        destino = alvo_dir / nome_arquivo

        if destino.exists():
            print(f"[{nome_arquivo}] JÁ ESTÁ EM matrizes_alvo — ignorado.")
            existentes += 1
            continue

        try:
            mat = scipy.io.mmread(mtx_path)
            if not scipy.sparse.issparse(mat):
                mat = scipy.sparse.csr_matrix(mat)

            if mat.shape[0] != mat.shape[1]:
                print(f"[{nome_arquivo}] REJEITADO - Não é quadrada.")
                if manter_rejeitadas:
                    rejeitadas_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(mtx_path, rejeitadas_dir / nome_arquivo)
                else:
                    mtx_path.unlink()
                rejeitadas += 1
                continue

            # Executa a verificação de dominância diagonal
            is_dd, mensagem = check_diagonal_dominance(mat, strict=True)

            if is_dd:
                print(f"[{nome_arquivo}] SUCESSO - {mensagem}")
                shutil.move(mtx_path, destino)

                # Gera o vetor b (lado direito) como um vetor coluna de 1s
                # Necessário para os benchmarks em Java (vai para a pasta de vetores b)
                b_coluna = np.ones((mat.shape[0], 1))
                scipy.io.mmwrite(alvo_b_dir / f"{nome_base}{SUFIXO_MATRIZ_B}.mtx", b_coluna)

                aprovadas += 1
            else:
                print(f"[{nome_arquivo}] DELETADO - {mensagem}")
                if manter_rejeitadas:
                    rejeitadas_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(mtx_path, rejeitadas_dir / nome_arquivo)
                else:
                    mtx_path.unlink()
                rejeitadas += 1

        except Exception as e:
            print(f"[{nome_arquivo}] ERRO - {e}")
            if mtx_path.exists():
                if manter_rejeitadas:
                    rejeitadas_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(mtx_path, rejeitadas_dir / nome_arquivo)
                else:
                    mtx_path.unlink()
            rejeitadas += 1

    print("-" * 75)
    print(f"RESUMO DO FILTRO: {aprovadas} aprovadas | {rejeitadas} rejeitadas "
          f"| {existentes} já existiam em {alvo_dir.name} "
          f"(vetores b em {alvo_b_dir.name}).")

    # ── FASE 3: limpeza das subpastas vazias do staging ────────────────────────
    for pasta in sorted((p for p in staging_dir.rglob("*") if p.is_dir()), reverse=True):
        try:
            pasta.rmdir()  # só remove se estiver vazia
        except OSError:
            pass


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa matrizes do MatrixMarket, filtra por dominância diagonal "
                    "e salva A em matrizes_alvo e o vetor b em matrizes_b."
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=PASTA_STAGING_PADRAO,
        help=f"Pasta de staging do download (padrão: {PASTA_STAGING_PADRAO})",
    )
    parser.add_argument(
        "--alvo", type=Path, default=PASTA_ALVO_PADRAO,
        help=f"Pasta de destino das matrizes A (padrão: {PASTA_ALVO_PADRAO})",
    )
    parser.add_argument(
        "--alvo-b", type=Path, default=PASTA_ALVO_B_PADRAO,
        help=f"Pasta de destino dos vetores b lado direito (padrão: {PASTA_ALVO_B_PADRAO})",
    )
    parser.add_argument(
        "--delay", "-d", type=float, default=1.0,
        help="Pausa em segundos entre requisições (padrão: 1.0)",
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=None,
        help="Máximo de matrizes a baixar, útil para testes",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Exibe URLs completas durante o execução",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Não baixa nada; roda apenas o filtro no que já está em --output",
    )
    parser.add_argument(
        "--skip-filter", action="store_true",
        help="Apenas baixa; não filtra nem move para --alvo",
    )
    parser.add_argument(
        "--manter-rejeitadas", action="store_true",
        help="Em vez de apagar as matrizes reprovadas, move-as para <output>/rejeitadas/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_download:
        baixar(output_dir=args.output, delay=args.delay, limit=args.limit, verbose=args.verbose)

    if args.skip_filter:
        return

    print("\n" + "=" * 55)
    print("FASE DE FILTRO (dominância diagonal)")
    print(f"  Staging : {args.output.resolve()}")
    print(f"  Alvo(A) : {args.alvo.resolve()}")
    print(f"  Alvo(b) : {args.alvo_b.resolve()}")
    print("=" * 55)

    if not args.output.is_dir():
        print(f"Pasta de staging inexistente: {args.output} — nada a filtrar.")
        return

    extraidos = extrair_gz(args.output)
    if extraidos:
        print(f"- {extraidos} arquivos .gz descompactados.")
    filtrar(args.output, args.alvo, args.alvo_b, manter_rejeitadas=args.manter_rejeitadas)
    print("\nTriagem e organização finalizadas!")


if __name__ == "__main__":
    main()
