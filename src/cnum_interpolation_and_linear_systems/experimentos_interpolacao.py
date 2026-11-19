"""
Experimentos da Questão 2 — Interpolação polinomial (itens 2.2 e 2.3).

  2.2  f(x) = e^x em [1, 3], n = 3, 4, 5, 10, pelo sistema linear e por Lagrange.
  2.3  f(x) = 1/(1 + 25x^2) em [0, 4], n = 4, 10, 15, pelo sistema linear.
  (n é o grau do polinômio interpolador, logo são usados n+1 pontos.)

As configurações (intervalos, graus, pontos de teste, malha, pasta de saída) ficam no
main.py e são passadas como parâmetros para as funções deste arquivo.

A comparação com a função original é feita em tabelas, impressas no terminal e
salvas em Markdown em resultados_questao2/ (tabelas_questao2_2.md e tabelas_questao2_3.md).

Como rodar (na raiz do projeto):
    uv run main                                                                   (questões 1 e 2)
    uv run python -m cnum_interpolation_and_linear_systems.experimentos_interpolacao   (só a 2)
"""
import math

import numpy as np

from .ls.interpolacao_lagrange import lagrange
from .ls.interpolacao_sist_linear import avaliar_polinomio, interpolacao_sist_linear, matriz_vandermonde

# TODO: reativar quando o módulo de diferenças divididas existir
# from .interpolacao_diferencas_divididas import coeficientes_newton, newton


def f_exp(x):
    """f(x) = e^x (item 2.2)."""
    return np.exp(x)


def f_runge(x):
    """f(x) = 1 / (1 + 25x^2) (item 2.3, função de Runge)."""
    return 1.0 / (1.0 + 25.0 * np.asarray(x, dtype=float) ** 2)


def pontos_igualmente_espacados(f, a, b, n):
    """
    Entrada: f -> função; [a, b] -> intervalo; n -> grau do polinômio.
    Saída:   (xs, ys) com n+1 pontos igualmente espaçados: xs_i = a + i*h, h = (b - a)/n.
    """
    h = (b - a) / n
    xs = a + h * np.arange(n + 1)
    return xs, f(xs)


def produto_nodal(xs, x):
    """
    Entrada: nós xs e pontos x.
    Saída:   (x - x_0)(x - x_1)...(x - x_n) em cada x (usado na cota do erro de truncamento).
    """
    w = np.ones_like(x)
    for xi in xs:
        w = w * (x - xi)
    return w


def _num(v):
    """Formata um número curto com vírgula decimal (ex.: 0.5 -> '0,5'), para títulos de tabela."""
    return f"{v:g}".replace(".", ",")


def tabela(titulo, cabecalho, linhas):
    """
    Imprime uma tabela alinhada no terminal e devolve a mesma tabela em Markdown.

    Entrada: titulo -> str; cabecalho -> lista de str;
             linhas -> lista de linhas, cada uma uma lista de str já formatadas.
    Saída:   str com a tabela em Markdown (para colar no relatório).
    """
    larguras = [max(len(c), *(len(lin[i]) for lin in linhas)) for i, c in enumerate(cabecalho)]
    print(f"\n{titulo}")
    print("  ".join(c.rjust(w) for c, w in zip(cabecalho, larguras)))
    print("  ".join("-" * w for w in larguras))
    for lin in linhas:
        print("  ".join(v.rjust(w) for v, w in zip(lin, larguras)))

    md = [f"**{titulo}**", "", "| " + " | ".join(cabecalho) + " |",
          "|" + "|".join("---:" for _ in cabecalho) + "|"]
    md += ["| " + " | ".join(lin) + " |" for lin in linhas]
    return "\n".join(md)


def questao_2_2(a, b, graus, pontos_teste, n_malha) -> list[str]:
    """
    Item 2.2: interpola e^x em [a, b] pelo sistema linear (SL) e por Lagrange (LG)
    e compara os dois com a função original.

    Entrada:
        a, b         -> intervalo [a, b];
        graus        -> lista com os graus n do polinômio interpolador;
        pontos_teste -> pontos x (que não são nós) onde P_n(x) é comparado com e^x;
        n_malha      -> nº de pontos da malha fina usada para medir o erro máximo.
    Tabelas geradas:
      - uma por grau n: P_n(x) e o erro |e^x - P_n(x)| nos pontos de teste, para cada método;
      - resumo: erro máximo na malha fina e erro relativo máximo de cada método, e a cota teórica
        do erro de truncamento  e^b/(n+1)! * max|(x-x0)...(x-xn)|.
    Saída: lista com as tabelas em Markdown.
    """
    malha = np.linspace(a, b, n_malha)
    f_malha = f_exp(malha)
    tabelas_md = []
    resumo = []

    print(f"============ Questão 2.2: f(x) = e^x em [{_num(a)}, {_num(b)}] "
          "(sistema linear e Lagrange) ============")

    for n in graus:
        xs, ys = pontos_igualmente_espacados(f_exp, a, b, n)
        coefs = interpolacao_sist_linear(xs, ys)

        # comparação ponto a ponto com e^x
        linhas = []
        for x in pontos_teste:
            fx = math.exp(x)
            p_sl = avaliar_polinomio(coefs, x)
            p_lg = lagrange(xs, ys, x)
            linhas.append([f"{x:.1f}", f"{fx:.8f}",
                           f"{p_sl:.8f}", f"{abs(fx - p_sl):.2e}",
                           f"{p_lg:.8f}", f"{abs(fx - p_lg):.2e}"])
        tabelas_md.append(tabela(
            f"2.2 (n = {n}): e^x x P_{n}(x)",
            ["x", "e^x", "P SL", "erro SL", "P LG", "erro LG"],
            linhas))

        # resumo do grau n (erro medido na malha fina)
        p_sl = avaliar_polinomio(coefs, malha)
        erro_sl = np.abs(p_sl - f_malha)
        p_lg = lagrange(xs, ys, malha)
        erro_lg = np.abs(p_lg - f_malha)
        # toda derivada de e^x é e^x, cujo máximo em [a, b] é e^b
        cota = math.exp(b) / math.factorial(n + 1) * np.max(np.abs(produto_nodal(xs, malha)))
        resumo.append([str(n), f"{(b - a) / n:.4f}",
                       f"{erro_sl.max():.3e}", f"{erro_lg.max():.3e}",
                       f"{np.max(erro_sl / f_malha):.3e}", f"{np.max(erro_lg / f_malha):.3e}",
                       f"{cota:.3e}"])

    tabelas_md.append(tabela(
        f"2.2 Resumo: erro máximo em [{_num(a)}, {_num(b)}]",
        ["n", "h", "erro máx SL", "erro máx LG", "erro rel máx SL", "erro rel máx LG", "cota teórica"],
        resumo))
    return tabelas_md


def questao_2_3(a, b, graus, pontos_teste, n_malha) -> list[str]:
    """
    Item 2.3: interpola f(x) = 1/(1 + 25x^2) em [a, b] pelo sistema linear e compara com f.

    Entrada:
        a, b         -> intervalo [a, b];
        graus        -> lista com os graus n do polinômio interpolador;
        pontos_teste -> pontos x (que não são nós) onde P_n(x) é comparado com f(x);
        n_malha      -> nº de pontos da malha fina usada para medir o erro máximo.
    Tabelas geradas:
      - comparação ponto a ponto: f(x) e P_n(x) de cada grau, com os erros, lado a lado;
      - resumo: erro máximo (e onde ocorre), erro máximo por trecho (início [a; a+0,5],
        meio, fim [b-0,5; b]), menor valor de P_n (f > 0 sempre), número de condição
        da matriz de Vandermonde e erro de P_n nos próprios nós.
    Saída: lista com as tabelas em Markdown.
    """
    malha = np.linspace(a, b, n_malha)
    f_malha = f_runge(malha)
    tabelas_md = []

    # trechos onde o erro é medido separadamente
    corte_inicio, corte_fim = a + 0.5, b - 0.5
    no_inicio = malha <= corte_inicio
    no_meio = (malha >= corte_inicio) & (malha <= corte_fim)
    no_fim = malha >= corte_fim

    print(f"\n============ Questão 2.3: f(x) = 1/(1+25x^2) em [{_num(a)}, {_num(b)}] "
          "(sistema linear) ============")

    polinomios = {}
    resumo = []
    for n in graus:
        xs, ys = pontos_igualmente_espacados(f_runge, a, b, n)
        coefs = interpolacao_sist_linear(xs, ys)
        polinomios[n] = coefs

        p = avaliar_polinomio(coefs, malha)
        erro = np.abs(p - f_malha)
        cond = np.linalg.cond(matriz_vandermonde(xs))  # só para análise do condicionamento
        erro_nos = np.max(np.abs(avaliar_polinomio(coefs, xs) - ys))
        resumo.append([str(n), f"{(b - a) / n:.4f}",
                       f"{erro.max():.4f}", f"{malha[np.argmax(erro)]:.3f}",
                       f"{erro[no_inicio].max():.2e}",
                       f"{erro[no_meio].max():.2e}",
                       f"{erro[no_fim].max():.2e}",
                       f"{p.min():.4f}", f"{cond:.2e}", f"{erro_nos:.1e}"])

    # comparação ponto a ponto com f(x), os graus lado a lado
    linhas = []
    for x in pontos_teste:
        fx = float(f_runge(x))
        lin = [f"{x:.1f}", f"{fx:.6f}"]
        for n in graus:
            pn = avaliar_polinomio(polinomios[n], x)
            lin += [f"{pn:.6f}", f"{abs(fx - pn):.2e}"]
        linhas.append(lin)
    cab = ["x", "f(x)"]
    for n in graus:
        cab += [f"P_{n}(x)", f"erro n={n}"]
    tabelas_md.append(tabela("2.3: f(x) x P_n(x)", cab, linhas))

    tabelas_md.append(tabela(
        f"2.3 Resumo: erro em [{_num(a)}, {_num(b)}]",
        ["n", "h", "erro máx", "em x",
         f"erro [{_num(a)}; {_num(corte_inicio)}]",
         f"erro [{_num(corte_inicio)}; {_num(corte_fim)}]",
         f"erro [{_num(corte_fim)}; {_num(b)}]",
         "mín P(x)", "cond(V)", "erro nos nós"],
        resumo))
    return tabelas_md


def salvar_tabelas(pasta, tabelas_md, nome_arquivo):
    """
    Salva tabelas em Markdown em pasta/nome_arquivo (para colar no relatório).

    Entrada: pasta -> Path da pasta de saída (criada se não existir);
             tabelas_md -> lista de str em Markdown; nome_arquivo -> nome do arquivo .md.
    Saída:   caminho (Path) do arquivo salvo.
    """
    pasta.mkdir(parents=True, exist_ok=True)
    arquivo = pasta / nome_arquivo
    arquivo.write_text("\n\n".join(tabelas_md) + "\n", encoding="utf-8")
    print(f"\nTabelas salvas em {arquivo}")
    return arquivo


if __name__ == "__main__":
    # Roda só a Questão 2, com as configurações definidas no main.py.
    from .main import questao_2

    questao_2()
