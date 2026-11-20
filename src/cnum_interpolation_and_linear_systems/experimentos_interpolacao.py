import math

import numpy as np

from .ls.interpolacao_lagrange import lagrange
from .ls.interpolacao_newton import newton
from .ls.interpolacao_sist_linear import avaliar_polinomio, interpolacao_sist_linear, matriz_vandermonde


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
    Item 2.2: interpola e^x em [a, b] pelo sistema linear (SL), por Lagrange (LG)
    e por Newton/diferenças divididas (NW) e compara os três com a função original.

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
          "(sistema linear, Lagrange e Newton) ============")

    for n in graus:
        xs, ys = pontos_igualmente_espacados(f_exp, a, b, n)
        coefs = interpolacao_sist_linear(xs, ys)

        # comparação ponto a ponto com e^x
        linhas = []
        for x in pontos_teste:
            fx = math.exp(x)
            p_sl = avaliar_polinomio(coefs, x)
            p_lg = lagrange(xs, ys, x)
            p_nw = newton(xs, ys, x)
            linhas.append([f"{x:.1f}", f"{fx:.8f}",
                           f"{p_sl:.8f}", f"{abs(fx - p_sl):.2e}",
                           f"{p_lg:.8f}", f"{abs(fx - p_lg):.2e}",
                           f"{p_nw:.8f}", f"{abs(fx - p_nw):.2e}"])
        tabelas_md.append(tabela(
            f"2.2 (n = {n}): e^x x P_{n}(x)",
            ["x", "e^x", "P SL", "erro SL", "P LG", "erro LG", "P NW", "erro NW"],
            linhas))

        # resumo do grau n (erro medido na malha fina)
        p_sl = avaliar_polinomio(coefs, malha)
        erro_sl = np.abs(p_sl - f_malha)
        p_lg = lagrange(xs, ys, malha)
        erro_lg = np.abs(p_lg - f_malha)
        p_nw = newton(xs, ys, malha)
        erro_nw = np.abs(p_nw - f_malha)
        # toda derivada de e^x é e^x, cujo máximo em [a, b] é e^b
        cota = math.exp(b) / math.factorial(n + 1) * np.max(np.abs(produto_nodal(xs, malha)))
        resumo.append([str(n), f"{(b - a) / n:.4f}",
                       f"{erro_sl.max():.3e}", f"{erro_lg.max():.3e}", f"{erro_nw.max():.3e}",
                       f"{np.max(erro_sl / f_malha):.3e}", f"{np.max(erro_lg / f_malha):.3e}",
                       f"{np.max(erro_nw / f_malha):.3e}",
                       f"{cota:.3e}"])

    tabelas_md.append(tabela(
        f"2.2 Resumo: erro máximo em [{_num(a)}, {_num(b)}]",
        ["n", "h", "erro máx SL", "erro máx LG", "erro máx NW",
         "erro rel máx SL", "erro rel máx LG", "erro rel máx NW", "cota teórica"],
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

def questao_2_4_b(pontos, ano_alvo, ppm_obs) -> list[str]:
    """
    Item 2.b: estima o CO2 em ano_alvo por interpolação de 2º e 3º graus e compara com o observado.

    Entrada: pontos -> lista de (xs, ys), um por polinômio (grau = len(xs) - 1);
             ano_alvo -> ano a estimar; ppm_obs -> valor observado nesse ano.
    Saída:   lista com a tabela em Markdown (grau, nós, P_n(ano_alvo), erro absoluto e relativo).
    """
    linhas = []
    for xs, ys in pontos:
        coefs = interpolacao_sist_linear(xs, ys)        # acha o polinômio
        p = avaliar_polinomio(coefs, ano_alvo)          # aplica no ano pedido
        erro = abs(p - ppm_obs)
        linhas.append([str(len(xs) - 1), ", ".join(map(str, xs)),
                       f"{p:.4f}", f"{erro:.4f}", f"{100 * erro / ppm_obs:.2f}"])
    return [tabela(f"2.4.b: estimativa para {ano_alvo} x observado ({ppm_obs} ppm)",
                   ["grau", "nós", f"P_n({ano_alvo})", "erro abs", "erro rel (%)"], linhas)]

def _bisseccao(f, a, b, tol=1e-10, max_iter=100):
    """
    Entrada: f -> função contínua com f(a) e f(b) de sinais opostos;
             [a, b] -> intervalo da busca; tol -> tolerância; max_iter -> nº máximo de iterações.
    Saída:   aproximação da raiz de f em [a, b] pelo método da bissecção.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) e f(b) precisam ter sinais opostos")
    for _ in range(max_iter):
        m = (a + b) / 2
        fm = f(m)
        if fm == 0 or (b - a) / 2 < tol:
            return m
        if fa * fm < 0:
            b = m
        else:
            a, fa = m, fm
    return (a + b) / 2

def questao_2_4_c(xs, ys, ppm_alvo, a, b) -> list[str]:
    """
    Item 2.c: ano em que o CO2 atinge ppm_alvo, com um polinômio de 2º grau.

    Entrada: xs, ys -> 3 pontos (ano, ppm); ppm_alvo -> concentração procurada;
             [a, b] -> intervalo de anos que contém a solução (P_2(a) e P_2(b) de lados opostos de ppm_alvo).
    Saída:   lista com a tabela em Markdown (ano estimado e P_2 nesse ano, para conferência).
    """
    coefs = interpolacao_sist_linear(xs, ys)                       # acha o polinômio
    ano = _bisseccao(lambda x: avaliar_polinomio(coefs, x) - ppm_alvo, a, b)   # resolve P_2(x) = ppm_alvo
    return [tabela(f"2.4.c: ano em que P_2 = {ppm_alvo} ppm",
                   ["ano estimado", "P_2(ano)"],
                   [[f"{ano:.2f}", f"{avaliar_polinomio(coefs, ano):.4f}"]])]

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
