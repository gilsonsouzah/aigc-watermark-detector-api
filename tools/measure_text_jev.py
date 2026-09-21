"""Mede o Jev avaliando TEXTO extraído: ele separa PDF gerado por IA de PDF real?

O ponto do teste: com 1 positivo e 80 negativos, o que dá para afirmar com força é o lado
NEGATIVO 每 se o Jev acusar muitos PDFs reais de terem texto de IA, o sinal não presta.
Só se ele ficar quieto nos reais e alto no gerado é que vale virar camada.

Uso:  python -m tools.measure_text_jev [--n-reais 30]
"""

import argparse
from pathlib import Path

from aiwd.config import Config
from aiwd.gateway import Gateway, boolean
from tools.measure_text import extrair

INSTRUCAO = (
    "O texto abaixo foi escrito por um modelo de linguagem (IA) ou por uma pessoa? "
    "Considere estilo, uniformidade de ritmo, escolhas de pontuação, estrutura de lista e "
    "marcadores típicos de texto gerado. Texto de contrato, apólice, nota fiscal ou extrato "
    "formal costuma ser escrito por pessoa ou por sistema de template — não acuse só por ser "
    "formal. Se o texto for curto demais ou genérico demais para dizer, diga que não sabe."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", default=str(Path(__file__).resolve().parents[1] / "bench" / "pdfs"))
    ap.add_argument("--n-reais", type=int, default=30)
    ap.add_argument("--min-palavras", type=int, default=80)
    ap.add_argument("--max-chars", type=int, default=12000)
    a = ap.parse_args()

    cfg = Config.from_env()
    juiz = Gateway(cfg)
    positivos, negativos = [], []
    for f in sorted(Path(a.pdfs).glob("*.pdf")):
        t = extrair(f)
        if len(t.split()) < a.min_palavras:
            continue
        (positivos if f.name.startswith("IA-") else negativos).append((f.name, t))
    negativos = negativos[:a.n_reais]

    def avaliar(nome: str, texto: str) -> float | None:
        try:
            r = juiz.evaluate({"texto": texto[:a.max_chars]}, {"texto_de_ia": boolean(INSTRUCAO)})
            return r.get("texto_de_ia", {}).get("probability")
        except Exception as e:
            print(f"  erro em {nome}: {type(e).__name__}: {e}")
            return None

    print(f"{len(positivos)} positivo(s), {len(negativos)} reais\n")
    for nome, t in positivos:
        p = avaliar(nome, t)
        print(f"  IA  {nome[:40]:<42} p={p}")
    reais = [(nome, avaliar(nome, t)) for nome, t in negativos]
    validos = [p for _, p in reais if p is not None]
    acusados = [n for n, p in reais if p is not None and p >= 0.5]
    if validos:
        import statistics
        print(f"\nreais: n={len(validos)} mediana={statistics.median(validos):.3f} "
              f"max={max(validos):.3f} | acusados (>=0.5): {len(acusados)}/{len(validos)}")
        for n in acusados[:8]:
            print(f"   acusado: {n[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
