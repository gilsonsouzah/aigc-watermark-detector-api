"""Mede se características de texto separam PDF gerado por IA de PDF real.

Ordem correta: medir primeiro, virar finding depois. Nada aqui entra no veredito antes de
mostrar número — foi assim que os limiares de pixel e o escore do detector neural foram feitos.

Uso:  python -m tools.measure_text [--n 60]
"""

import argparse
import re
import statistics
from pathlib import Path

from pypdf import PdfReader


def extrair(caminho: Path) -> str:
    try:
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(caminho)).pages)
    except Exception:
        return ""


def caracteristicas(t: str) -> dict:
    frases = [f.strip() for f in re.split(r"[.!?]+", t) if len(f.strip()) > 15]
    palavras = re.findall(r"\b[\wÀ-ÿ']+\b", t.lower())
    comp_frases = [len(f.split()) for f in frases]
    tipos = len(set(palavras)) / len(palavras) if palavras else None
    linhas = [l.strip() for l in t.splitlines() if l.strip()]
    return {
        "n_palavras": len(palavras),
        "media_palavras_frase": statistics.mean(comp_frases) if comp_frases else None,
        "desvio_palavras_frase": statistics.pstdev(comp_frases) if len(comp_frases) > 1 else None,
        "razao_tipo_token": tipos,
        "fracao_bullets": sum(1 for l in linhas if l[:2] in ("- ", "* ", "• ")) / len(linhas) if linhas else None,
        "travessao_curto": t.count("—") / len(palavras) * 1000 if palavras else None,
        "curvas_aspas": (t.count("“") + t.count("”")) / len(palavras) * 1000 if palavras else None,
        "linhas_por_paragrafo": len(linhas) / max(1, len([l for l in t.split("\n\n") if l.strip()])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", default=str(Path(__file__).resolve().parents[1] / "bench" / "pdfs"))
    ap.add_argument("--min-palavras", type=int, default=60)
    a = ap.parse_args()

    positivos, negativos = [], []
    for f in sorted(Path(a.pdfs).glob("*.pdf")):
        t = extrair(f)
        if len(t.split()) < a.min_palavras:
            continue
        c = caracteristicas(t)
        (positivos if f.name.startswith("IA-") else negativos).append((f.name, c))

    print(f"{len(positivos)} PDF(s) gerado(s) por IA, {len(negativos)} reais com texto extraível "
          f"(>= {a.min_palavras} palavras)\n")
    if not positivos:
        print("sem positivo: nomeie o arquivo com prefixo IA-")
        return 1
    chaves = list(positivos[0][1])
    print(f"{'característica':<26} {'IA':>10} {'reais mediana':>14} {'p10':>8} {'p90':>8}")
    for k in chaves:
        vp = [c[k] for _, c in positivos if c.get(k) is not None]
        vn = sorted(c[k] for _, c in negativos if c.get(k) is not None)
        if not vp or len(vn) < 5:
            continue
        p10 = vn[max(0, int(len(vn) * 0.1))]
        p90 = vn[min(len(vn) - 1, int(len(vn) * 0.9))]
        print(f"{k:<26} {statistics.mean(vp):>10.3f} {statistics.median(vn):>14.3f} {p10:>8.3f} {p90:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
