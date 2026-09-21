"""Varre o LIMIAR DE VEREDITO sobre o cache congelado. O corte não pode ser número de gosto.

Os limiares de veredito (0.90/0.65/0.35/0.10) foram escolhidos por intuição — mesma classe de erro
que os limiares de pixel, que eu tive de trocar por percentil medido depois de acusarem 40% das
fotos reais. Aqui o corte é varrido por máximo de Youden sobre amostra rotulada.

Usa bench/jev_cache.json (scores por camada já congelados): sem chamada de API.

Uso:  python -m tools.sweep_threshold [--peso-jev 0.3]
"""

import argparse
import json
from pathlib import Path

from aiwd.questions import LAYER_WEIGHTS, RELIABILITY
from tools.sweep_jev_weight import RAIZ, logit, sigmoide

POSITIVO = {"ai", "likely_ai"}


def score_de(amostra: dict, peso_jev: float) -> float | None:
    """Mesmo cálculo do aggregate, mas devolvendo o score (não o veredito) para poder varrer corte."""
    camadas = list(amostra["camadas"].get("is_ai_generated", []))
    respostas = amostra.get("respostas_do_juiz") or {}
    a = respostas.get("is_ai_generated")
    if isinstance(a, dict) and isinstance(a.get("probability"), (int, float)):
        camadas.append({"layer": "jev", "score": float(a["probability"]), "reliability": "model"})
    determ = [c for c in camadas if c["reliability"] == "deterministic" and c["score"] is not None]
    if determ:
        return max(c["score"] for c in determ)
    if not camadas:
        return None
    total = 0.0
    for c in camadas:
        if c["score"] is None:
            continue
        w = peso_jev if c["layer"] == "jev" else LAYER_WEIGHTS.get(
            (c["layer"], "is_ai_generated"), RELIABILITY.get(c["reliability"], 0.1))
        total += w * logit(c["score"])
    return sigmoide(total)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--peso-jev", type=float, default=0.3)
    ap.add_argument("--cache", default=str(RAIZ / "bench" / "jev_cache.json"))
    a = ap.parse_args()

    dados = json.loads(Path(a.cache).read_text())
    pos = [d for d in dados if d["rotulo"] != "real"]
    neg = [d for d in dados if d["rotulo"] == "real"]
    ps = [(score_de(d, a.peso_jev), d["confianca"]) for d in pos]
    ns = [(score_de(d, a.peso_jev), d["confianca"]) for d in neg]

    print(f"cache: {len(pos)} gerados, {len(neg)} reais | peso do Jev = {a.peso_jev}")
    print(f"score: gerados min={min(s for s,_ in ps if s is not None) if any(s for s,_ in ps) else '--'} "
          f"max={max((s for s,_ in ps if s is not None), default=None)} | "
          f"reais min={min((s for s,_ in ns if s is not None), default=None)} "
          f"max={max((s for s,_ in ns if s is not None), default=None)}")
    print(f"\n{'corte':<8} {'recall':<8} {'falso positivo':<16} {'acurácia balanceada'}")
    melhor = None
    for corte in [i / 100 for i in range(20, 96, 5)]:
        r = sum(1 for s, c in ps if s is not None and s >= corte and c >= 0.2) / len(ps)
        f = sum(1 for s, c in ns if s is not None and s >= corte and c >= 0.2) / len(ns)
        ba = (r + 1 - f) / 2
        marca = ""
        if melhor is None or ba > melhor[1]:
            melhor, marca = (corte, ba), "  <- melhor"
        print(f"{corte:<8.2f} {r:<8.3f} {f:<16.3f} {ba:<.3f}{marca}")
    med = [s for s, _ in ns if s is not None]
    print(f"\nmaior score em arquivo REAL: {max(med) if med else '--'} "
          f"-> qualquer corte acima disso zera o falso positivo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
