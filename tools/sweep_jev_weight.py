"""Varre o peso do Jev em `is_ai_generated` sobre o cache congelado. Medição, não opinião.

Recalcula o veredito para cada peso aplicando o mesmo log-odds da API sobre os scores já
gravados em `bench/jev_cache.json` — sem chamada de API.

Uso:  python -m tools.sweep_jev_weight
"""

import json
import math
from pathlib import Path

from aiwd.questions import LAYER_WEIGHTS, RELIABILITY

RAIZ = Path(__file__).resolve().parents[1]
POSITIVO = {"ai", "likely_ai"}


def logit(p):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoide(x):
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, x))))


def recomputar(amostra: dict, peso_jev: float, limiar_conviccao: float = 0.65) -> str:
    """Mesma regra do aggregate: determinístico domina; senão soma em log-odds."""
    por_questao: dict[str, list[dict]] = {}
    for qid, camadas in amostra["camadas"].items():
        for l in camadas:
            por_questao.setdefault(qid, []).append(l)
    respostas = amostra.get("respostas_do_juiz")
    if respostas:
        for qid, a in respostas.items():
            p = a.get("probability") if isinstance(a, dict) else None
            if isinstance(p, (int, float)):
                por_questao.setdefault(qid, []).append({"layer": "jev", "score": float(p),
                                                        "reliability": "model"})
    camadas = por_questao.get("is_ai_generated", [])
    determ = [c for c in camadas if c["reliability"] == "deterministic"]
    if determ:
        p = max(c["score"] for c in determ)
    else:
        total = 0.0
        for c in camadas:
            if c["layer"] == "jev" and c["score"] is not None:
                w = peso_jev
            else:
                w = LAYER_WEIGHTS.get((c["layer"], "is_ai_generated"), RELIABILITY.get(c["reliability"], 0.1))
            if c["score"] is not None:
                total += w * logit(c["score"])
        p = sigmoide(total) if camadas else None
    if p is None:
        return "inconclusive"
    conf = amostra["confianca"]
    if conf < 0.2:
        return "inconclusive"
    return ("ai" if p >= 0.9 else "likely_ai" if p >= limiar_conviccao
            else "human" if p <= 0.1 else "likely_human" if p <= 0.35 else "inconclusive")


def main() -> int:
    dados = json.loads((RAIZ / "bench" / "jev_cache.json").read_text())
    com_juiz = [d for d in dados if d.get("respostas_do_juiz")]
    sem_juiz = [d for d in dados if not d.get("respostas_do_juiz")]
    print(f"cache: {len(dados)} arquivos, {len(com_juiz)} com resposta do juiz "
          f"({len(sem_juiz)} sem — rate limit na hora de congelar)\n")
    print(f"{'peso do Jev':<12} {'recall':<8} {'falso positivo':<16} {'acurácia balanceada':<20} {'abstenção'}")
    melhor = None
    for peso in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6):
        pos = [d for d in dados if d["rotulo"] != "real"]
        neg = [d for d in dados if d["rotulo"] == "real"]
        vpos = [recomputar(d, peso) for d in pos]
        vneg = [recomputar(d, peso) for d in neg]
        recall = sum(1 for v in vpos if v in POSITIVO) / len(vpos)
        fp = sum(1 for v in vneg if v in POSITIVO) / len(vneg)
        absten = sum(1 for v in vpos + vneg if v == "inconclusive") / (len(vpos) + len(vneg))
        ba = (recall + 1 - fp) / 2
        marca = ""
        if melhor is None or ba > melhor[1]:
            melhor, marca = (peso, ba), "  <- melhor"
        print(f"{peso:<12} {recall:<8.3f} {fp:<16.3f} {ba:<20.3f} {absten:.3f}{marca}")
    print(f"\nmelhor peso medido: {melhor[0]} (acurácia balanceada {melhor[1]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
