"""Combina as camadas DENTRO de cada questão. Nenhuma camada decide sozinha o resultado."""

import math
import os

import numpy as np

from .questions import BY_ID, LAYER_WEIGHTS, QUESTIONS, RELIABILITY


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoide(x: float) -> float:
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, x))))


def effective_weight(f: dict) -> float:
    """Peso da camada na questão: declarado (LAYER_WEIGHTS) ou o padrão do tier de confiabilidade.

    O peso do Jev em `is_ai_generated` é ajustável por `AIGC_WEIGHT_JEV` para poder ser MEDIDO:
    ele é um avaliador text-only que hedgeia para perto de 0.5, e com peso alto ele derruba
    evidência estrutural medida (num PDF com marcador forte, o Jev votou 0.28 e levou o
    resultado de `likely_ai` para `inconclusive`). O valor final vem da varredura medida.
    """
    if f.get("layer") == "jev" and f["question"] == "is_ai_generated":
        override = os.environ.get("AIGC_WEIGHT_JEV")
        if override:
            return float(override)
    return LAYER_WEIGHTS.get((f.get("layer", ""), f["question"]), RELIABILITY.get(f["reliability"], 0.1))


def _combine(findings: list[dict]) -> tuple[float | None, float, str]:
    """-> (score, confiança, reliability dominante). None = todas as camadas se abstiveram."""
    if not findings:
        return None, 0.05, "none"
    determ = [f for f in findings if f["reliability"] == "deterministic"]
    if determ:
        top = max(determ, key=lambda f: f["score"])
        return top["score"], 0.9, "deterministic"
    w = np.array([effective_weight(f) for f in findings])
    s = np.array([f["score"] for f in findings])
    # Duas formas de combinar camadas dentro de uma questão:
    #   mean    -> média ponderada (dilui: três sinais em 0.75 dão 0.75, somar evidência não soma)
    #   logodds -> soma em log-odds com o peso como desconto de confiabilidade (evidência
    #              independente SOMA, que é o que Bayes manda); é o padrão porque foi medido
    #              em 480 imagens de gerador vs 300 fotos reais — ver README
    if os.environ.get("AIGC_COMBINE", "logodds") == "mean":
        score = float((s * w).sum() / w.sum())
    else:
        score = _sigmoide(float((np.array([_logit(x) for x in s]) * w).sum()))
    tiers = {f["reliability"] for f in findings}
    conf = 0.5 if len(findings) >= 2 else (0.35 if "strong" in tiers else 0.22)
    return round(score, 4), conf, min(tiers)


def resolve(layer_results: list[dict], jev_findings: list[dict]) -> list[dict]:
    """Monta as N questões, cada uma com as N camadas que opinaram sobre ela."""
    by_question: dict[str, list[dict]] = {q.id: [] for q in QUESTIONS}
    for layer in layer_results:
        for f in layer["findings"]:
            by_question[f["question"]].append({**f, "layer": layer["name"]})
    for f in jev_findings:
        by_question[f["question"]].append({**f, "layer": "jev"})

    out = []
    for q in QUESTIONS:
        findings = by_question[q.id]
        score, conf, reliability = _combine(findings)
        out.append({
            "id": q.id,
            "question": q.text,
            "kind": q.kind,
            "primary": q.primary,
            "answer": _answer(findings, score, q.kind),
            "score": score,
            "confidence": round(conf, 3),
            "reliability": reliability,
            "layers": [{**f, "weight": round(effective_weight(f), 4)}
                       for f in sorted(findings, key=lambda f: -f["score"])],
        })
    return out


def _answer(findings: list[dict], score: float | None, kind: str) -> str | float | None:
    determ = [f for f in findings if f.get("answer")]
    if determ:
        return max(determ, key=lambda f: f["score"])["answer"]
    if score is None:
        return "indeterminado" if kind != "score" else None
    if kind == "boolean":
        if score >= 0.65:
            return "sim"
        if score <= 0.35:
            return "não"
        return "indeterminado"
    return round(score, 4)


def confidence_for(results: list[dict], question_id: str) -> float:
    q = next((r for r in results if r["id"] == question_id), None)
    return q["confidence"] if q else 0.05


def score_for(results: list[dict], question_id: str) -> float | None:
    q = next((r for r in results if r["id"] == question_id), None)
    return q["score"] if q else None


def strength_label(confidence: float) -> str:
    for limit, label in ((0.2, "nenhuma"), (0.5, "moderada"), (0.75, "boa"), (0.9, "forte")):
        if confidence < limit:
            return label
    return "muito forte"


def verdict_for(score: float | None, confidence: float) -> str:
    if score is None:
        return "inconclusive"
    if confidence < 0.2:
        return "inconclusive"
    if score >= 0.9:
        return "ai"
    if score >= 0.65:
        return "likely_ai"
    if score <= 0.1:
        return "human"
    if score <= 0.35:
        return "likely_human"
    return "inconclusive"


__all__ = ["BY_ID", "confidence_for", "effective_weight", "resolve", "score_for",
           "strength_label", "verdict_for"]
