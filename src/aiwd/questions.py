"""As perguntas que a API responde, e o que cada camada pode dizer sobre cada uma.

Hierarquia do resultado: RESULTADO -> N QUESTOES -> N CAMADAS (cada camada com seu score
dentro da questão). Nenhuma camada fala sozinha pelo resultado final.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    id: str
    kind: str          # boolean | choice | score
    text: str
    primary: bool = False


QUESTIONS: tuple[Question, ...] = (
    Question("is_ai_generated", "boolean",
             "O conteúdo foi gerado por IA generativa?", primary=True),
    Question("generator_family", "choice",
             "Qual família de gerador é mais provável?"),
    Question("has_provenance_marker", "boolean",
             "O arquivo carrega marcação de procedência verificável (Content Credentials/C2PA)?"),
    Question("watermark_declared", "boolean",
             "O arquivo declara watermark ou rótulo de IA (soft binding, AIGC/TC260)?"),
    Question("screenshot_or_reencoded", "boolean",
             "Os metadados foram destruídos por screenshot, reencode ou upload?"),
    Question("is_document", "boolean",
             "O conteúdo é um documento (página/print/scan) e não uma imagem/foto?"),
    Question("locally_manipulated", "boolean",
             "Há indício de alteração LOCAL por IA (inpainting, patch) e não geração da imagem toda?"),
    Question("evidence_strength", "score",
             "Quão forte é a evidência reunida, no total?"),
)
BY_ID = {q.id: q for q in QUESTIONS}

# peso de cada camada por questão: quanto ela sabe sobre AQUELA pergunta
LAYER_WEIGHTS = {
    ("provenance", "is_ai_generated"): 1.0,
    ("provenance", "generator_family"): 1.0,
    ("provenance", "has_provenance_marker"): 1.0,
    ("provenance", "watermark_declared"): 1.0,
    ("provenance", "screenshot_or_reencoded"): 0.5,
    ("forensics", "is_ai_generated"): 0.35,
    ("forensics", "screenshot_or_reencoded"): 0.7,
    ("forensics", "evidence_strength"): 0.5,
    # MEDIDO (tools/sweep_jev_weight.py sobre cache congelado de 66 arquivos rotulados):
    #   peso 0.0  -> recall 0.722 (sem o Jev o detector perde caso)
    #   peso 0.05-0.2 -> acurácia balanceada 1.000, abstenção 0.44 -> 0.35
    #   peso 0.3-0.6  -> acurácia balanceada 1.000, abstenção 0.076
    # 0.3 escolhido: mesmo resultado do peso alto com o Jev pesando menos na decisão.
    ("jev", "is_ai_generated"): 0.3,
    ("jev", "generator_family"): 0.7,
    ("jev", "has_provenance_marker"): 0.3,
    ("jev", "watermark_declared"): 0.3,
    ("jev", "screenshot_or_reencoded"): 0.6,
    ("jev", "evidence_strength"): 0.8,
    ("synthid", "watermark_declared"): 0.5,
    ("synthid", "is_ai_generated"): 0.3,
    # medido AUC 0.31-0.59 (tools/measure_probe.py): não discrimina — peso simbólico
    ("image_model", "is_ai_generated"): 0.55,
    ("document", "is_document"): 1.0,
    ("document", "screenshot_or_reencoded"): 0.6,
    ("forensics", "locally_manipulated"): 0.6,
    ("forensics", "is_document"): 0.4,
    ("ocr", "watermark_declared"): 0.8,
    ("ocr", "is_ai_generated"): 0.5,
    ("community_probe", "is_ai_generated"): 0.03,
}

# deterministic domina: marcador explícito não é rebaixado por inferência estatística ou modelo
# tier -> peso padrão quando a camada não tem peso declarado para a questão.
# "experimental" é o piso: entra no resultado, mas nunca decide (marcado como não calibrado).
RELIABILITY = {"deterministic": 1.0, "strong": 0.4, "weak": 0.12, "model": 0.3, "experimental": 0.03}


def finding(question: str, score: float, reliability: str, evidence: list[str] | None = None,
            answer: str | None = None) -> dict:
    if question not in BY_ID:
        raise KeyError(f"questão desconhecida: {question}")
    return {"question": question, "score": round(float(score), 4), "reliability": reliability,
            "evidence": evidence or [], "answer": answer}
