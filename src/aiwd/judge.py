"""Camada de arbitragem — Jev (typesafe-ai/jev) via Vercel AI Gateway.

Único modelo do projeto. Jev é modelo de AVALIAÇÃO, não de visão: recebe as evidências
já extraídas como estado textual e responde cada questão com probabilidade. Sem chave,
sem rede ou com erro do gateway, nenhuma finding de modelo é gerada e o resultado sai
só das camadas determinísticas — a API nunca falha por causa do modelo.
"""

from .config import Config
from .gateway import Gateway, GatewayUnavailable, boolean, choice, score
from .questions import QUESTIONS, finding

GENERATOR_CHOICES = {
    "openai": "OpenAI (DALL-E / GPT Image / Sora)",
    "midjourney": "Midjourney",
    "stable_diffusion": "Stable Diffusion / SDXL / ComfyUI / A1111",
    "firefly": "Adobe Firefly",
    "google": "Google (Imagen / Gemini / SynthID)",
    "microsoft": "Microsoft (Designer / Bing Image Creator)",
    "flux": "FLUX (Black Forest Labs)",
    "xai": "Grok / xAI",
    "other_generator": "outro gerador",
    "real_capture": "câmera/software real, sem sinal de IA",
    "unknown": "não dá para dizer",
}
INSTRUCTIONS = {
    "is_ai_generated": "O conteúdo (ou as imagens dentro do arquivo) foi gerado por IA generativa? "
                       "Marcador explícito de gerador em metadado, nome de arquivo ou Content Credentials é "
                       "assinatura determinística, não indício estatístico: se houver um, a resposta é sim.",
    "has_provenance_marker": "O arquivo carrega Content Credentials/C2PA verificável, ou só metadado comum?",
    "watermark_declared": "O arquivo declara watermark ou rótulo de IA (soft binding C2PA, rótulo AIGC/TC260)?",
    "screenshot_or_reencoded": "É provável que a imagem seja screenshot ou tenha sido reencodada, destruindo "
                               "metadados e watermark no caminho?",
    "is_document": "O conteúdo é um documento (página de texto, print de documento, scan, PDF) em vez de "
                   "fotografia ou ilustração? Estrutura de página com fundo claro e faixas de texto indica sim.",
    "locally_manipulated": "Há indício de alteração LOCAL (inpainting, remoção, troca de um pedaço, valor ou "
                           "texto) em vez de geração da imagem inteira? Pergunta separada de 'gerado por IA': "
                           "um documento real com um número trocado é alterado e não é gerado.",
}
STRENGTH = ["nenhuma", "fraca", "moderada", "forte", "muito forte"]


def build_questions() -> dict:
    payload = {}
    faltando = [q.id for q in QUESTIONS
                if q.id not in INSTRUCTIONS and q.id not in ("generator_family", "evidence_strength")]
    if faltando:
        raise ValueError(f"questão sem instrução para o juiz: {faltando} — adicione em INSTRUCTIONS")
    for q in QUESTIONS:
        if q.id == "generator_family":
            payload[q.id] = choice("Qual família de gerador é mais provável, dado os marcadores encontrados?",
                                   GENERATOR_CHOICES)
        elif q.id == "evidence_strength":
            payload[q.id] = score("Quão forte é a evidência reunida, no total? (marque o nível)", STRENGTH[:4])
        else:
            payload[q.id] = boolean(INSTRUCTIONS[q.id])
    return payload


def ask_jev(evidence: dict, config: Config) -> tuple[dict | None, list[str]]:
    if not config.key:
        return None, ["Jev não configurado (VERCEL_AI_GATEWAY_KEY ausente): resultado só das camadas locais"]
    try:
        answers = Gateway(config).evaluate(_state(evidence), build_questions())
    except GatewayUnavailable as e:
        return None, [f"Jev indisponível ({e}): resultado só das camadas locais"]
    return answers, []


def model_findings(answers: dict) -> list[dict]:
    out: list[dict] = []
    for q in QUESTIONS:
        a = answers.get(q.id) or {}
        if q.id == "generator_family":
            ch = a.get("choice")
            if ch:
                p = (a.get("probabilities") or {}).get(ch)
                out.append(finding(q.id, float(p) if isinstance(p, (int, float)) else 0.6, "model",
                                   [f"avaliador Jev: {GENERATOR_CHOICES.get(ch, ch)}"
                                    + (f" (p={p})" if p is not None else "")], answer=ch))
            continue
        if q.id == "evidence_strength":
            if isinstance(a.get("score"), (int, float)):
                idx = float(a["score"])
                out.append(finding(q.id, min(0.95, (idx + 0.5) / len(STRENGTH)), "model",
                                   [f"avaliador Jev: força {STRENGTH[max(0, min(int(round(idx)), len(STRENGTH) - 1))]} ({idx})"]))
            continue
        p = a.get("probability")
        if isinstance(p, (int, float)):
            out.append(finding(q.id, float(p), "model", [f"avaliador Jev: P={p:.2f}"]))
    return out


def _state(evidence: dict) -> dict:
    """Estado enxuto: Jev tem 32k de contexto, mas evidência boa é evidência curta."""
    estado = {
        "arquivo": evidence.get("file"),
        "camadas": [
            {"nome": l["name"], "status": l["status"],
             "achados": [{"questao": f["question"], "score": f["score"], "confiabilidade": f["reliability"],
                          "evidencia": f["evidence"]} for f in l["findings"]],
             "notas": l["notes"][:6],
             "sinais": {k: v for k, v in l["signals"].items()
                        if k not in ("metadata_keys", "spectrum_db", "blockiness", "jpeg_quant_tables")}}
            for l in evidence.get("layers", [])
        ],
        "por_imagem": evidence.get("per_image"),
    }
    # descrição forense entra como evidência textual: é o que o Jev consegue usar de fato,
    # já que ele não vê a imagem (verificado: não decodifica nem base64)
    if evidence.get("descricao"):
        estado["descricao_forense_da_imagem"] = evidence["descricao"]
    return estado
