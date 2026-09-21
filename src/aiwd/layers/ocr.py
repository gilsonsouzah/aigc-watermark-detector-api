"""Camada de leitura DENTRO da imagem (OCR): lê rótulo de IA escrito na própria figura.

Por que isso importa: a marcação que a lei europeia está produzindo aparece como TEXTO na imagem —
legenda, faixa de rodapé, marca d'água visível, etiqueta do gerador. É o único sinal que sobrevive
a screenshot, porque vem do pixel e não do metadado.

Escolha medida, não por gosto (`tools/measure_ocr.py`, medido no host): OCR clássico resolve, e um
VLM pequeno NÃO é necessário. `rapidocr` (ONNX, mesmo runtime do serviço) lê "Exemplo de imagem
gerada por uma IA generativa" em texto claro e "Gerado por IA · SynthID" em cinza claro (215,215,215)
— os dois casos que a pesquisa de VLM tratava como difíceis — a 0.2-0.4 s por imagem, com 31.7 MB de
modelo já embutido no pacote. Um VLM <1B teria MMMU 29-34 (aleatório ~25%) e entrada de 384-512 px,
que destrói justamente o sinal de alta frequência.

Peso: reliability `strong`, não `deterministic` — o rótulo pode vir de uma legenda de artigo
FALANDO sobre a imagem (é o caso do meu teste), o que é indício forte mas não é assinatura do
gerador. E `rotulo_de_ia` no texto nem sempre declara geração: "AI" aparece em "airfare".
"""

import io
import re
import time

from ..questions import finding

# Rótulos que declaram geração por IA. Fronteira de palavra obrigatória: "AI" dentro de "chair"
# ou "airfare" não é rótulo.
ROTULOS = (
    r"gerad[oa]s?\s+por\s+ia", r"gerad[oa]s?\s+por\s+intelig[êe]ncia\s+artificial",
    # variações MEDIDAS de erro de OCR: "gerada por uma lA" (I confundido com l) — o caso real
    # que apareceu no teste de mutação, não hipótese
    r"gerad[oa]s?\s+por\s+(?:uma?\s+)?[il1]a\b", r"gerad[oa]s?\s+por\s+[il1]a\b",
    r"imagem\s+gerada", r"criado\s+por\s+ia", r"feito\s+com\s+ia",
    r"\bai[- ]generated\b", r"\bgenerated\s+by\s+ai\b", r"\bmade\s+with\s+ai\b",
    r"\bgenerative\s+ai\b", r"\bsynthetic\s+image\b", r"\bai\s+generated\b",
    r"\bsynthid\b", r"\bdall[\-·.]?e\b", r"\bmidjourney\b", r"\bstable\s+diffusion\b",
    r"\bimagen\b", r"\bfirefly\b", r"\bchatgpt\b", r"\bgpt-image\b", r"\bgrok\b",
    r"\bflux\b", r"\bnano\s+banana\b",
)
ROTULO_RE = re.compile("|".join(ROTULOS), re.IGNORECASE)
MAX_LADO = 1600          # OCR em imagem gigante é lento; reduz para caber
MIN_CONFIANCA = 0.5


def _motor():
    try:
        from rapidocr import RapidOCR
    except ImportError:
        return None, "rapidocr não instalado (uv pip install -e '.[ocr]')"
    # o rapidocr loga cada modelo carregado e cada imagem sem texto a cada inferência — sem
    # silenciar, a API cospe dezenas de linhas de INFO/WARNING por requisição. Ele usa
    # logging.getLogger("RapidOCR"), não loguru.
    try:
        import logging

        logging.getLogger("RapidOCR").setLevel(logging.ERROR)
    except Exception:
        pass
    return RapidOCR(), None


def analyze(data: bytes, kind: str) -> dict:
    if kind != "image":
        return {"name": "ocr", "status": "skipped", "findings": [], "signals": {},
                "notes": ["só imagem: documento de texto tem a camada `text`"]}
    motor, erro = _motor()
    if motor is None:
        return {"name": "ocr", "status": "disabled", "findings": [], "signals": {}, "notes": [erro]}

    from PIL import Image

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return {"name": "ocr", "status": "skipped", "findings": [], "signals": {},
                "notes": ["não decodifica como imagem"]}
    if max(img.size) > MAX_LADO:
        escala = MAX_LADO / max(img.size)
        img = img.resize((int(img.width * escala), int(img.height * escala)), Image.BICUBIC)

    inicio = time.time()
    try:
        resultado = motor(img)
    except Exception as e:
        return {"name": "ocr", "status": "error", "findings": [], "signals": {},
                "notes": [f"OCR falhou: {type(e).__name__}: {e}"[:200]]}
    ms = int((time.time() - inicio) * 1000)

    textos = [t for t in (getattr(resultado, "txts", None) or []) if t and t.strip()]
    confiancas = list(getattr(resultado, "scores", None) or [])
    achados: list[dict] = []
    sinais: dict = {"textos": textos[:20], "n_textos": len(textos), "ms": ms}

    candidatos = [(t, c) for t, c in zip(textos, confiancas + [1.0] * len(textos)) if c >= MIN_CONFIANCA]
    hits = [(t, c) for t, c in candidatos if ROTULO_RE.search(t)]
    if hits:
        sinais["rotulos_de_ia"] = [t for t, _ in hits]
        achados.append(finding("watermark_declared", 0.9, "strong",
                               [f"rótulo de IA escrito na imagem (lido por OCR, confiança {c:.2f}): {t!r}"
                                for t, c in hits[:3]], answer="sim"))
        achados.append(finding("is_ai_generated", 0.88, "strong",
                               ["a própria imagem declara ter sido gerada por IA em texto visível"]))
    return {"name": "ocr", "status": "ok", "findings": achados, "signals": sinais, "notes": []}
