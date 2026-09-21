"""Orquestração: arquivo entra, resultado hierárquico sai.

RESULTADO -> N QUESTÕES -> N CAMADAS (cada camada com seu score dentro da questão).
"""

import hashlib
import time

from .aggregate import confidence_for, resolve, score_for, strength_label, verdict_for
from .calibrate import apply as aplicar_calibracao
from .calibrate import load as carregar_calibracao
from .config import Config
from .describe import descrever
from .files import MAX_IMAGES, detect_kind, extract_images
from .judge import ask_jev, model_findings
from .layers import community_probe, document, forensics, image_model, ocr, provenance, synthid, text

FORMATO_NAO_SUPORTADO = (
    "formato não suportado: só imagem (JPEG/PNG/WebP/TIFF/…) e container com imagem dentro "
    "(PDF, DOCX, PPTX, XLSX, ZIP)"
)


def detect(data: bytes, filename: str, config: Config | None = None) -> dict:
    config = config or Config.from_env()
    started = time.time()
    warnings: list[str] = []

    kind = detect_kind(data, filename)
    if kind == "unsupported":
        return _unsupported(data, filename, started)

    images = extract_images(data, kind, filename)
    if kind != "image" and not images:
        warnings.append("nenhuma imagem dentro do arquivo: só a procedência do container foi avaliada")

    prov = provenance.analyze(images[0][1] if kind == "image" else data, _suffix(filename, data), kind, filename)
    if kind != "image":
        prov["notes"].append(f"arquivo container ({kind}), {len(images)} imagem(ns) embutida(s)")
    ocr_imagem = ocr.analyze(images[0][1], kind) if images else {"signals": {}, "findings": [], "status": "skipped", "notes": []}
    layers = [prov,
              document.analyze(data, kind, filename, ocr_imagem["signals"]),
              text.analyze(data, kind, filename)]
    imagem = images[0][1] if images else data
    layers.append(synthid.analyze(imagem, kind, config, prov["signals"]))
    if images:
        layers.append(ocr_imagem)
        layers.append(image_model.analyze(imagem, kind, _regime_de(layers)))
        layers.append(community_probe.analyze(imagem, kind))

    per_image = []
    for name, blob in images:
        f = forensics.analyze(blob, _suffix(name, blob))
        per_image.append({"image": name, "size": f["signals"].get("size"), "findings": f["findings"]})
    if per_image:
        layers.append(_merge_forensics(per_image, images))
        if len(images) >= MAX_IMAGES:
            warnings.append(f"só as {MAX_IMAGES} primeiras imagens do container foram analisadas")

    descricao = descrever(imagem) if images else {"status": "skipped", "descricao": None}
    if descricao["status"] == "error":
        warnings.append(f"descritor indisponível ({descricao.get('erro')}); Jev recebe só as features")

    evidence = {"file": {"name": filename, "bytes": len(data), "kind": kind,
                         "sha256": hashlib.sha256(data).hexdigest()[:16]},
                "layers": layers, "per_image": per_image,
                "descricao": descricao.get("descricao")}
    answers, warns = ask_jev(evidence, config)
    warnings += warns

    questions = resolve(layers, model_findings(answers) if answers else [])
    warnings += _avisos_de_fragilidade(questions)
    bruto = score_for(questions, "is_ai_generated")
    confidence = confidence_for(questions, "is_ai_generated")
    generator = _generator(questions)

    # calibração só faz sentido sobre score estatístico: evidência determinística já é a resposta
    determinista = any(l["reliability"] == "deterministic"
                       for q in questions if q["id"] == "is_ai_generated" for l in q["layers"])
    regime = next((l["signals"].get("regime") for l in layers if l["name"] == "forensics"), None)
    if determinista:
        score, calib_info = bruto, {"calibrated": False, "reason": "evidência determinística não é recalibrada"}
    else:
        score, calib_info = aplicar_calibracao(bruto, carregar_calibracao(), regime)
    if not calib_info["calibrated"] and not determinista:
        warnings.append(f"score não calibrado: {calib_info['reason']}")

    return {
        "verdict": verdict_for(score, confidence),
        "ai_probability": score,
        "ai_probability_raw": bruto,
        "calibration": calib_info,
        "confidence": round(confidence, 3),
        "confidence_label": strength_label(confidence),
        "generator_guess": generator,
        "kind": kind,
        "filename": filename,
        "sha256": evidence["file"]["sha256"],
        "bytes": len(data),
        "images_analyzed": len(images),
        "questions": questions,
        # as evidências BRUTAS de cada camada (textos lidos pelo OCR, dispersão de blocos,
        # estilometria, espectro). Sem isso o cliente só vê o veredito agregado e não consegue
        # auditar por que uma camada opinou o que opinou.
        "layers": [{"name": l["name"], "status": l["status"], "notes": l["notes"],
                    "signals": l["signals"]} for l in layers],
        "per_image": per_image,
        "description": descricao.get("descricao"),
        "describer": {"status": descricao["status"], "endpoint": descricao.get("endpoint")},
        "judge": {"model": config.judge_model if answers else None, "used": bool(answers),
                  "answers": answers},
        "elapsed_ms": int((time.time() - started) * 1000),
        "warnings": warnings,
    }


def _avisos_de_fragilidade(questions: list[dict]) -> list[str]:
    """Aviso quando a acusação se apoia em UMA camada só.

    Medido no caso da foto oficial do Vaticano (arquivo genuíno, copyright e XMP do Photoshop):
    a camada de imagem acusou com 0.97 e foi a única a acusar — o veredito saiu `likely_ai` sobre
    um falso positivo do modelo. A investigação testou e REFUTOU três explicações (denoise+sharpen
    dá no máximo 0.025 em foto real; reescalonar para 4x dá menos ainda; a outra foto papal real
    pontua 0.0002). `processado com IA` (AI Denoise/Super Resolution de editor) tem a MESMA
    assinatura de `gerado por IA` para este modelo, e o veredito não distingue os dois.

    Não uso o XMP como defesa: metadado é trivialmente forjável, e um gerador pode injetar um
    histórico do Photoshop para escapar. O aviso informa sem abrir essa porta.
    """
    q = next((x for x in questions if x["id"] == "is_ai_generated"), None)
    if not q or q["score"] is None or q["score"] < 0.65:
        return []
    # o Jev NÃO conta como corroboração independente: ele é downstream, lê o escore das outras
    # camadas no estado que eu mando — concordar com o modelo não é evidência nova, é o mesmo
    # sinal contado duas vezes. No caso do Vaticano ele deu 0.68 justamente porque viu o 0.97.
    independentes = [l for l in q["layers"]
                     if l["score"] is not None and l["score"] >= 0.65
                     and l["reliability"] != "weak" and l["layer"] != "jev"]
    acusam = independentes
    if len(acusam) == 1:
        return [f"veredito apoiado em uma única camada ({acusam[0]['layer']}): "
                "foto real processada com IA (AI Denoise/Super Resolution de editor de imagem) "
                "produz a mesma assinatura que geração — ver 'Tetos' no README"]
    return []


def _regime_de(layers: list[dict]) -> str | None:
    """Regime detectado pela camada forense (o espectro da própria imagem denuncia a degradação)."""
    for l in layers:
        if l["name"] == "forensics":
            return l["signals"].get("regime")
    return None


def _merge_forensics(per_image: list[dict], images: list[tuple[str, bytes]]) -> dict:
    """Uma finding por questão, vinda da imagem que mais acusou (o resto fica em per_image)."""
    best = max(per_image, key=lambda d: max((f["score"] for f in d["findings"]), default=-1))
    notes = [f"{d['image']}: {f['score']} ({f['reliability']})"
             for d in per_image for f in d["findings"] if f["reliability"] != "weak"]
    findings = [{**f, "evidence": [f"{best['image']}: {e}" for e in f["evidence"]]} for f in best["findings"]]
    return {"name": "forensics", "status": "ok", "findings": findings, "notes": notes,
            "signals": {"images": len(images), "best_image": best["image"]}}


def _generator(questions: list[dict]) -> str | None:
    q = next((q for q in questions if q["id"] == "generator_family"), None)
    if not q or not q["answer"] or q["answer"] in ("unknown", "indeterminado"):
        return None
    return q["answer"]


def _suffix(filename: str, data: bytes) -> str:
    if "." in (filename or ""):
        return "." + filename.rsplit(".", 1)[-1].lower()
    for magic, ext in ((b"\xff\xd8\xff", ".jpg"), (b"\x89PNG", ".png"), (b"%PDF", ".pdf")):
        if data.startswith(magic):
            return ext
    return ""


def _unsupported(data: bytes, filename: str, started: float) -> dict:
    return {
        "verdict": "unsupported", "ai_probability": None, "confidence": 0.0,
        "confidence_label": "nenhuma", "generator_guess": None, "kind": "unsupported",
        "filename": filename, "sha256": hashlib.sha256(data).hexdigest()[:16], "bytes": len(data),
        "images_analyzed": 0, "questions": [], "per_image": [],
        "judge": {"model": None, "used": False, "answers": None},
        "elapsed_ms": int((time.time() - started) * 1000),
        "warnings": [FORMATO_NAO_SUPORTADO],
    }
