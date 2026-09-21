"""Detector neural de imagem gerada por IA, via ONNX (onnxruntime, CPU, sem torch).

Modelo: Community Forensics (CVPR 2025), ViT-S/16 @384, 21.8M params, licença MIT.
ONNX fp32 de 87 MB, mantido FORA do repo (baixe e aponte `AIGC_IMAGE_MODEL_ONNX`).

VALIDADO contra o checkpoint original nesta máquina (`tools/validate_cf_onnx.py`):
  onnx/model.onnx       max|Δ| sigmoide = 7.3e-06   -> ok
  onnx/model_int8.onnx  max|Δ| sigmoide = 4.2e-01   -> REPROVADO, não usar
O int8 publicado não reproduz o checkpoint — mesmo tipo de defeito que a versão
anterior desse ONNX já teve (heads trocados, sem center-crop).

Pré-processamento do model card: resize da borda menor para 440, center crop 384,
normalização CLIP. É o que faz a inferência bater com o oficial.
"""

import io
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

from ..questions import finding

MEDIA = (0.48145466, 0.4578275, 0.40821073)
DESVIO = (0.26862954, 0.26130258, 0.27577711)
SIDE = 384
RESIZE = 440
_MODELO = "Community Forensics ViT-S/16"
# AUC medida nesta bancada (tools/measure_cf.py) -> define o tier na agregação
TIER = "strong"


def preparar(data: bytes) -> np.ndarray | None:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None
    escala = RESIZE / min(img.size)
    img = img.resize((max(SIDE, round(img.width * escala)), max(SIDE, round(img.height * escala))), Image.BICUBIC)
    esq, topo = (img.width - SIDE) // 2, (img.height - SIDE) // 2
    img = img.crop((esq, topo, esq + SIDE, topo + SIDE))
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - np.array(MEDIA, dtype=np.float32)) / np.array(DESVIO, dtype=np.float32)
    return a.transpose(2, 0, 1)[None]


def _calibracao(regime: str | None) -> dict | None:
    """Platt scaling medido (tools/calibrate_model.py). Sem o arquivo, devolve o escore cru."""
    caminho = os.environ.get("AIGC_MODEL_CALIBRATION", str(Path(__file__).resolve().parents[3] / "model_calibration.json"))
    try:
        dados = json.loads(Path(caminho).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return (dados.get("regimes") or {}).get(regime or "limpo")


def _calibrar(p: float, calib: dict | None) -> tuple[float, dict]:
    """Mapeia o escore cru para probabilidade pela tabela de densidade medida (Bayes).

    p=0.5 é o corte de Bayes, então o valor tem significado de probabilidade — o sigmoide cru
    do modelo não tem (medido: a mesma imagem gerada vai de 0.97 para 0.009 só por recompressão).
    """
    if not calib or not calib.get("grade_log10"):
        return p, {"calibrado": False}
    x = math.log10(min(max(p, 1e-8), 1.0))
    grade, tabela = calib["grade_log10"], calib["probabilidade"]
    if x <= grade[0]:
        q = tabela[0]
    elif x >= grade[-1]:
        q = tabela[-1]
    else:                                   # interpolação linear no eixo log10
        i = max(0, min(len(grade) - 2, int((x - grade[0]) / (grade[1] - grade[0]))))
        t = (x - grade[i]) / (grade[i + 1] - grade[i])
        q = tabela[i] + t * (tabela[i + 1] - tabela[i])
    return float(q), {"calibrado": True, "corte_bruto": calib.get("corte"),
                      "ba_medida": calib.get("acuracia_balanceada"),
                      "ba_apos_mapeamento": calib.get("ba_apos_mapeamento")}


def _sessao(caminho: str):
    try:
        import onnxruntime as ort
    except ImportError:
        return None, "onnxruntime não instalado (uv pip install -e '.[onnx]')"
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(caminho, sess_options=opts, providers=["CPUExecutionProvider"]), None


def analyze(data: bytes, kind: str, regime: str | None = None) -> dict:
    caminho = os.environ.get("AIGC_IMAGE_MODEL_ONNX")
    if not caminho or not os.path.exists(caminho):
        return {"name": "image_model", "status": "disabled", "findings": [], "signals": {},
                "notes": ["sem AIGC_IMAGE_MODEL_ONNX: detector neural desligado "
                          "(pesos de 87 MB ficam fora do repo)"]}
    sess, erro = _sessao(caminho)
    if sess is None:
        return {"name": "image_model", "status": "skipped", "findings": [], "signals": {}, "notes": [erro]}

    x = preparar(data)
    if x is None:
        return {"name": "image_model", "status": "skipped", "findings": [], "signals": {},
                "notes": ["não decodifica como imagem"]}
    entrada = sess.get_inputs()[0]
    if isinstance(entrada.shape[2], int) and entrada.shape[2] != x.shape[2]:
        x = preparar(data)
    logits = np.asarray(sess.run(None, {entrada.name: x})[0]).reshape(-1)
    # 1 saída = logit binário (sigmoid); 2+ = softmax e usa a última classe
    if logits.size == 1:
        p = float(1 / (1 + np.exp(-logits[0])))
    else:
        e = np.exp(logits - logits.max())
        p = float((e / e.sum())[-1])

    calibrado, info = _calibrar(p, _calibracao(regime))
    notas = []
    if not info["calibrado"]:
        notas.append("escore do detector neural NÃO calibrado: o sigmoide cru não é probabilidade "
                     "(medido: mesma imagem vai de 0.97 para 0.009 só por recompressão). "
                     "Rode tools/calibrate_model.py para gerar model_calibration.json")
    return {"name": "image_model", "status": "ok",
            "findings": [finding("is_ai_generated", calibrado, TIER,
                                 [f"{_MODELO} (ONNX validado contra o checkpoint oficial): "
                                  f"escore {p:.4f} -> probabilidade calibrada {calibrado:.3f}"
                                  if info["calibrado"] else
                                  f"{_MODELO}: escore cru {p:.4f} (sem calibração medida)"])],
            "signals": {"model": _MODELO, "score_raw": round(p, 6), "probability": round(calibrado, 4),
                        "calibracao": info, "input": list(x.shape[1:]), "license": "MIT"},
            "notes": notas}
