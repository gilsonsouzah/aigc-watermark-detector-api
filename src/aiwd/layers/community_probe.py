"""Camada de sonda neural comunitária (ONNX), peso mínimo e rótulo explícito.

MEDIDO NESTA BANCADA (480 imagens de 6 geradores vs 200 fotos COCO, 3 regimes,
`python -m tools.measure_probe`) — e o resultado condena a sonda como detector de SynthID:

    modelo                 AUC em gerador COM SynthID   AUC em gerador SEM SynthID
    efficientnet_b0                     0.383                        0.588
    resnet18                            0.309                        0.480
    resnet34                            0.370                        0.520

Fosse detector de SynthID, o AUC seria ALTO na coluna da esquerda. É o contrário: em imagem
com SynthID (imagen-4 0.30, Nano-Banana 0.46) o escore fica ABAIXO de 0.5, isto é, a sonda
acha essas imagens menos suspeitas que foto real; e pontua mais alto justamente em gerador
sem watermark (Seedream-4 0.77, GPT-Image-1.5 0.70). Não é detector de watermark nem
classificador útil de IA nesta amostra — no máximo reconhece o "jeitão" de alguns geradores.

Por isso a camada fica DESLIGADA por padrão, entra com tier `experimental` (peso 0.03) quando
ligada, e o texto da finding diz o que a medição mostrou. Os pesos vêm de
`newideas99/gpt-image-synthid-detector` (ResNet/EfficientNet sem código e sem contrato de
pré-processamento publicado); o ONNX exportado foi validado contra o checkpoint (Δ ≤ 5e-6,
`tools/validate_onnx_probe.py`), então a divergência acima é do modelo, não da conversão.

LICENÇA: os pesos são PolyForm Noncommercial. Este projeto é uma prova de conceito declarada
como não comercial (ver LICENSE), então o uso aqui está coberto — mas é isso que impede uso
comercial da API enquanto esta camada estiver ligada. Por isso os pesos não são redistribuídos
no repositório: aponte `AIGC_PROBE_ONNX` para o ONNX exportado fora do repo (ver
tools/export_synthid_surrogates.py). Sem a variável, a camada fica desligada.
"""

import io
import os
from pathlib import Path

import numpy as np
from PIL import Image

from ..questions import finding

MEDIA = (0.485, 0.456, 0.406)
DESVIO = (0.229, 0.224, 0.225)


def _models() -> list[Path]:
    return [Path(p) for p in (os.environ.get("AIGC_PROBE_ONNX") or "").split(",") if p.strip()]


def _sessao(path: Path):
    try:
        import onnxruntime as ort
    except ImportError:
        return None, "onnxruntime não instalado (uv pip install -e '.[onnx]')"
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"]), None


def _preparar(data: bytes, side: int) -> np.ndarray | None:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None
    img = img.resize((side, side), Image.BICUBIC)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - np.array(MEDIA, dtype=np.float32)) / np.array(DESVIO, dtype=np.float32)
    return a.transpose(2, 0, 1)[None, ...]


def _prob(logits: np.ndarray) -> float:
    """1 saída = logit único (sigmoide); 2+ saídas = softmax e usa a última classe."""
    v = np.asarray(logits).reshape(-1).astype(np.float64)
    if v.size == 1:
        return float(1 / (1 + np.exp(-v[0])))
    e = np.exp(v - v.max())
    return float((e / e.sum())[-1])


def analyze(data: bytes, kind: str) -> dict:
    signals: dict = {"probes": []}
    findings: list[dict] = []
    notes: list[str] = []

    paths = _models()
    if not paths:
        return {"name": "community_probe", "status": "disabled", "findings": [], "signals": signals,
                "notes": ["sem AIGC_PROBE_ONNX configurado (pesos são Noncommercial, ficam fora do repo)"]}

    for path in paths:
        if not path.is_file():
            notes.append(f"modelo ausente: {path}")
            continue
        sess, erro = _sessao(path)
        if sess is None:
            notes.append(erro)
            break
        entrada = sess.get_inputs()[0]
        side = int(entrada.shape[2]) if isinstance(entrada.shape[2], int) else 224
        x = _preparar(data, side)
        if x is None:
            notes.append(f"{path.name}: não decodifica como imagem")
            continue
        if x.shape[2] != side:
            x = _preparar(data, side)
        logits = sess.run(None, {entrada.name: x})[0]
        p = _prob(logits)
        signals["probes"].append({"model": path.name, "side": side, "probability": round(p, 4),
                                  "output_shape": list(np.asarray(logits).shape)})
        findings.append(finding("is_ai_generated", p, "experimental",
                                [f"sonda neural comunitária {path.name}: {p:.3f} — AUC medido nesta bancada "
                                 "0.31-0.59 (abaixo de 0.5 justamente em imagem COM SynthID); não é "
                                 "detector de watermark, entra com peso mínimo"]))

    if signals["probes"] and not os.environ.get("AIGC_PROBE_CALIBRATED"):
        notes.append("sonda comunitária sem calibração medida: contribui com peso mínimo e não decide sozinha")
    if signals["probes"]:
        notes.append("pesos sob PolyForm Noncommercial: uso comercial da API fica impedido com esta camada ligada")
    return {"name": "community_probe", "status": "ok" if signals["probes"] else "skipped",
            "findings": findings, "signals": signals, "notes": notes}
