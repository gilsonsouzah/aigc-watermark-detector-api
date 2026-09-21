"""Camada SynthID (watermark imperceptível do Google) — com peso baixo e explícito.

Situação verificada em 2026-09-21, e ela manda no desenho desta camada:

- NÃO existe detector aberto de SynthID-image que funcione. Todos os candidatos populares
  foram medidos e caem na chance (AUC pareada 0.517 no mais citado, 0/400 em outro,
  214/600 fotos reais aceitas num terceiro). O teto teórico calculado para uma única
  imagem é ~0.75 AUC mesmo com template perfeitamente alinhado.
- O paper oficial (arXiv 2510.09263) NÃO testa screenshot nem recapture: a robustez de
  99,97% é contra crop/resize/rotação/JPEG. Print de tela não está coberto por nenhuma
  afirmação verificada.

Por isso a camada tem três fontes, e só a primeira funciona sem acesso externo:

1. DECLARAÇÃO (`declaration`) — determinística. O C2PA do Google traz ação
   `c2pa.edited` com descrição "Applied imperceptible SynthID watermark.", e a OpenAI
   marca com C2PA + SynthID desde mai/2026. Ler a declaração é honesto e funciona.
2. DETECTOR EXTERNO (`external`) — plugável. Configure `SYNTHID_DETECTOR_URL` (POST com a
   imagem, resposta {"probability": 0..1} ou {"detected": bool}) quando tiver acesso ao
   portal do Google, à GetReal Security, ou a um detector futuro. Sem URL, a fonte fica off.
3. SONDA LOCAL (`probe`) — DESLIGADA por padrão (`SYNTHID_LOCAL_PROBE=1` liga). Só roda se
   houver `SYNTHID_TEMPLATE` (uma imagem sabidamente marcada) e faz correlação espectral
   entre o resíduo da candidata e o do template — o único método com alguma base publicada
   (o "template match" do reverse-SynthID). Sem template ela declara `unavailable` em vez de
   inventar número. Peso mínimo (tier `experimental`), nunca decide sozinha.
"""

import io
import os
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from ..config import Config
from ..questions import finding

SYNTHID_DECL_RE = None  # preenchido abaixo para evitar import circular de regex
MIN_SIDE = 128


def _declaration_texts(kind: str, provenance_signals: dict) -> list[str]:
    c2 = (provenance_signals or {}).get("c2pa") or {}
    out = [d for d in (c2.get("watermark_declarations") or [])]
    if c2.get("claim_generator"):
        out.append(str(c2["claim_generator"]))
    return out


def analyze(data: bytes, kind: str, config: Config, provenance_signals: dict | None = None) -> dict:
    signals: dict = {}
    notes: list[str] = []
    findings: list[dict] = []

    decl = _declaration_texts(kind, provenance_signals or {})
    hit = [d for d in decl if "synthid" in d.lower()]
    signals["declaration"] = hit or None
    if hit:
        findings.append(finding("watermark_declared", 0.95, "deterministic",
                                [f"manifest declara SynthID: {hit[0]}"], answer="sim"))
        findings.append(finding("is_ai_generated", 0.9, "deterministic",
                                ["conteúdo marcado com SynthID é gerado por IA (declarado pelo provedor)"]))
        notes.append("declaração de SynthID no arquivo: prova de origem, não detecção do watermark")

    ext = _external(data)
    signals["external"] = ext
    if ext.get("probability") is not None:
        p = float(ext["probability"])
        findings.append(finding("watermark_declared", p, "strong", [f"detector externo ({ext['endpoint']}): p={p}"],
                                answer="sim" if p >= 0.5 else "não"))

    probe = _probe(data, config)
    signals["probe"] = probe
    if probe.get("score") is not None:
        findings.append(finding("watermark_declared", probe["score"], "experimental",
                                [f"sonda local por template ({probe['template']}): correlação {probe['score']} "
                                 "— NÃO calibrada, não é detector de SynthID"]))
    if probe.get("status") == "unavailable":
        notes.append("sonda local de SynthID indisponível: " + probe["reason"])
    if not findings:
        notes.append(
            "sem detecção local de SynthID: não existe detector aberto que funcione hoje "
            "(candidatos medidos caem na chance); só a declaração no arquivo é verificável")

    return {"name": "synthid", "status": "ok", "findings": findings, "signals": signals, "notes": notes}


def _external(data: bytes) -> dict:
    url = os.environ.get("SYNTHID_DETECTOR_URL")
    if not url:
        return {"status": "not_configured", "endpoint": None, "probability": None}
    try:
        r = httpx.post(url, content=data, timeout=30.0,
                       headers={"content-type": "application/octet-stream"})
        r.raise_for_status()
        body = r.json()
        p = body.get("probability")
        if p is None and "detected" in body:
            p = 1.0 if body["detected"] else 0.0
        return {"status": "ok", "endpoint": url, "probability": p, "raw": body}
    except Exception as e:
        return {"status": "error", "endpoint": url, "probability": None, "error": f"{type(e).__name__}: {e}"[:200]}


def _residual(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    box = sum(p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)) / 9.0
    return a - box


def _whiten(mag: np.ndarray) -> np.ndarray:
    """Divide o espectro pelo próprio perfil radial: remove o envelope 1/f que toda imagem
    natural tem e que faz duas imagens quaisquer correlacionarem ~1 sem isso."""
    h, w = mag.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    bins = np.clip((r / 0.5 * 32).astype(int), 0, 31)
    profile = np.zeros_like(mag)
    for b in range(32):
        m = bins == b
        if m.any():
            profile[m] = np.median(mag[m])
    return mag / (profile + 1e-9)


def _spectrum(data: bytes) -> np.ndarray | None:
    try:
        img = Image.open(io.BytesIO(data)).convert("L")
    except Exception:
        return None
    if min(img.size) < MIN_SIDE:
        return None
    side = min(512, min(img.size))
    img = img.crop(((img.width - side) // 2, (img.height - side) // 2,
                    (img.width + side) // 2, (img.height + side) // 2))
    a = np.asarray(img, dtype=np.float32)
    res = _residual(a)
    win = np.hanning(side)[:, None] * np.hanning(side)[None, :]
    mag = np.abs(np.fft.fftshift(np.fft.fft2(res * win)))
    mag[side // 2 - 2:side // 2 + 3, side // 2 - 2:side // 2 + 3] = 0  # zera DC
    return _whiten(mag / (mag.mean() + 1e-9))


def _probe(data: bytes, config: Config) -> dict:
    if os.environ.get("SYNTHID_LOCAL_PROBE", "").lower() not in ("1", "true", "yes"):
        return {"status": "disabled", "reason": "SYNTHID_LOCAL_PROBE não ligado", "score": None}
    paths = [p for p in (config.synthid_template or "").split(",") if p.strip()]
    paths = [p for p in paths if Path(p).is_file()]
    if not paths:
        return {"status": "unavailable", "reason": "sem SYNTHID_TEMPLATE (imagem sabidamente marcada)",
                "score": None}
    cand = _spectrum(data)
    if cand is None:
        return {"status": "unavailable", "reason": "imagem pequena demais para o espectro", "score": None}
    # template match só faz sentido na MESMA escala de pixel: por isso a lista de templates
    # por escala (o print de tela muda a escala e a sonda precisa do template daquela escala)
    best, sizes = None, []
    for path in paths:
        tmpl = _spectrum(Path(path).read_bytes())
        if tmpl is None:
            continue
        sizes.append(list(tmpl.shape))
        if tmpl.shape != cand.shape:
            continue
        corr = float(np.corrcoef(cand.ravel(), tmpl.ravel())[0, 1])
        if best is None or corr > best[0]:
            best = (corr, Path(path).name)
    if best is None:
        return {"status": "unavailable", "score": None,
                "reason": f"nenhum template na escala da candidata ({cand.shape}); templates: {sizes}"}
    return {"status": "ok", "template": best[1], "score": round(max(0.0, best[0]), 4),
            "raw_correlation": round(best[0], 4), "template_scales": sizes}
