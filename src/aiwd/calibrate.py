"""Calibração do score bruto em probabilidade, com τ medido — não chutado.

Motivo (medido, não teórico): em detector de imagem IA o AUC sobrevive à degradação, mas a
acurácia em limiar fixo colapsa. Corvi et al. treinado em ProGAN: StyleGAN2 vai de 99.9 AUC
incondicional para 94.8 sob resize+JPEG, enquanto a acurácia no limiar fixo cai de 98.1 para
63.3. Ou seja: o problema não é o ranking, é o ponto de corte.

Implementação: deslocamento de logit (score_calibrado = logit(p) + τ), com τ escolhido por
máximo de Youden (J = sensibilidade + especificidade − 1) sobre a amostra rotulada — a forma
supervisionada do AIGI-Det-Calib (arXiv 2602.01973), sem KDE: com algumas centenas de amostras
a CDF empírica em grade é mais estável que KDE, e não precisa de scipy.

`calibration.json` é gerado por tools/fit_calibration.py a partir de medição em dataset
rotulado. Sem o arquivo, a API responde o score NÃO calibrado e diz isso em `warnings`.
"""

import json
import math
import os
from pathlib import Path

GRID = [i / 200 for i in range(200)]


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def load(path: str | os.PathLike | None = None) -> dict | None:
    caminho = Path(path or os.environ.get("AIGC_CALIBRATION", "calibration.json"))
    if not caminho.is_file():
        return None
    try:
        dados = json.loads(caminho.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return dados if isinstance(dados, dict) and "tau" in dados else None


def apply(raw: float | None, calib: dict | None, regime: str | None = None) -> tuple[float | None, dict]:
    """Aplica o deslocamento de logit. Sem calibração, devolve o score cru com aviso."""
    if raw is None:
        return None, {"calibrated": False, "reason": "score ausente"}
    if not calib:
        return raw, {"calibrated": False, "reason": "sem calibration.json (rode tools/fit_calibration.py)"}
    por_regime = (calib.get("regimes") or {}).get(regime or calib.get("default_regime") or "")
    tau = (por_regime or {}).get("tau", calib.get("tau"))
    if tau is None:
        return raw, {"calibrated": False, "reason": f"sem τ para o regime {regime!r}"}
    p = sigmoid(logit(raw) + float(tau))
    return round(p, 4), {"calibrated": True, "tau": float(tau), "regime": regime,
                         "n_pos": (por_regime or {}).get("n_pos"), "n_neg": (por_regime or {}).get("n_neg"),
                         "balanced_accuracy": (por_regime or {}).get("balanced_accuracy")}


def fit_youden(pos: list[float], neg: list[float]) -> dict:
    """τ que maximiza sensibilidade + especificidade − 1. Empírico, sem suposto de distribuição."""
    p = [x for x in pos if x is not None]
    n = [x for x in neg if x is not None]
    if not p or not n:
        return {"tau": 0.0, "balanced_accuracy": None, "n_pos": len(p), "n_neg": len(n)}
    melhor = {"tau": 0.0, "balanced_accuracy": 0.0}
    for tau in GRID:
        sens = sum(1 for x in p if sigmoid(logit(x) + tau) >= 0.5) / len(p)
        espec = sum(1 for x in n if sigmoid(logit(x) + tau) < 0.5) / len(n)
        ba = (sens + espec) / 2
        if ba > melhor["balanced_accuracy"]:
            melhor = {"tau": round(tau, 4), "balanced_accuracy": round(ba, 4),
                      "sensibilidade": round(sens, 4), "especificidade": round(espec, 4)}
    return {**melhor, "n_pos": len(p), "n_neg": len(n)}
