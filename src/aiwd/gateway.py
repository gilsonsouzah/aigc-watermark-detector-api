import time

import httpx

from .config import Config

TENTATIVAS = 3
ESPERA_BASE = 1.5      # segundos; dobra a cada tentativa
STATUS_TRANSITORIO = (429, 500, 502, 503, 504)


class GatewayUnavailable(RuntimeError):
    pass


class Gateway:
    """Cliente do Vercel AI Gateway. Único modelo usado: Jev (typesafe-ai/jev),
    um modelo de avaliação — recebe um `state` e perguntas tipadas, devolve
    probabilidades. Não gera texto nem vê imagem."""

    def __init__(self, config: Config):
        self.config = config

    @property
    def configured(self) -> bool:
        return bool(self.config.key)

    def evaluate(self, state: dict, questions: dict) -> dict:
        if not self.configured:
            raise GatewayUnavailable("VERCEL_AI_GATEWAY_KEY não configurada")
        payload = {"model": self.config.judge_model, "state": state, "questions": questions}
        # 429 do provedor é transitório ("upstream provider high demand") e já foi observado em
        # uso real: sem retry, a camada de modelo some e o resultado cai para o das camadas locais
        ultimo = None
        for tentativa in range(TENTATIVAS):
            try:
                with httpx.Client(timeout=self.config.timeout) as client:
                    r = client.post(
                        f"{self.config.gateway_url}/evaluate",
                        json=payload,
                        headers={"Authorization": f"Bearer {self.config.key}"},
                    )
            except httpx.HTTPError as e:
                ultimo = f"falha de rede: {e}"
            else:
                if r.status_code == 200:
                    break
                ultimo = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code not in STATUS_TRANSITORIO:
                    raise GatewayUnavailable(ultimo)
            if tentativa < TENTATIVAS - 1:
                time.sleep(ESPERA_BASE * (2 ** tentativa))
        else:
            raise GatewayUnavailable(f"{ultimo} (após {TENTATIVAS} tentativas)")
        if r.status_code != 200:
            raise GatewayUnavailable(f"{ultimo} (após {TENTATIVAS} tentativas)")
        answers = r.json().get("answers")
        if not isinstance(answers, dict):
            raise GatewayUnavailable(f"resposta sem `answers`: {r.text[:300]}")
        return answers


def boolean(instructions: str, true_label: str = "sim", false_label: str = "não") -> dict:
    return {"type": "boolean", "instructions": instructions, "criteria": {"true": true_label, "false": false_label}}


def choice(instructions: str, options: dict[str, str]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": options}


def score(instructions: str, levels: list[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}


def probability(answers: dict, key: str) -> float | None:
    a = answers.get(key) or {}
    if a.get("type") == "boolean" or "probability" in a:
        p = a.get("probability")
        return float(p) if isinstance(p, (int, float)) else None
    if "score" in a and isinstance(a["score"], (int, float)):
        # score é índice fracionário na escala de níveis; sem níveis aqui, normaliza depois
        return None
    return None
