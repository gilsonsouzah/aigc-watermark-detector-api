import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_env(path: Path | None = None) -> None:
    """Lê KEY=VALUE de .env para o ambiente sem sobrescrever o que já existe."""
    for f in (path or ROOT / ".env", Path.cwd() / ".env"):
        if not f.is_file():
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


@dataclass(frozen=True)
class Config:
    gateway_url: str = "https://ai-gateway.vercel.sh/v1"
    key: str | None = None
    judge_model: str = "typesafe-ai/jev"
    timeout: float = 60.0
    max_upload_mb: int = 25
    max_state_chars: int = 12000
    synthid_template: str | None = None

    @classmethod
    def from_env(cls) -> "Config":
        load_env()
        return cls(
            gateway_url=os.environ.get("AI_GATEWAY_URL", cls.gateway_url),
            key=os.environ.get("VERCEL_AI_GATEWAY_KEY") or None,
            judge_model=os.environ.get("JUDGE_MODEL", cls.judge_model),
            timeout=float(os.environ.get("AI_GATEWAY_TIMEOUT", cls.timeout)),
            max_upload_mb=int(os.environ.get("MAX_UPLOAD_MB", cls.max_upload_mb)),
            synthid_template=os.environ.get("SYNTHID_TEMPLATE") or None,
        )
