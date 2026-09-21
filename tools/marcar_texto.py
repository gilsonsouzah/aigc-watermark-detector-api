"""Marca uma sequência de token ids para a chave dada (o watermarker vive em aiwd.synthid_text).

Uso:  python -m tools.marcar_texto --n 60 --vocab 300
"""

import argparse
import json
from pathlib import Path

from aiwd.synthid_text import CHAVES_DEMO, marcar, verificar


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--vocab", type=int, default=300)
    ap.add_argument("--out", default="/tmp/tokens_marcados.json")
    args = ap.parse_args()

    tokens = marcar(args.n, args.vocab, CHAVES_DEMO)
    Path(args.out).write_text(json.dumps(tokens))
    print(f"{len(tokens)} tokens marcados -> {args.out}")
    print("sob a chave CERTA:", verificar(tokens, CHAVES_DEMO)["media"])
    print("sob chave ERRADA: ", verificar(tokens, [c + 7 for c in CHAVES_DEMO])["media"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
