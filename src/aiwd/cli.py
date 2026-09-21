"""Uso: aiwd ARQUIVO [--json] [--verbose]"""

import json
import sys
from pathlib import Path

from .config import Config
from .pipeline import detect


def _render(result: dict, verbose: bool) -> None:
    print(f"{result['verdict'].upper()}  p(IA)={result['ai_probability']}  confiança={result['confidence']} "
          f"({result['confidence_label']})  gerador={result['generator_guess']}  {result['elapsed_ms']}ms")
    for q in result["questions"]:
        print(f"  ? {q['id']}: {q['answer']}  [score={q['score']} conf={q['confidence']} {q['reliability']}]")
        for l in q["layers"]:
            if verbose or l["reliability"] in ("deterministic", "strong", "model"):
                print(f"      {l['layer']:<11} score={l['score']:<7} peso={l['weight']:<6} {l['reliability']:<13} "
                      f"{'; '.join(l['evidence'])[:100]}")
    for w in result["warnings"]:
        print(f"  ! {w}")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    flags = {a for a in args if a.startswith("--")}
    paths = [a for a in args if not a.startswith("--")]
    if not paths:
        print(__doc__ or "aiwd ARQUIVO [--json] [--verbose]", file=sys.stderr)
        return 2
    path = Path(paths[0])
    if not path.is_file():
        print(f"não é arquivo: {path}", file=sys.stderr)
        return 2
    result = detect(path.read_bytes(), path.name, Config.from_env())
    if "--json" in flags:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _render(result, "--verbose" in flags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
