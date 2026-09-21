"""Congela UMA passada do pipeline com o Jev ligado, para poder varrer pesos offline.

Roda o pipeline em amostra rotulada e guarda o JSON completo de cada arquivo (incluindo as
respostas do Jev). A partir do cache, `tools/sweep_jev_weight.py` recalcula veredito em
qualquer peso SEM gastar chamada de API — e sem esbarrar em rate limit no meio da medição.
"""

import argparse
import json
import zipfile
from pathlib import Path

from aiwd.config import Config
from aiwd.pipeline import detect

RAIZ = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fakes", type=int, default=8)
    ap.add_argument("--n-reais", type=int, default=30)
    ap.add_argument("--out", default=str(RAIZ / "bench" / "jev_cache.json"))
    a = ap.parse_args()

    cfg = Config.from_env()
    with zipfile.ZipFile(RAIZ / "bench" / "reals" / "val2017.zip") as z:
        reais = [z.read(n) for n in z.namelist() if n.endswith(".jpg")][:a.n_reais]
    fakes = {d.name: [p.read_bytes() for p in sorted(d.iterdir())[:a.n_fakes]]
             for d in sorted((RAIZ / "bench" / "gen").iterdir()) if d.is_dir()}

    amostras = [("real", f"coco_{i}.jpg", b) for i, b in enumerate(reais)]
    amostras += [(g, f"{g}_{i}.png", b) for g, blobs in fakes.items() for i, b in enumerate(blobs)]

    saida = []
    for rotulo, nome, blob in amostras:
        r = detect(blob, nome, cfg)
        juiz = r["judge"]
        saida.append({"rotulo": rotulo, "arquivo": nome, "kind": r["kind"],
                      "verdict": r["verdict"], "ai_probability": r["ai_probability"],
                      "confianca": r["confidence"], "respostas_do_juiz": juiz.get("answers"),
                      "camadas": {q["id"]: [{k: l[k] for k in ("layer", "score", "reliability")}
                                            for l in q["layers"]] for q in r["questions"]}})
        p = r["ai_probability"]
        print(f"  {nome[:30]:<30} {r['verdict']:<12} p={p}  jev={'sim' if juiz['used'] else 'NAO'}", flush=True)
        Path(a.out).write_text(json.dumps(saida, indent=2, ensure_ascii=False))
    ok = sum(1 for x in saida if x["respostas_do_juiz"])
    print(f"\n{len(saida)} arquivos, {ok} com resposta do juiz -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
