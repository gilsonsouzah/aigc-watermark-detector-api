"""Mede o pipeline INTEIRO (camadas offline) contra amostra rotulada, por regime.

Diferente de tools/measure.py, que mede sinais isolados, aqui vale o veredito final:
acurácia balanceada, taxa de falso positivo em foto real, e quanto cada regime degrada.
Roda com Jev desligado para medir só o que é determinístico/local e reprodutível.

Uso:  python -m tools.evaluate_pipeline [--n-fakes 30] [--n-reais 120]
"""

import argparse
import json
import zipfile
from pathlib import Path

from aiwd.config import Config
from aiwd.pipeline import detect
from tools.measure import degradar

OFFLINE = Config(key=None)
POSITIVO = {"ai", "likely_ai"}
NEGATIVO = {"human", "likely_human"}


def avaliar(blobs: list[bytes], nome: str, regime: str) -> list[dict]:
    saidas = []
    for i, b in enumerate(blobs):
        r = detect(degradar(b, regime), f"{nome}_{i}.jpg", OFFLINE)
        saidas.append({"p": r["ai_probability"], "verdict": r["verdict"], "conf": r["confidence"]})
    return saidas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fakes", default=str(Path(__file__).resolve().parents[1] / "bench" / "gen"))
    ap.add_argument("--reais", default=str(Path(__file__).resolve().parents[1] / "bench" / "reals" / "val2017.zip"))
    ap.add_argument("--n-fakes", type=int, default=30)
    ap.add_argument("--n-reais", type=int, default=120)
    ap.add_argument("--regimes", nargs="+", default=["limpo", "jpeg75", "screenshot"])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "bench" / "pipeline_eval.json"))
    args = ap.parse_args()

    fakes = {d.name: [p.read_bytes() for p in sorted(d.iterdir())[:args.n_fakes]]
             for d in sorted(Path(args.fakes).iterdir()) if d.is_dir()}
    with zipfile.ZipFile(args.reais) as z:
        reais = [z.read(n) for n in z.namelist() if n.endswith(".jpg")][:args.n_reais]

    relatorio = {}
    for regime in args.regimes:
        res_reais = avaliar(reais, "coco", regime)
        fp = sum(1 for r in res_reais if r["verdict"] in POSITIVO) / len(res_reais)
        abst_r = sum(1 for r in res_reais if r["verdict"] == "inconclusive") / len(res_reais)
        linhas = {}
        for gen, blobs in fakes.items():
            res = avaliar(blobs, gen, regime)
            tp = sum(1 for r in res if r["verdict"] in POSITIVO) / len(res)
            abst = sum(1 for r in res if r["verdict"] == "inconclusive") / len(res)
            linhas[gen] = {"recall": round(tp, 3), "abstencao": round(abst, 3)}
        recall_medio = sum(v["recall"] for v in linhas.values()) / len(linhas)
        relatorio[regime] = {"falso_positivo_em_foto_real": round(fp, 3),
                             "abstencao_em_foto_real": round(abst_r, 3),
                             "recall_medio": round(recall_medio, 3),
                             "acuracia_balanceada": round((recall_medio + (1 - fp)) / 2, 3),
                             "por_gerador": linhas}
        print(f"\n=== {regime}: acurácia balanceada={relatorio[regime]['acuracia_balanceada']} "
              f"| recall médio={recall_medio:.3f} | falso positivo em foto={fp:.3f} "
              f"| abstenção em foto={abst_r:.3f}")
        for gen, v in linhas.items():
            print(f"    {gen:<16} recall={v['recall']:<6} abstenção={v['abstencao']}")
    Path(args.out).write_text(json.dumps(relatorio, indent=2))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
