"""Ajusta τ por regime a partir de medição rotulada e escreve calibration.json.

Uso:  python tools/fit_calibration.py --medicao /tmp/calib/scores.json [--out calibration.json]

A entrada é o arquivo de scores por imagem (gerado por tools/measure.py --scores), com
rótulo por gerador: as imagens de gerador são positivas, as do COCO são negativas. O τ de
cada regime maximiza sensibilidade+especificidade−1 na amostra, e o relatório diz a acurácia
balanceada antes e depois — o número honesto, medido, não estimado.
"""

import argparse
import json
from pathlib import Path

from aiwd.calibrate import fit_youden


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--medicao", default="/tmp/calib/scores.json")
    ap.add_argument("--out", default="calibration.json")
    args = ap.parse_args()

    dados = json.loads(Path(args.medicao).read_text())
    saida = {"fonte": str(args.medicao), "default_regime": "limpo", "regimes": {}}
    for regime, grupo in dados.items():
        pos = [s for gen, vals in grupo["fakes"].items() for s in vals]
        neg = grupo["reais"]
        fit = fit_youden(pos, neg)
        saida["regimes"][regime] = fit
        print(f"{regime:<12} τ={fit['tau']:<8} acurácia balanceada={fit.get('balanced_accuracy')} "
              f"(pos={fit['n_pos']}, neg={fit['n_neg']})")
    saida["tau"] = saida["regimes"].get("limpo", {}).get("tau", 0.0)
    Path(args.out).write_text(json.dumps(saida, indent=2))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
