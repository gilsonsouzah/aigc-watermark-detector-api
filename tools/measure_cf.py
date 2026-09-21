"""Mede o detector Community Forensics (ONNX fp32) por família e por regime.

Módulo em uso: onnx fp32 (o int8 diverge 0.415 do checkpoint oficial — ver
tools/validate_cf_onnx.py; não usar).
"""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

from aiwd.layers.image_model import preparar
from tools.measure import auc, degradar


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(Path(__file__).resolve().parents[1] / "bench" / "models" / "model.onnx"))
    ap.add_argument("--fakes", default=str(Path(__file__).resolve().parents[1] / "bench" / "gen"))
    ap.add_argument("--reais", default=str(Path(__file__).resolve().parents[1] / "bench" / "reals" / "val2017.zip"))
    ap.add_argument("--n-reais", type=int, default=200)
    ap.add_argument("--regimes", nargs="+", default=["limpo", "jpeg75", "resize50", "screenshot"])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "bench" / "cf_medicao.json"))
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    ent = sess.get_inputs()[0]

    def prob(data: bytes) -> float | None:
        x = preparar(data)
        if x is None:
            return None
        v = float(np.asarray(sess.run(None, {ent.name: x})[0]).reshape(-1)[0])
        return float(1 / (1 + np.exp(-v)))

    with zipfile.ZipFile(args.reais) as z:
        reais = [z.read(n) for n in z.namelist() if n.endswith(".jpg")][:args.n_reais]
    fakes = {d.name: [p.read_bytes() for p in sorted(d.iterdir())] for d in sorted(Path(args.fakes).iterdir()) if d.is_dir()}
    print(f"reais={len(reais)} | fakes={sum(len(v) for v in fakes.values())}")

    saida = {}
    for regime in args.regimes:
        p_reais = [prob(degradar(b, regime)) for b in reais]
        linha = {gen: auc([p for p in (prob(degradar(b, regime)) for b in blobs) if p is not None],
                          [p for p in p_reais if p is not None]) for gen, blobs in fakes.items()}
        saida[regime] = linha
        vals = [v for v in linha.values() if v is not None]
        print(f"{regime:<11} AUC médio={np.mean(vals):.3f}  | "
              + " ".join(f"{k[:11]}={v:.2f}" for k, v in linha.items() if v is not None))
    Path(args.out).write_text(json.dumps(saida, indent=2))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
