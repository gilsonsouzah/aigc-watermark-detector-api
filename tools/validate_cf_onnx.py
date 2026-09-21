"""Valida o ONNX do Community Forensics contra o checkpoint oficial, numericamente.

Por que existe: esse ONNX já foi publicado com defeito uma vez (heads trocados, sem
center-crop), e a versão int8 que está no repo HOJE também não reproduz o checkpoint
(medido: Δ 0.42 no sigmoide). Export de terceiro não se confia por leitura.

    # lado torch (precisa de venv com torch + timm; o checkpoint oficial está no repo)
    /tmp/venv312/bin/python tools/_cf_torch_logits.py --ckpt /tmp/cf/oficial.pt \\
        --imagens /tmp/probeval/* --out /tmp/cf_torch.json
    # lado onnx + comparação
    python tools/validate_cf_onnx.py --torch-json /tmp/cf_torch.json --imagens /tmp/probeval

Sai != 0 se algum arquivo divergir além da tolerância.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from aiwd.layers.image_model import preparar


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf-dir", default="/tmp/cf")
    ap.add_argument("--imagens", default="/tmp/probeval")
    ap.add_argument("--torch-json", required=True)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    referencia = json.loads(Path(args.torch_json).read_text())
    ref = next(iter(referencia.values()), [])
    imagens = sorted(p for p in Path(args.imagens).iterdir() if p.is_file())
    falhas = 0

    for onnx_path in sorted(Path(args.cf_dir).glob("*.onnx")):
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        entrada = sess.get_inputs()[0]
        difs = []
        for img, esperado in zip(imagens, ref, strict=False):
            x = preparar(img.read_bytes())
            if x is None:
                continue
            logit = float(np.asarray(sess.run(None, {entrada.name: x})[0]).reshape(-1)[0])
            e = float(np.asarray(esperado).reshape(-1)[0])
            difs.append(abs(1 / (1 + np.exp(-logit)) - 1 / (1 + np.exp(-e))))
        maior = max(difs) if difs else float("nan")
        ok = bool(difs) and maior <= args.tol
        print(f"{onnx_path.name:<22} n={len(difs):<4} max|Δ|p={maior:.2e} {'ok' if ok else 'DIVERGE — não usar'}")
        falhas += 0 if ok else 1

    if falhas:
        print(f"\n{falhas} arquivo(s) reprovado(s) na validação")
        return 1
    print("\nONNX reproduz o checkpoint oficial")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
