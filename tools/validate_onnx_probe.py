"""Valida o ONNX exportado contra o .pt original, numericamente.

O modelo comunitário já foi publicado uma vez com pesos errados (o ONNX do Community
Forensics saiu com heads trocados e sem center-crop), então export de terceiro não se
confia por leitura: compara lado a lado com o checkpoint.

    /tmp/venv312/bin/python tools/_probe_torch_logits.py --imagens /tmp/probeval/* \\
        --out /tmp/probeval_torch.json --side 224 --modo resize
    python tools/validate_onnx_probe.py --torch-json /tmp/probeval_torch.json --imagens /tmp/probeval

Sai != 0 se algum modelo divergir além da tolerância.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from aiwd.layers import community_probe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", default="/tmp/aigc-probes")
    ap.add_argument("--imagens", default="/tmp/probeval")
    ap.add_argument("--torch-json", required=True)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    torch_logits = json.loads(Path(args.torch_json).read_text())
    imagens = sorted(p for p in Path(args.imagens).iterdir() if p.is_file())
    falhas = 0
    for onnx_path in sorted(Path(args.pesos).glob("*.onnx")):
        chave = onnx_path.stem.replace("_224", "")
        if chave not in torch_logits:
            print(f"{onnx_path.name}: sem logits do torch para {chave!r} — pule")
            falhas += 1
            continue
        sess, erro = community_probe._sessao(onnx_path)
        if sess is None:
            print(f"{onnx_path.name}: {erro}")
            falhas += 1
            continue
        entrada = sess.get_inputs()[0]
        side = int(entrada.shape[2]) if isinstance(entrada.shape[2], int) else 224
        difs = []
        for img, esperado in zip(imagens, torch_logits[chave], strict=False):
            x = community_probe._preparar(img.read_bytes(), side)
            if x is None:
                continue
            p_onnx = community_probe._prob(sess.run(None, {entrada.name: x})[0])
            p_torch = 1 / (1 + np.exp(-float(esperado)))
            difs.append(abs(p_onnx - p_torch))
        maior = max(difs) if difs else float("nan")
        ok = difs and maior <= args.tol
        print(f"{onnx_path.name:<44} n={len(difs):<4} max|Δ|p={maior:.2e} {'ok' if ok else 'DIVERGE'}")
        falhas += 0 if ok else 1

    if falhas:
        print(f"\n{falhas} modelo(s) sem validação — NÃO usar antes de resolver")
        return 1
    print("\ntodos os modelos batem com o checkpoint original")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
