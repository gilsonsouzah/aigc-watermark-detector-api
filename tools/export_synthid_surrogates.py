"""Exporta os pesos .pt do repo comunitário gpt-image-synthid-detector para ONNX.

Por que em script separado: os pesos são PolyForm Noncommercial (uso comercial proibido),
então NÃO entram neste repositório. Este script baixa para um diretório fora do repo e
exporta ONNX, que é o que a inferência usa (onnxruntime, sem torch).

Precisa rodar num venv com torch (Python 3.12, porque torch não tem wheel para 3.14):

    uv venv /tmp/venv312 --python 3.12
    uv pip install --python /tmp/venv312 torch torchvision onnx --index-url \\
        https://download.pytorch.org/whl/cpu
    /tmp/venv312/bin/python tools/export_synthid_surrogates.py --dest /tmp/aigc-probes

Uso:  python tools/export_synthid_surrogates.py [--dest DIR] [--side 224]
"""

import argparse
import urllib.request
from pathlib import Path

REPO = "https://raw.githubusercontent.com/newideas99/gpt-image-synthid-detector/main/weights"
MODELOS = ("surrogate_efficientnet_b0", "surrogate_resnet18", "surrogate_resnet34")


def baixar(nome: str, dest: Path) -> Path:
    alvo = dest / f"{nome}.pt"
    if alvo.is_file():
        return alvo
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{REPO}/{nome}.pt", timeout=300) as r, alvo.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    return alvo


def construir(nome: str):
    """Os checkpoints têm 1 saída (logit único, detector binário) — o head tem que bater."""
    from torch import nn
    from torchvision import models

    if "efficientnet_b0" in nome:
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    elif "resnet18" in nome:
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 1)
    else:
        m = models.resnet34(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="/tmp/aigc-probes")
    ap.add_argument("--side", type=int, default=224)
    args = ap.parse_args()
    import torch

    dest = Path(args.dest)
    for nome in MODELOS:
        pt = baixar(nome, dest)
        sd = torch.load(pt, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        modelo = construir(nome)
        faltando, extras = modelo.load_state_dict(sd, strict=False)
        pesos = dict(modelo.named_parameters())
        n_params = sum(v.numel() for v in pesos.values())
        cabeca = pesos["fc.weight"] if "fc.weight" in pesos else pesos["classifier.1.weight"]
        print(f"  params={n_params/1e6:.2f}M  saída={tuple(cabeca.shape)} esperado_fp32={n_params*4/1e6:.1f} MB")
        print(f"{nome}: chaves faltando={len(faltando)} extras={len(extras)}")
        modelo.eval()
        onnx_path = dest / f"{nome}_{args.side}.onnx"
        torch.onnx.export(modelo, torch.zeros(1, 3, args.side, args.side), onnx_path,
                          input_names=["input"], output_names=["logits"], opset_version=17,
                          dynamic_axes={"input": {0: "batch"}})
        # o exporter grava os pesos como arquivo externo (.onnx.data); junta tudo num arquivo só
        import onnx

        modelo_onnx = onnx.load(str(onnx_path))
        onnx.save_model(modelo_onnx, str(onnx_path), save_as_external_data=False)
        for sobra in dest.glob(f"{onnx_path.name}.data"):
            sobra.unlink()
        print(f"  -> {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
