"""Roda os .pt no torch e grava os logits — lado torch da validação do ONNX. Venv 3.12."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models


def construir(nome):
    from torch import nn
    if "efficientnet_b0" in nome:
        m = models.efficientnet_b0(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    elif "resnet18" in nome:
        m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, 1)
    else:
        m = models.resnet34(weights=None); m.fc = nn.Linear(m.fc.in_features, 1)
    return m.eval()


def tensor(path, side, modo):
    img = Image.open(path).convert("RGB")
    if modo == "resize":
        img = img.resize((side, side), Image.BICUBIC)
    else:  # resize-curto + center crop (padrão torchvision eval)
        r = side / min(img.size)
        img = img.resize((max(side, round(img.width * r)), max(side, round(img.height * r))), Image.BICUBIC)
        l, t = (img.width - side) // 2, (img.height - side) // 2
        img = img.crop((l, t, l + side, t + side))
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(a.transpose(2, 0, 1))[None]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", default="/tmp/aigc-probes")
    ap.add_argument("--imagens", nargs="+", required=True)
    ap.add_argument("--side", type=int, default=224)
    ap.add_argument("--modo", default="resize")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {}
    for pt in sorted(Path(args.pesos).glob("*.pt")):
        m = construir(pt.stem)
        sd = torch.load(pt, map_location="cpu", weights_only=False)
        m.load_state_dict(sd.get("state_dict", sd) if isinstance(sd, dict) else sd)
        vals = []
        for img in args.imagens:
            with torch.no_grad():
                vals.append(float(m(tensor(img, args.side, args.modo)).reshape(-1)[0]))
        out[pt.stem] = vals
        print(f"{pt.stem}: {len(vals)} imagens", file=sys.stderr)
    Path(args.out).write_text(json.dumps(out))
