"""Logits do checkpoint OFICIAL do Community Forensics — lado torch da validação do ONNX.

Pré-processamento do model card: resize borda-curta -> 440, center crop 384, normalização CLIP.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MEDIA = (0.48145466, 0.4578275, 0.40821073)
DESVIO = (0.26862954, 0.26130258, 0.27577711)


def preprocessar(path, side=384, resize=440):
    img = Image.open(path).convert("RGB")
    escala = resize / min(img.size)
    img = img.resize((max(side, round(img.width * escala)), max(side, round(img.height * escala))), Image.BICUBIC)
    l, t = (img.width - side) // 2, (img.height - side) // 2
    img = img.crop((l, t, l + side, t + side))
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - np.array(MEDIA, dtype=np.float32)) / np.array(DESVIO, dtype=np.float32)
    return torch.from_numpy(a.transpose(2, 0, 1))[None]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/tmp/cf/oficial.pt")
    ap.add_argument("--imagens", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import timm  # o checkpoint usa a arquitetura timm do model card

    bruto = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    estado = bruto["model"] if isinstance(bruto, dict) and "model" in bruto else bruto
    # o checkpoint usa prefixo "vit." e head de 1 saída (logit binário)
    estado = {k.removeprefix("module.").removeprefix("vit."): v for k, v in estado.items()}
    saidas = int(estado["head.weight"].shape[0])
    modelo = timm.create_model("vit_small_patch16_384", pretrained=False, num_classes=saidas)
    faltando, extras = modelo.load_state_dict(estado, strict=False)
    print(f"faltando={len(faltando)} extras={len(extras)} | head={tuple(estado['head.weight'].shape)} "
          f"| params={sum(v.numel() for v in modelo.parameters())/1e6:.1f}M", file=sys.stderr)
    modelo.eval()
    vals = []
    with torch.no_grad():
        for p in a.imagens:
            vals.append(modelo(preprocessar(p)).reshape(-1).tolist())
    Path(a.out).write_text(json.dumps({"oficial.pt": vals}))
    print(f"{len(vals)} imagens -> {a.out}", file=sys.stderr)
