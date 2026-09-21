"""Hipótese: o detector neural vê IA DENTRO da foto quando olha o bloco certo?

A dispersão de ruído entre blocos falhou (AUC ~0.5, medido). Esta é a outra rota: se um pedaço
foi gerado por IA e colado numa foto real, o detector Community Forensics — que dá AUC 0.999 no
global — deve pontuar alto NO BLOCO do patch e baixo no resto da mesma imagem.

Controle pareado: para cada imagem, compara o score no bloco centrado no patch com o score num
bloco de controle da mesma imagem. O que importa é a DIFERENÇA dentro da imagem, o que cancela
cena, câmera e compressão — é o teste mais limpo possível para "tem IA aqui dentro".

Uso:  bench/run.sh python -m tools.measure_tile_patch --n 80
"""

import argparse
import io
import random
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFilter

from aiwd.layers.image_model import preparar

RAIZ = Path(__file__).resolve().parents[1]
rng = random.Random(7)
LADO_BLOCO = 320


def patch_em(destino: bytes, fonte: bytes) -> tuple[bytes, tuple[int, int]]:
    """Cola recorte da fonte no destino e devolve os bytes e o CENTRO do patch."""
    a = Image.open(io.BytesIO(destino)).convert("RGB")
    b = Image.open(io.BytesIO(fonte)).convert("RGB")
    lado = min(LADO_BLOCO, int(min(a.size) * 0.5))
    corte = b.resize((lado, lado), Image.BICUBIC)
    mascara = Image.new("L", (lado, lado), 0)
    ImageDraw.Draw(mascara).ellipse((3, 3, lado - 3, lado - 3), fill=255)
    mascara = mascara.filter(ImageFilter.GaussianBlur(lado * 0.05))
    x, y = rng.randint(0, max(0, a.width - lado)), rng.randint(0, max(0, a.height - lado))
    a.paste(corte, (x, y), mascara)
    buf = io.BytesIO(); a.save(buf, "JPEG", quality=88)
    return buf.getvalue(), (x + lado // 2, y + lado // 2)


def score_em(sess, data: bytes, centro=None, lado: int = 320) -> float:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if centro is None:
        cx, cy = rng.randint(lado, max(lado, img.width - lado)), rng.randint(lado, max(lado, img.height - lado))
    else:
        cx, cy = centro
    cx = min(max(cx, lado // 2), img.width - lado // 2)
    cy = min(max(cy, lado // 2), img.height - lado // 2)
    recorte = img.crop((cx - lado // 2, cy - lado // 2, cx + lado // 2, cy + lado // 2))
    buf = io.BytesIO(); recorte.save(buf, "JPEG", quality=95)
    # usa o mesmo pré-processamento da camada (resize curto 440 + crop 384 + norm CLIP)
    from PIL import Image as I
    recorte = I.open(io.BytesIO(buf.getvalue())).convert("RGB")
    entrada = preparar(_para_bytes(recorte))
    logit = float(np.asarray(sess.run(None, {sess.get_inputs()[0].name: entrada})[0]).reshape(-1)[0])
    return float(1 / (1 + np.exp(-logit)))


def _para_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO(); img.save(b, "JPEG", quality=95); return b.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--modelo", default=str(RAIZ / "bench" / "models" / "model.onnx"))
    a = ap.parse_args()

    sess = ort.InferenceSession(a.modelo, providers=["CPUExecutionProvider"])
    with zipfile.ZipFile(RAIZ / "bench" / "reals" / "val2017.zip") as z:
        reais = [z.read(n) for n in z.namelist() if n.endswith(".jpg")][:a.n]
    gerados = [p.read_bytes() for d in sorted((RAIZ / "bench" / "gen").iterdir()) if d.is_dir()
               for p in sorted(d.iterdir())[:a.n // 6 or 1]]

    difs = []
    for foto, fonte in zip(reais, gerados * (len(reais) // max(1, len(gerados)) + 1), strict=False):
        com_patch, centro = patch_em(foto, fonte)
        p_patch = score_em(sess, com_patch, centro)
        p_controle = score_em(sess, com_patch, None)      # outro ponto da MESMA imagem
        p_intacta = score_em(sess, foto, None)            # imagem sem patch
        difs.append((p_patch, p_controle, p_intacta))

    patch = np.array([d[0] for d in difs])
    cont = np.array([d[1] for d in difs])
    intact = np.array([d[2] for d in difs])
    print(f"n={len(difs)} imagens reais, cada uma com um patch gerado colado\n")
    print(f"score no bloco DO PATCH : média={patch.mean():.3f} mediana={np.median(patch):.3f}")
    print(f"score em bloco CONTROLE : média={cont.mean():.3f} mediana={np.median(cont):.3f}")
    print(f"score em foto INTACTA   : média={intact.mean():.3f} mediana={np.median(intact):.3f}")
    acerto = float(((patch > cont).mean() + (patch > intact).mean()) / 2)
    print(f"\ntaxa de acerto pareada (patch pontua mais alto que o controle): {acerto:.3f}")
    print(f"diferença média patch-controle: {(patch - cont).mean():+.3f}")
    print(f"quantos patches passam de 0.5 (limiar cru de 'é IA'): {(patch > 0.5).mean():.3f}")
    print(f"quantos controles passam de 0.5:                      {(cont > 0.5).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
