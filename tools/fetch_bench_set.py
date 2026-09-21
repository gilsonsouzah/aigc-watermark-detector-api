"""Baixa o conjunto de bancada para bench/ (persistente; bench/ está no .gitignore).

Três partes, todas reprodutíveis e com licença declarada no README:

  bench/gen/<gerador>/   amostra do T2I-CoReBench (Apache-2.0) — extraída por STREAMING do
                         tar.gz remoto, que tem ~7 GB por gerador: para de baixar assim que
                         junta N imagens, em vez de trazer o arquivo inteiro.
  bench/reals/           COCO val2017 (termos do COCO) — fotos reais como negativos.
  bench/models/          Community Forensics ONNX fp32 + checkpoint oficial (MIT) para validar.

Uso:  bench/run.sh python tools/fetch_bench_set.py --por-gerador 80 --n-reais 400
"""

import argparse
import io
import tarfile
import time
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
BENCH = RAIZ / "bench"
COREBENCH = "https://huggingface.co/datasets/lioooox/T2I-CoReBench-Images/resolve/main"
GERADORES_PADRAO = ("GPT-Image-1.5", "imagen-4", "FLUX.1-dev", "SD-3.5-Large", "Nano-Banana", "Seedream-4")
COCO = "http://images.cocodataset.org/zips/val2017.zip"
CF = ("https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT/resolve/main/"
      "onnx/model.onnx", "model.onnx")
CF_CKPT = ("https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT/resolve/main/"
           "pretrained_weights/model_v11_ViT_384_base_ckpt.pt", "oficial.pt")
EXTS = (".png", ".jpg", ".jpeg", ".webp")


def amostrar_gerador(nome: str, n: int) -> int:
    destino = BENCH / "gen" / nome
    destino.mkdir(parents=True, exist_ok=True)
    if len(list(destino.glob("*"))) >= n:
        return len(list(destino.glob("*")))
    lidos, inicio = 0, time.time()

    class Contador(io.RawIOBase):
        def readable(self): return True
        def readinto(self, b):
            nonlocal lidos
            pedaco = resposta.read(len(b))
            lidos += len(pedaco)
            b[:len(pedaco)] = pedaco
            return len(pedaco)

    resposta = urllib.request.urlopen(f"{COREBENCH}/{nome}.tar.gz", timeout=120)
    with tarfile.open(fileobj=Contador(), mode="r|gz") as tar:
        got = 0
        for membro in tar:
            if not membro.isfile() or not membro.name.lower().endswith(EXTS):
                continue
            f = tar.extractfile(membro)
            if f is None:
                continue
            dados = f.read()
            if len(dados) < 20_000:      # pula thumbs e placeholders
                continue
            (destino / f"{got:04d}{Path(membro.name).suffix.lower()}").write_bytes(dados)
            got += 1
            if got >= n:
                break
    print(f"  {nome}: {got} imagens, {lidos/1e6:.0f} MB baixados, {time.time()-inicio:.0f}s", flush=True)
    return got


def baixar(url: str, destino: Path) -> None:
    if destino.exists() and destino.stat().st_size > 1000:
        print(f"  já existe: {destino.name} ({destino.stat().st_size/1e6:.1f} MB)")
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"  baixando {destino.name}...", flush=True)
    with urllib.request.urlopen(url, timeout=300) as r, destino.open("wb") as f:
        while pedaco := r.read(1 << 20):
            f.write(pedaco)
    print(f"  {destino.name}: {destino.stat().st_size/1e6:.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--por-gerador", type=int, default=80)
    ap.add_argument("--n-reais", type=int, default=400)
    ap.add_argument("--geradores", nargs="*", default=list(GERADORES_PADRAO))
    ap.add_argument("--sem-modelos", action="store_true")
    args = ap.parse_args()

    print("geradores (T2I-CoReBench, Apache-2.0):")
    for g in args.geradores:
        amostrar_gerador(g, args.por_gerador)
    print("fotos reais (COCO val2017):")
    baixar(COCO, BENCH / "reals" / "val2017.zip")
    if not args.sem_modelos:
        print("modelos (Community Forensics, MIT):")
        baixar(CF[0], BENCH / "models" / CF[1])
        baixar(CF_CKPT[0], BENCH / "models" / CF_CKPT[1])
    print(f"\npronto em {BENCH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
