"""Mede a sonda neural comunitária por FAMÍLIA de gerador — separa "detector de SynthID" de
"classificador de jeitão IA".

O desenho do teste: geradores do Google (imagen-4, Nano-Banana) aplicam SynthID; FLUX, SD e
Seedream não aplicam watermark. Se a sonda fosse detector de SynthID, o AUC ficaria alto só nos
do Google. Se o AUC for alto em todos, ela detecta "cara de imagem gerada", não watermark — e
tem que ser nomeada e pesada como tal.
"""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from aiwd.layers import community_probe
from tools.measure import auc, degradar  # reuso: mesmo AUC e mesmas degradações

GOOGLE = {"imagen-4", "Nano-Banana"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", default=str(Path(__file__).resolve().parents[1] / "bench" / "probes"))
    ap.add_argument("--fakes", default=str(Path(__file__).resolve().parents[1] / "bench" / "gen"))
    ap.add_argument("--reais", default=str(Path(__file__).resolve().parents[1] / "bench" / "reals" / "val2017.zip"))
    ap.add_argument("--n-reais", type=int, default=200)
    ap.add_argument("--regimes", nargs="+", default=["limpo", "resize50", "screenshot"])
    ap.add_argument("--out", default="/tmp/calib/probe_medicao.json")
    args = ap.parse_args()

    modelos = sorted(Path(args.pesos).glob("*.onnx"))
    sessoes = {p.stem: community_probe._sessao(p)[0] for p in modelos}
    print("modelos:", list(sessoes))

    def prob(modelo: str, data: bytes) -> float | None:
        sess = sessoes[modelo]
        entrada = sess.get_inputs()[0]
        side = int(entrada.shape[2]) if isinstance(entrada.shape[2], int) else 224
        x = community_probe._preparar(data, side)
        if x is None:
            return None
        return community_probe._prob(sess.run(None, {entrada.name: x})[0])

    with zipfile.ZipFile(args.reais) as z:
        reais = [z.read(n) for n in z.namelist() if n.endswith(".jpg")][:args.n_reais]
    fakes = {d.name: [p.read_bytes() for p in sorted(d.iterdir())] for d in sorted(Path(args.fakes).iterdir()) if d.is_dir()}
    print(f"reais={len(reais)} fakes={ {k: len(v) for k, v in fakes.items()} }")

    resultado = {}
    for regime in args.regimes:
        reais_p = {m: [prob(m, degradar(b, regime)) for b in reais] for m in sessoes}
        tabela = {}
        for gen, blobs in fakes.items():
            probs = {m: [prob(m, degradar(b, regime)) for b in blobs] for m in sessoes}
            tabela[gen] = {m: auc([p for p in probs[m] if p is not None], reais_p[m]) for m in sessoes}
        resultado[regime] = tabela
        print(f"\n=== regime {regime}  (google={sorted(GOOGLE)})")
        for m in sessoes:
            g = [tabela[k][m] for k in tabela if k in GOOGLE and tabela[k][m] is not None]
            ng = [tabela[k][m] for k in tabela if k not in GOOGLE and tabela[k][m] is not None]
            por_gerador = " ".join(f"{k[:10]}={tabela[k][m]:.2f}" for k in tabela if tabela[k][m] is not None)
            print(f"  {m:<32} AUC google={np.mean(g):.3f}  não-google={np.mean(ng):.3f}  | {por_gerador}")
    Path(args.out).write_text(json.dumps(resultado, indent=2))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
