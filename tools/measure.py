"""Mede o poder discriminativo de cada sinal, por regime de degradação, e calibra τ.

Uso:  python tools/measure.py [--fakes DIR] [--reais ZIP] [--n-reais 300] [--out medicao.json]

não é parte do pacote: é ferramenta de bancada. Roda offline sobre a amostra baixada.
Mede AUC (Mann-Whitney) de cada sinal isolado — fakes contra reais — em quatro regimes:
limpo, JPEG q75, resize 0.5 + JPEG q75, e "screenshot" (resize + leve blur + reencode),
que é o regime que mais interessa e o que nenhum paper publica para IA.
"""

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from aiwd.layers import forensics

REGIMES = ("limpo", "jpeg75", "resize50", "screenshot")
SINAIS = ("spectral_peak_db", "generator_resolution", "noise_sigma_flat", "noise_luma_corr",
          "hf_minus_mid_db", "uniform_row_fraction", "flat_fraction")


def degradar(data: bytes, regime: str) -> bytes:
    if regime == "limpo":
        return data
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if regime == "resize50":
        img = img.resize((max(64, img.width // 2), max(64, img.height // 2)), Image.LANCZOS)
    elif regime == "screenshot":
        img = img.resize((int(img.width * 1.7), int(img.height * 1.7)), Image.BICUBIC)
        img = img.filter(ImageFilter.GaussianBlur(0.6))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=75 if regime != "limpo" else 92)
    return buf.getvalue()


def features(data: bytes) -> dict:
    camada = forensics.analyze(data, "")
    s = camada["signals"]
    out = {k: s.get(k) for k in SINAIS}
    out["hf_minus_mid_db"] = (s.get("spectrum_db") or {}).get("hf_minus_mid_db")
    out["spectral_peak_db"] = s.get("spectral_peak_db") or 0.0
    out["generator_resolution"] = float(bool(s.get("generator_resolution")))
    out["uniform_row_fraction"] = float(s.get("uniform_row_fraction") or 0.0)
    notas = camada["notes"]
    out["_n_findings"] = len(camada["findings"])
    out["_screenshot_cues"] = any("screenshot" in n for n in notas)
    out["_score_composto"] = _composto(out)
    return out


def _composto(f: dict) -> float:
    """Score 0..1 combinando os sinais na direção MEDIDA (não na suposta), para calibrar τ.

    Direções vêm da medição em /tmp/calib/medicao.json: resolução canônica e déficit de alta
    frequência apontam para IA; ruído de sensor acoplado à luminância aponta para câmera.
    """
    pistas = []
    pistas.append(0.9 if f.get("generator_resolution") else 0.1)
    hf = f.get("hf_minus_mid_db")
    if hf is not None:
        pistas.append(float(np.clip((10.0 - hf) / 20.0, 0.05, 0.95)))  # medido: fake tem hf menor
    corr = f.get("noise_luma_corr")
    if corr is not None:
        pistas.append(float(np.clip((0.25 - abs(corr)) * 3, 0.05, 0.95)))
    ff = f.get("flat_fraction")
    if ff is not None:
        pistas.append(float(np.clip(ff * 4, 0.05, 0.95)))
    picos = f.get("spectral_peak_db") or 0.0
    pistas.append(float(np.clip(picos / 25.0, 0.05, 0.95)))
    return round(float(np.mean(pistas)), 4)


def _ranks_medios(x: np.ndarray) -> np.ndarray:
    """Ranks com empate resolvido pela média — sem isso, sinal com muitos valores iguais
    (ex. resolução canônica toda zerada após upscale) devolve AUC 0.0 em vez de 0.5."""
    ordem = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[ordem] = np.arange(1, len(x) + 1, dtype=float)
    xs = x[ordem]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[ordem[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return ranks


def auc(pos: list[float], neg: list[float]) -> float | None:
    """AUC por estatística de rank (Mann-Whitney) com empate por média. Ignora None."""
    p = [x for x in pos if x is not None]
    n = [x for x in neg if x is not None]
    if len(p) < 10 or len(n) < 10:
        return None
    ranks = _ranks_medios(np.array(p + n, dtype=float))
    rp = ranks[:len(p)].sum()
    return float((rp - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def reais_do_zip(caminho: Path, n: int) -> list[bytes]:
    out = []
    with zipfile.ZipFile(caminho) as z:
        for nome in z.namelist():
            if not nome.lower().endswith(".jpg"):
                continue
            out.append(z.read(nome))
            if len(out) >= n:
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fakes", default=str(Path(__file__).resolve().parents[1] / "bench" / "gen"))
    ap.add_argument("--reais", default=str(Path(__file__).resolve().parents[1] / "bench" / "reals" / "val2017.zip"))
    ap.add_argument("--n-reais", type=int, default=300)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "bench" / "medicao.json"))
    ap.add_argument("--scores", default=str(Path(__file__).resolve().parents[1] / "bench" / "scores.json"),
                    help="scores por imagem, para o ajuste de τ em tools/fit_calibration.py")
    args = ap.parse_args()

    fakes: dict[str, list[bytes]] = {}
    for d in sorted(Path(args.fakes).iterdir()):
        if d.is_dir():
            fakes[d.name] = [p.read_bytes() for p in sorted(d.iterdir()) if p.is_file()]
    reais = reais_do_zip(Path(args.reais), args.n_reais)
    print(f"fakes: {sum(len(v) for v in fakes.values())} de {len(fakes)} geradores | reais: {len(reais)}")

    resultado: dict = {"fakes_por_gerador": {k: len(v) for k, v in fakes.items()}, "n_reais": len(reais),
                       "regimes": {}}
    # score composto das camadas = soma dos |sinal| que apontam para IA, na direção já conhecida
    scores: dict = {}
    for regime in REGIMES:
        feat_reais = [features(degradar(b, regime)) for b in reais]
        por_gerador = {}
        for gen, blobs in fakes.items():
            por_gerador[gen] = [features(degradar(b, regime)) for b in blobs]
        tabela = {}
        for sinal in SINAIS + ("_n_findings",):
            linha = {}
            for gen, feats in por_gerador.items():
                linha[gen] = auc([f[sinal] for f in feats], [f[sinal] for f in feat_reais])
            tabela[sinal] = linha
        resultado["regimes"][regime] = tabela
        scores[regime] = {
            "fakes": {gen: [features(degradar(b, regime))["_score_composto"] for b in blobs]
                      for gen, blobs in fakes.items()},
            "reais": [f["_score_composto"] for f in feat_reais],
        }
        print(f"\n=== regime {regime}")
        for sinal, linha in tabela.items():
            aucs = [v for v in linha.values() if v is not None]
            media = sum(aucs) / len(aucs) if aucs else None
            print(f"  {sinal:<22} " + (f"AUC médio {media:.3f}  " if media is not None else "sem amostra   ")
                  + " ".join(f"{g[:12]}={v:.2f}" if v is not None else f"{g[:12]}=--" for g, v in linha.items()))

    Path(args.out).write_text(json.dumps(resultado, indent=2))
    Path(args.scores).write_text(json.dumps(scores, indent=2))
    print(f"\n-> {args.out}")
    print(f"-> {args.scores}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
