"""Bancada de alteração LOCAL e de documento. Três experimentos:

1. ALTERAÇÃO LOCAL em foto: foto intacta (COCO) vs foto com pedaço colado — dois tipos de
   colagem, patch de imagem gerada e patch de outra foto real (splicing). Mede a AUC das
   features de dispersão entre blocos e calibra o limiar no p95 das fotos intactas.
2. DOCUMENTO: página renderizada vs versão "print de tela" (reescalonada, desfocada,
   recomprimida). Mede acerto da camada document.
3. PDF: arquivo novo vs arquivo salvo com atualização INCREMENTAL de verdade (pypdf),
   que é como um editor altera PDF sem reescrever o arquivo — o rastro fica no container.

Negativos de alteração local são fotos reais não tocadas; positivos são colagens que eu
mesmo gero, então o número vale para ESTE tipo de manipulação — inpainting real é mais
sofisticado e o teto é declarado no README.

Uso:  python -m tools.measure_tamper [--n 120]
"""

import argparse
import io
import json
import random
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from aiwd.layers import document, forensics
from tools.measure import auc

rng = random.Random(0)


def fotos_reais(zip_path: str, n: int) -> list[bytes]:
    with zipfile.ZipFile(zip_path) as z:
        return [z.read(x) for x in z.namelist() if x.endswith(".jpg")][:n]


def patches_gerados(n: int) -> list[bytes]:
    base = Path(__file__).resolve().parents[1] / "bench" / "gen"
    if not base.is_dir():
        return []
    arquivos = [p for d in sorted(base.iterdir()) if d.is_dir() for p in sorted(d.iterdir())]
    return [p.read_bytes() for p in rng.sample(arquivos, min(n, len(arquivos)))]


def colar(destino: bytes, fonte: bytes, fracao: float = 0.18) -> bytes:
    """Cola um recorte da fonte no destino com borda suavizada, depois recomprime."""
    a = Image.open(io.BytesIO(destino)).convert("RGB")
    b = Image.open(io.BytesIO(fonte)).convert("RGB")
    lado = max(32, int(min(a.size) * fracao))
    corte = b.resize((lado, lado), Image.BICUBIC)
    mascara = Image.new("L", (lado, lado), 0)
    ImageDraw.Draw(mascara).ellipse((4, 4, lado - 4, lado - 4), fill=255)
    mascara = mascara.filter(ImageFilter.GaussianBlur(lado * 0.06))
    x = rng.randint(0, max(0, a.width - lado))
    y = rng.randint(0, max(0, a.height - lado))
    a.paste(corte, (x, y), mascara)
    buf = io.BytesIO()
    a.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def experimento_local(n: int, zip_path: str) -> dict:
    reais = fotos_reais(zip_path, n)
    gerados = patches_gerados(n)
    intactas = [forensics.analyze(b, ".jpg")["signals"].get("dispersao_blocos", {}) for b in reais]
    ganhos = [forensics.analyze(colar(b, g), ".jpg")["signals"].get("dispersao_blocos", {})
              for b, g in zip(reais[:len(gerados)], gerados, strict=False)]
    splice = [forensics.analyze(colar(b, o), ".jpg")["signals"].get("dispersao_blocos", {})
              for b, o in zip(reais, reais[n // 2:] + reais[:n // 2], strict=False)]
    saida = {"n_intactas": len(intactas), "n_gerado": len(ganhos), "n_splice": len(splice)}
    for chave in ("sigma_cv", "sigma_destaque", "hf_mad", "hf_destaque", "grade_cv"):
        neg = [d[chave] for d in intactas if d.get(chave) is not None]
        pos_g = [d[chave] for d in ganhos if d.get(chave) is not None]
        pos_s = [d[chave] for d in splice if d.get(chave) is not None]
        if len(neg) < 20 or len(pos_g) < 20:
            continue
        p95 = float(np.percentile(neg, 95))
        saida[chave] = {
            "p95_intacta": round(p95, 4),
            "auc_patch_gerado": round(auc(pos_g, neg) or 0, 3),
            "auc_splice_real": round(auc(pos_s, neg) or 0, 3),
            "recall_patch_gerado": round(float(np.mean([p > p95 for p in pos_g])), 3),
            "recall_splice_real": round(float(np.mean([p > p95 for p in pos_s])), 3),
            "fp_em_foto_intacta": round(float(np.mean([x > p95 for x in neg])), 3),
        }
    return saida


def pagina(texto_valor: str) -> Image.Image:
    im = Image.new("RGB", (1240, 1754), (255, 255, 255))
    d = ImageDraw.Draw(im)
    for i in range(48):
        d.text((80, 90 + i * 32), f"Contrato n. 4471-B  ·  linha {i}  ·  valor R$ {texto_valor},00", fill=(25, 25, 25))
    return im


def experimento_documento(n: int) -> dict:
    limpos, prints = [], []
    for i in range(n):
        im = pagina(str(1000 + i))
        b = io.BytesIO(); im.save(b, "PDF")
        limpos.append(b.getvalue())
        tela = im.resize((int(im.width * 0.75), int(im.height * 0.75)), Image.BICUBIC)
        tela = tela.filter(ImageFilter.GaussianBlur(0.7))
        b2 = io.BytesIO(); tela.save(b2, "JPEG", quality=70)
        prints.append(b2.getvalue())
    acerto_pdf = sum(1 for b in limpos
                     if any(f["question"] == "is_document" and f["score"] > 0.5
                            for f in document.analyze(b, "pdf", "d.pdf")["findings"])) / len(limpos)
    det_print = sum(1 for b in prints
                    if any(f["question"] == "is_document" and f["score"] > 0.5
                           for f in document.analyze(b, "image", "p.jpg")["findings"])) / len(prints)
    eofs_limpos = [document._pdf(b)["incremental_updates"] for b in limpos]
    return {"paginas": len(limpos), "reconhecido_como_documento": round(acerto_pdf, 3),
            "print_reconhecido_como_documento": round(det_print, 3),
            "eof_medio_pdf_novo": round(float(np.mean(eofs_limpos)), 2)}


def experimento_pdf_incremental(n: int) -> dict:
    """PDF com atualização incremental DE VERDADE.

    `PdfWriter.write` num arquivo aberto em modo append anexa ao arquivo existente, que é como um
    editor altera PDF sem reescrever — e é o que deixa o rastro de %%EOF repetido. Reescrever o
    arquivo inteiro (`clone_from` + `write` em stream novo) NÃO serve: medido, gera 1 %%EOF e não
    exercita o sinal.
    """
    from pypdf import PdfReader, PdfWriter

    novos, editados = [], []
    for i in range(n):
        b = io.BytesIO(); pagina(str(2000 + i)).save(b, "PDF")
        novos.append(b.getvalue())
        tmp = Path(tempfile.mkdtemp(prefix="aiwd-pdf-")) / "editado.pdf"
        tmp.write_bytes(b.getvalue())
        with tmp.open("ab") as f:
            escritor = PdfWriter()
            escritor.append(PdfReader(str(tmp)))
            escritor.add_blank_page(width=200, height=200)
            escritor.write(f)
        editados.append(tmp.read_bytes())
    eof_novos = [document._pdf(b)["incremental_updates"] for b in novos]
    eof_edit = [document._pdf(b)["incremental_updates"] for b in editados]
    detectados = sum(1 for e in eof_edit if e > 1)
    return {"eof_novo": round(float(np.mean(eof_novos)), 2),
            "eof_editado": round(float(np.mean(eof_edit)), 2),
            "detectados_como_editados": f"{detectados}/{len(eof_edit)}", "n": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--reais", default=str(Path(__file__).resolve().parents[1] / "bench" / "reals" / "val2017.zip"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "bench" / "tamper.json"))
    args = ap.parse_args()

    resultado = {
        "local": experimento_local(args.n, args.reais),
        "documento": experimento_documento(min(20, args.n // 5 or 4)),
        "pdf_incremental": experimento_pdf_incremental(min(20, args.n // 5 or 4)),
    }
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    Path(args.out).write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
