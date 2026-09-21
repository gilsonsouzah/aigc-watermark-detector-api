"""Bancada de PDF: roda a camada document sobre um diretório rotulado e reporta o veredito.

Rotulagem por nome de arquivo: prefixo `IA-` é positivo, o resto é negativo (mundo real).
Sem isso não dá para saber se o sinal estrutural discrimina ou só acusa tudo.

Uso:  python -m tools.measure_pdf [--dir bench/pdfs] [--limiar 0.7]

É barato (regex em arquivos pequenos), roda fora do Docker.
"""

import argparse
from pathlib import Path

from aiwd.layers import document


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="bench/pdfs")
    ap.add_argument("--limiar", type=float, default=0.7)
    args = ap.parse_args()

    linhas = []
    for f in sorted(Path(args.dir).iterdir()):
        if f.suffix.lower() != ".pdf":
            continue
        try:
            r = document.analyze(f.read_bytes(), "pdf", f.name)
        except Exception as e:
            print(f"{f.name[:44]:<46} ERRO {type(e).__name__}: {e}")
            continue
        pdf = r["signals"].get("pdf", {})
        score = max([x["score"] for x in r["findings"] if x["question"] == "is_ai_generated"], default=None)
        linhas.append({"nome": f.name, "positivo": f.name.startswith("IA-"), "score": score,
                       "ferramenta": (pdf.get("ferramentas_declaradas") or [None])[0],
                       "stream_cru": pdf.get("streams_nao_comprimidos"),
                       "metadata_vazio": bool(pdf.get("campos_vazios")),
                       "incremental": pdf.get("incremental_updates")})

    pos = [x for x in linhas if x["positivo"]]
    neg = [x for x in linhas if not x["positivo"]]
    acusa = lambda x: x["score"] is not None and x["score"] >= args.limiar
    print(f"{'arquivo':<44} {'score':<7} {'ferramenta':<26} {'cru':<6} {'vazio':<6}")
    for x in linhas:
        print(f"{x['nome'][:42]:<44} {x['score']!s:<7} {str(x['ferramenta'] or '(nenhuma)')[:24]:<26} "
              f"{x['stream_cru']!s:<6} {x['metadata_vazio']!s:<6}{'  <-- IA' if x['positivo'] else ''}")
    print(f"\npositivos: {sum(acusa(x) for x in pos)}/{len(pos)} acusados"
          f" | negativos: {sum(acusa(x) for x in neg)}/{len(neg)} acusados (falso positivo)")
    med = [x["score"] for x in neg if x["score"] is not None]
    if med:
        print(f"negativos com score parcial (não acusam): {sorted(set(med))} em {len(med)} arquivo(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
