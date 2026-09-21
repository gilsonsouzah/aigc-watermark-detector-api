"""A/B: quem decide melhor, a agregação determinística ou o Jev como juiz final?

Braço A (determinístico): pipeline RODANDO OFFLINE (Jev desligado). O veredito sai da
agregação das camadas por questão, com os pesos.

Braço B (Jev juiz): as MESMAS evidências locais, com score e peso de cada camada em cada
questão, vão como estado para o Jev, que responde a decisão final. Isola exatamente a
pergunta "quem decide", sem trocar a evidência entre os braços.

Métricas: acurácia balanceada, recall em imagem de gerador, falso positivo em foto real,
e em quantos casos o Jev discordou do determinístico.

Uso:  python -m tools.measure_jev_judge [--n-fakes 15] [--n-reais 60]
"""

import argparse
import json
import zipfile
from pathlib import Path

from aiwd.config import Config
from aiwd.gateway import Gateway, boolean
from aiwd.pipeline import detect

POSITIVO = {"ai", "likely_ai"}

INSTRUCOES = (
    "Você é o juiz final de um detector de conteúdo gerado por IA. Recebe as evidências já "
    "extraídas, com o score e o peso de cada camada dentro de cada questão. Decida.\n"
    "Regras: (1) marcador determinístico de procedência (Content Credentials, chunk de "
    "gerador, rótulo legal) é assinatura, não indício — se existir, a resposta é sim; "
    "(2) evidência estatística fraca que aparece em VÁRIAS camadas independentes soma, não "
    "se dilui; (3) se as camadas se contradizem sem determinístico, diga que não sabe em vez "
    "de escolher o lado da maioria; (4) pergunta sobre documento (print/scan) e alteração "
    "local são perguntas distintas de 'gerado por IA' — não confunda geração total com "
    "edição de um pedaço."
)


def estado_para_juiz(r: dict) -> dict:
    return {
        "arquivo": {"nome": r["filename"], "tipo": r["kind"]},
        "descricao_forense_da_imagem": r.get("description"),
        "questoes": [{
            "id": q["id"], "pergunta": q["question"], "resposta_da_agregacao": q["answer"],
            "score_agregado": q["score"], "confianca": q["confidence"], "confiabilidade": q["reliability"],
            "camadas": [{"camada": l["layer"], "score": l["score"], "peso": l["weight"],
                         "confiabilidade": l["reliability"], "evidencia": l["evidence"][:3]}
                        for l in q["layers"]],
        } for q in r["questions"]],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fakes", default=str(Path(__file__).resolve().parents[1] / "bench" / "gen"))
    ap.add_argument("--reais", default=str(Path(__file__).resolve().parents[1] / "bench" / "reals" / "val2017.zip"))
    ap.add_argument("--n-fakes", type=int, default=15)
    ap.add_argument("--n-reais", type=int, default=60)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "bench" / "jev_judge.json"))
    args = ap.parse_args()

    cfg = Config.from_env()
    juiz = Gateway(cfg)
    offline = Config(key=None)

    with zipfile.ZipFile(args.reais) as z:
        reais = [z.read(n) for n in z.namelist() if n.endswith(".jpg")][:args.n_reais]
    fakes = {d.name: [p.read_bytes() for p in sorted(d.iterdir())[:args.n_fakes]]
             for d in sorted(Path(args.fakes).iterdir()) if d.is_dir()}

    amostras = [("real", f"coco_{i}.jpg", b) for i, b in enumerate(reais)]
    amostras += [(gen, f"{gen}_{i}.png", b) for gen, blobs in fakes.items() for i, b in enumerate(blobs)]
    print(f"{len(amostras)} arquivos ({len(reais)} reais, {len(amostras)-len(reais)} gerados)")

    linhas = []
    for rotulo, nome, blob in amostras:
        local = detect(blob, nome, offline)          # braço A: sem Jev
        estado = estado_para_juiz(local)
        try:
            respostas = juiz.evaluate(estado, {"decisao": boolean(INSTRUCOES)})
            p_juiz = respostas.get("decisao", {}).get("probability")
        except Exception as e:
            print(f"  erro no juiz em {nome}: {type(e).__name__}: {e}")
            p_juiz = None
        linhas.append({"rotulo": rotulo, "arquivo": nome,
                       "p_local": local["ai_probability"], "verdict_local": local["verdict"],
                       "p_juiz": p_juiz,
                       "verdict_juiz": None if p_juiz is None else ("sim" if p_juiz >= 0.5 else "não")})
        marca = " " if p_juiz is None else ("=" if (local["verdict"] in POSITIVO) == (p_juiz >= 0.5) else "!")
        print(f"  {marca} {nome[:32]:<32} local={local['verdict']:<12} p={local['ai_probability']}  "
              f"juiz={p_juiz if p_juiz is None else round(p_juiz,3)}", flush=True)

    def metrica(chave_p, chave_v):
        pos = [x for x in linhas if x["rotulo"] != "real" and x.get(chave_p) is not None]
        neg = [x for x in linhas if x["rotulo"] == "real" and x.get(chave_p) is not None]
        recall = sum(1 for x in pos if x[chave_v] in POSITIVO) / len(pos) if pos else None
        fp = sum(1 for x in neg if x[chave_v] in POSITIVO) / len(neg) if neg else None
        return {"recall": round(recall, 3) if recall is not None else None,
                "falso_positivo": round(fp, 3) if fp is not None else None,
                "acuracia_balanceada": round((recall + 1 - fp) / 2, 3) if recall is not None and fp is not None else None}

    def veredito_do_juiz(x):
        if x["p_juiz"] is None:
            return None
        return "ai" if x["p_juiz"] >= 0.65 else ("likely_ai" if x["p_juiz"] >= 0.5 else "inconclusive")

    for x in linhas:
        x["verdict_juiz"] = veredito_do_juiz(x)

    a, b = metrica("p_local", "verdict_local"), metrica("p_juiz", "verdict_juiz")
    divergem = sum(1 for x in linhas if x["p_juiz"] is not None
                   and (x["verdict_local"] in POSITIVO) != (x["verdict_juiz"] in POSITIVO))
    resultado = {"n": len(linhas), "deterministico": a, "jev_juiz": b,
                 "discordancias": divergem, "linhas": linhas}
    print(f"\nA) agregação determinística : {a}")
    print(f"B) Jev como juiz final      : {b}")
    print(f"discordâncias entre os dois : {divergem}/{len(linhas)}")
    Path(args.out).write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
