"""Calibra o detector neural: acha o corte real e grava model_calibration.json.

Motivo, medido: o modelo tem AUC 0.977-0.999 mas o sigmoide CRU não é probabilidade calibrada —
imagem do imagen-4 dá 0.9733 no original e 0.0093 depois de reescalonada+JPEG, e 0.0001 em outra
imagem do mesmo gerador. A AUC fica alta porque o ranking se mantém (os gerados continuam acima
das fotos, que ficam em ~0.0000), mas no limiar 0.5 a agregação classifica esses casos como
"não IA". Foi exatamente o que derrubou o caso do print para `likely_human`.

O que este script faz: mede as distribuições por regime, acha o corte que maximiza
sensibilidade+especificidade-1, e grava em JSON que a camada passa a usar.

Uso:  bench/run.sh python tools/calibrate_model.py --n-reais 200
"""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

from aiwd.layers.image_model import preparar
from tools.measure import degradar

RAIZ = Path(__file__).resolve().parents[1]


EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def _aplicar(scores, grade, tabela_p):
    """Interpola a probabilidade no eixo log10(escore)."""
    x = np.log10(np.clip(np.asarray(scores, dtype=float), 1e-8, 1.0))
    return np.interp(x, grade, tabela_p)


def _mapear(pos, neg, bins: int = 48):
    """f(escore|IA) e f(escore|real) por histograma no eixo log10 -> probabilidade posterior."""
    xp = np.log10(np.clip(np.asarray(pos, dtype=float), 1e-8, 1.0))
    xn = np.log10(np.clip(np.asarray(neg, dtype=float), 1e-8, 1.0))
    lo = min(xp.min(), xn.min()) - 0.2
    hi = max(xp.max(), xn.max()) + 0.2
    grade = np.linspace(lo, hi, bins)
    fp, _ = np.histogram(xp, bins=grade)
    fn, _ = np.histogram(xn, bins=grade)
    alfa = 0.5                                     # Laplace: evita p=0/1 em bin vazio
    centros = (grade[:-1] + grade[1:]) / 2
    p = (fp + alfa) / (fp + fn + 2 * alfa)
    return centros, np.clip(p, 1e-4, 1 - 1e-4)


def _platt(pos, neg, iteracoes: int = 3000, taxa: float = 0.5):
    """Ajusta a,b de sigmoid(a*logit(s)+b) por gradiente descendente. Sem sklearn."""
    x = np.concatenate([_logit(pos), _logit(neg)])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    a, b = 1.0, 0.0
    for _ in range(iteracoes):
        erro = _sig(a * x + b) - y
        ga = float((erro * x).mean()) + 1e-3 * a      # L2 leve evita a explodir
        gb = float(erro.mean())
        a -= taxa * ga
        b -= taxa * gb
    return a, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=str(RAIZ / "bench" / "models" / "model.onnx"))
    ap.add_argument("--n-reais", type=int, default=200)
    ap.add_argument("--out", default=str(RAIZ / "model_calibration.json"))
    ap.add_argument("--regimes", nargs="+", default=["limpo", "jpeg75", "resize50", "screenshot"])
    ap.add_argument("--n-fakes", type=int, default=0)   # 0 = todos
    args = ap.parse_args()

    sess = ort.InferenceSession(args.modelo, providers=["CPUExecutionProvider"])
    ent = sess.get_inputs()[0]

    def prob(data: bytes) -> float | None:
        x = preparar(data)
        if x is None:
            return None
        v = float(np.asarray(sess.run(None, {ent.name: x})[0]).reshape(-1)[0])
        return float(1 / (1 + np.exp(-v)))

    with zipfile.ZipFile(RAIZ / "bench" / "reals" / "val2017.zip") as z:
        reais = [z.read(n) for n in z.namelist() if n.endswith(".jpg")][:args.n_reais]
    fakes = [p.read_bytes() for d in sorted((RAIZ / "bench" / "gen").iterdir()) if d.is_dir()
             for p in sorted(d.iterdir())[:args.n_fakes or None]]

    saida = {"fonte": "bench/gen (360) vs bench/reals (COCO)", "modelo": Path(args.modelo).name, "regimes": {}}
    for regime in args.regimes:
        p_fake = [prob(degradar(b, regime)) for b in fakes]
        p_real = [prob(degradar(b, regime)) for b in reais]
        f = np.array([x for x in p_fake if x is not None])
        r = np.array([x for x in p_real if x is not None])
        # corte por máximo de Youden, varrendo a faixa observada (o valor absoluto é minúsculo)
        grade = np.unique(np.concatenate([f, r]))
        melhor = {"corte": 0.5, "ba": 0.0}
        for corte in grade:
            sens = float((f >= corte).mean())
            espec = float((r < corte).mean())
            ba = (sens + espec) / 2
            if ba > melhor["ba"]:
                melhor = {"corte": float(corte), "ba": round(ba, 4),
                          "sensibilidade": round(sens, 4), "especificidade": round(espec, 4)}
        # Mapeamento por DENSIDADE (Bayes), não por Platt.
        # Motivo medido: o Platt otimiza probabilidade (log-loss), não decisão — o ponto onde ele
        # cruza 0.5 não é o corte ótimo, e pior, uma foto real com escore 0.001 saía com p=0.85 e
        # virava falso positivo na agregação. Aqui estimo f(escore|IA) e f(escore|real) por
        # histograma no eixo log10 e devolvo p = f1/(f1+f0) com prior 0.5 — assim p=0.5 É o corte
        # de Bayes, e o valor tem significado de probabilidade de verdade.
        grade, tabela_p = _mapear(f, r)
        saida["regimes"][regime] = {
            "grade_log10": [round(float(x), 5) for x in grade],
            "probabilidade": [round(float(x), 5) for x in tabela_p],
            "ba_apos_mapeamento": round(float(((_aplicar(f, grade, tabela_p) >= 0.5).mean()
                                               + (_aplicar(r, grade, tabela_p) < 0.5).mean()) / 2), 4),
            "corte": melhor["corte"], "acuracia_balanceada": melhor["ba"],
            "sensibilidade": melhor.get("sensibilidade"), "especificidade": melhor.get("especificidade"),
            "n_pos": len(f), "n_neg": len(r),
            "fake_min": round(float(f.min()), 6), "fake_mediana": round(float(np.median(f)), 6),
            "real_max": round(float(r.max()), 6), "real_mediana": round(float(np.median(r)), 6),
            "acuracia_no_limiar_0_5": round(float(((f >= 0.5).mean() + (r < 0.5).mean()) / 2), 4),
        }
        s = saida["regimes"][regime]
        print(f"{regime:<11} corte={s['corte']:<10} BA={s['acuracia_balanceada']:<7} "
              f"(no 0.5 seria {s['acuracia_no_limiar_0_5']}) | fake mediana={s['fake_mediana']} "
              f"real máx={s['real_max']}")

    Path(args.out).write_text(json.dumps(saida, indent=2, ensure_ascii=False))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
