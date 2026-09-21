"""Verificador de SynthID-Text — port fiel do detector oficial (Apache-2.0, Google DeepMind).

Diferente do SynthID de imagem, o de TEXTO tem código ABERTO: `google-deepmind/synthid-text`
traz `hashing_function.py`, `detector_mean.py` e a derivação de g-values. Este módulo porta a
parte de VERIFICAÇÃO para inteiros puros (sem torch, sem jax, sem numpy).

Fidelidade ao original, ponto a ponto:
  - acumulador: h = (h + data) * 6364136223846793005 + 1, com wrap-around de int64 (o original
    roda em torch.long, então o estouro é parte do algoritmo — aqui emulado com máscara de 64 bits
    e sinal reconstruído, porque `>>` em Python é aritmético como no torch);
  - IV: sha256(bytes dos inteiros da chave) % 2^63-1;
  - g-values: 12 aplicações de acumulador com [1], deslocando 5 bits a cada passo, e no fim
    (h >> 30) % 2, por profundidade.

VALIDADO pelo critério do próprio repositório oficial (o teste deles não fixa valores, fixa
propriedade estatística): sequência aleatória dá média de g ≈ 0.5; sequência construída para
ficar toda em g=1 dá ≈ 1.0; e a MESMA sequência marcada com a chave A, medida com a chave B,
volta a ≈ 0.5. Ver tests/selfcheck.py.

TETO, e ele é duro: a marca é definida sobre os TOKEN IDS do tokenizador do fornecedor. Sem a
chave secreta E o tokenizador exato, detecção cega de texto de terceiro é impossível — não é
limitação desta implementação, é do esquema (a própria doc do repo assume isso).
"""

import hashlib

MULTIPLICADOR = 6364136223846793005
INCREMENTO = 1
MASCARA = (1 << 64) - 1
SINAL = 1 << 63
NGRAM_LEN = 5
PROFUNDIDADE = 30
APLICACOES = 12


def _somar(h: int, x: int) -> int:
    """Soma com wrap de int64 e sinal reconstruído (torch.long estoura, e o estouro é parte)."""
    h = (h + x) & MASCARA
    return h - (1 << 64) if h & SINAL else h


def acumular(h: int, dados: list[int]) -> int:
    for x in dados:
        h = _somar(_somar(h, x) * MULTIPLICADOR, INCREMENTO)
    return h


def iv_da_chave(chaves: list[int]) -> int:
    bruto = b"".join(int(k).to_bytes(8, "little", signed=True) for k in chaves)
    return int.from_bytes(hashlib.sha256(bruto).digest(), "big") % ((1 << 63) - 1)


def _gvals_de_ngram(ngram: list[int], iv: int, chaves: list[int]) -> list[int]:
    """g-values de um ngram, uma por profundidade — mesma ordem de operações do original."""
    saida = []
    for chave in chaves:
        h = acumular(iv, list(ngram) + [chave])
        for _ in range(APLICACOES):
            h = acumular(h, [1]) >> (64 // APLICACOES)
        saida.append((h >> 30) % 2)
    return saida


def g_values(tokens: list[int], chaves: list[int], ngram_len: int = NGRAM_LEN) -> list[list[int]]:
    """Matriz (num_ngrams, profundidade) de g-values, como `compute_g_values` do original."""
    if len(tokens) < ngram_len:
        return []
    iv = iv_da_chave(chaves)
    return [_gvals_de_ngram(tokens[i:i + ngram_len], iv, chaves) for i in range(len(tokens) - ngram_len + 1)]


def media(gvals: list[list[int]]) -> float | None:
    if not gvals:
        return None
    valores = [v for linha in gvals for v in linha]
    return sum(valores) / len(valores) if valores else None


def media_ponderada(gvals: list[list[int]]) -> float | None:
    """Pesos lineares decrescentes de 10 a 1, como `weighted_mean_score` do original."""
    if not gvals:
        return None
    prof = len(gvals[0])
    pesos = [10 - (10 - 1) * i / (prof - 1) for i in range(prof)] if prof > 1 else [10.0]
    escala = prof / sum(pesos)
    pesos = [p * escala for p in pesos]
    total = sum(gvals[linha][i] * pesos[i] for linha in range(len(gvals)) for i in range(prof))
    return total / (prof * len(gvals))


CHAVES_DEMO = [654, 400, 836, 1212, 1480, 1866, 2248, 2548, 2840, 3134, 3388, 3678,
               3952, 4236, 4504, 4778, 5056, 5334, 5606, 5878]


def marcar(n: int, vocab: int, chaves: list[int], semente: int = 0, candidatos: int = 300) -> list[int]:
    """Watermarker MÍNIMO, para teste — sem ele não há como provar que o verificador dispara.

    Em vez de amostrar de um modelo de linguagem (exigiria torch e um LM), escolhe gulosamente,
    a cada posição, o token cujos g-values têm mais 1s sob a chave. É o mesmo EFEITO do
    processador de logits oficial, que multiplica a probabilidade do token por (1 + g − g_mass).
    """
    import random

    rng = random.Random(semente)
    iv = iv_da_chave(chaves)
    tokens = [rng.randrange(vocab) for _ in range(ngram_len_ou_padrao() - 1)]
    while len(tokens) < n:
        contexto = tokens[-(ngram_len_ou_padrao() - 1):]
        melhor, maior = None, -1
        for cand in rng.sample(range(vocab), min(candidatos, vocab)):
            nota = sum(_gvals_de_ngram(contexto + [cand], iv, chaves))
            if nota > maior:
                melhor, maior = cand, nota
        tokens.append(melhor)
    return tokens


def ngram_len_ou_padrao() -> int:
    return NGRAM_LEN


def verificar(tokens: list[int], chaves: list[int]) -> dict:
    """Score de watermark para uma sequência de token ids sob uma chave.

    Devolve média, média ponderada e veredito grosseiro. Piso de confiabilidade: com menos de
    ~100 tokens a média é ruidosa (o próprio paper do SynthID-Text usa piso de 100).
    """
    if not chaves:
        return {"status": "sem_chave", "media": None, "media_ponderada": None, "n_tokens": len(tokens)}
    gvals = g_values(tokens, chaves)
    m, mp = media(gvals), media_ponderada(gvals)
    confiavel = len(tokens) >= 100
    veredito = "indeterminado"
    if m is not None and confiavel:
        veredito = "marcado" if m > 0.6 else ("sem_marca" if m < 0.4 else "indeterminado")
    return {"status": "ok", "media": m, "media_ponderada": mp, "n_tokens": len(tokens),
            "n_ngrams": len(gvals), "confiavel": confiavel, "veredito": veredito}
