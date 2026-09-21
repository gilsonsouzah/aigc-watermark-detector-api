"""Descrição forense da imagem para alimentar o avaliador text-only (Jev).

Por que não "descreva a imagem": legenda genérica ("um homem de casaco branco") não ajuda a
decidir nada. O que ajuda é descrição **de atributo**, pedida como rubrica — os traços que o
processo generativo deixa escapar e que um humano não produz do mesmo jeito:

  uniformidade de tecido/dobra · lisura de pele e cabelo · anatomia de mãos e dedos ·
  legibilidade de texto · coerência de sombra e reflexo · repetição e simetria do fundo ·
  profundidade de campo · bordas e recorte · plausibilidade física (peso, apoio, perspectiva)

O descritor é plugável e opcional (`AIGC_VLM_URL`): recebe os bytes da imagem, devolve texto.
Sem descritor configurado, nada muda — o Jev segue recebendo só as features numéricas.
Medir antes de confiar: `tools/measure_description.py` compara veredito com e sem descrição.
"""

import os

import httpx

RUBRICA = """Descreva a imagem APENAS nos pontos abaixo, em uma linha cada, e depois liste
anomalias. Não escreva legenda genérica nem opinião sobre a pessoa ou o assunto.

1. Enquadramento e composição (proporção, centralização, o que está em primeiro/segundo plano).
2. Luz: direção, dureza, se a luz é coerente com as sombras projetadas.
3. Profundidade de campo e desfoque de fundo: natural ou uniforme/artificial?
4. Pele, cabelo e tecido: textura, poros, fios soltos, rugas de tecido — lisos demais? repetidos?
5. Dobras e volumes: o tecido cede como matéria real? há peso e gravidade?
6. Mãos, dedos, dentes, olhos e orelhas: contagem, articulação e simetria corretas?
7. Texto, logotipo ou número visível na imagem: transcreva e diga se está legível ou deformado.
8. Fundo: elementos repetidos, incompletos, fundidos ou com perspectiva impossível?
9. Reflexos, sombras e bordas: coerentes com a cena? recorte limpo ou halo?
10. Qualquer artefato que indique imagem sintetizada (grade, suavização plástica, saturação
    excessiva em uma região, transição abrupta).

Depois, a linha final: `artefatos: <nenhum|leves|claros|evidentes> — <o que mais chama atenção>`."""

MAX_CARACTERES = 4_000


def disponivel() -> bool:
    return bool(os.environ.get("AIGC_VLM_URL"))


def descrever(data: bytes) -> dict:
    """-> {"status", "descricao", "endpoint", "erro"}. Nunca levanta: falha vira status."""
    url = os.environ.get("AIGC_VLM_URL")
    if not url:
        return {"status": "not_configured", "descricao": None, "endpoint": None}
    try:
        r = httpx.post(url, content=data, timeout=float(os.environ.get("AIGC_VLM_TIMEOUT", "120")),
                       headers={"content-type": "application/octet-stream",
                                "x-prompt": RUBRICA})
        r.raise_for_status()
        corpo = r.json()
        texto = corpo.get("description") or corpo.get("text") or corpo.get("choices", [{}])[0].get("message", {}).get("content")
        if not texto:
            return {"status": "error", "descricao": None, "endpoint": url,
                    "erro": f"resposta sem descrição: {str(corpo)[:200]}"}
        return {"status": "ok", "descricao": str(texto)[:MAX_CARACTERES], "endpoint": url}
    except Exception as e:
        return {"status": "error", "descricao": None, "endpoint": url,
                "erro": f"{type(e).__name__}: {e}"[:200]}


def prompt() -> str:
    return RUBRICA
