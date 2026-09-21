"""Camada de texto: extrai o texto do arquivo e avalia se foi escrito por IA.

Por que existe: PDF de texto puro não tem imagem para as camadas de pixel e a estrutura do
container dá UM sinal só — o que faltava era ler o conteúdo.

Três fontes, e a honestidade sobre cada uma:

1. ESTILOMETRIA (`estilometria`) — features baratas (palavras por frase, razão tipo/token,
   fração de bullets, pontuação). MEDIDO (`tools/measure_text.py`) em 1 PDF gerado por IA contra
   80 PDFs reais: parece separar tudo (palavras/frase 180 vs 25; bullets 0.85 vs 0.00), mas está
   CONFUNDIDO por gênero — o positivo é uma lista de compras e os negativos são contratos e
   faturas. Isso separa lista de prosa, não IA de humano. Por isso as features vão para `signals`
   para inspeção e **não geram finding**: sem positivo e negativo do mesmo gênero, não há como
   calibrar, e um limiar escolhido aqui seria overfitting ao arquivo que eu tenho em mãos.
2. JEV avaliando "este texto é de IA?" — MEDIDO E REJEITADO (`tools/measure_text_jev.py`):
   nos 30 PDFs reais ele acusou 23 (mediana 0.545) e no PDF gerado por IA deu **0.22**. Acusa a
   maioria dos documentos reais e não pega o de IA: pior que inútil como sinal. Por isso a camada
   NÃO emite finding de "texto escrito por IA" e a pergunta correspondente foi removida do
   contrato — sem fonte válida, pergunta sem resposta só polui a saída.
3. SYNTHID-TEXT (`synthid_text`) — o verificador está implementado e provado (ver
   tests/selfcheck.py: dispara na chave certa, dá chance na errada). Mas a marca é definida sobre
   os TOKEN IDS do tokenizador do fornecedor: sem a chave E o tokenizador, detecção de texto de
   terceiro é impossível. A camada reporta `unavailable` com o motivo em vez de fingir.
"""

import io
import re
import statistics
import zipfile

MAX_CHARS = 12_000
MIN_PALAVRAS = 40


def extrair(data: bytes, kind: str, filename: str = "") -> str:
    if kind == "pdf":
        try:
            from pypdf import PdfReader

            return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
        except Exception:
            return ""
    if kind in ("docx", "pptx", "xlsx"):
        try:
            alvo = {"docx": "word/document.xml", "pptx": "ppt/slides/", "xlsx": "xl/sharedStrings.xml"}[kind]
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                partes = [z.read(n).decode("utf-8", "replace")
                          for n in z.namelist() if n.startswith(alvo) and n.endswith(".xml")]
            return re.sub(r"<[^>]+>", " ", " ".join(partes))
        except Exception:
            return ""
    if filename.lower().endswith((".txt", ".md", ".csv", ".json", ".log")):
        return data.decode("utf-8", "replace")
    return ""


def estilometria(texto: str) -> dict:
    frases = [f.strip() for f in re.split(r"[.!?]+", texto) if len(f.strip()) > 15]
    palavras = re.findall(r"\b[\wÀ-ÿ']+\b", texto.lower())
    comp = [len(f.split()) for f in frases]
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]
    return {
        "n_palavras": len(palavras),
        "media_palavras_frase": round(statistics.mean(comp), 2) if comp else None,
        "desvio_palavras_frase": round(statistics.pstdev(comp), 2) if len(comp) > 1 else None,
        "razao_tipo_token": round(len(set(palavras)) / len(palavras), 3) if palavras else None,
        "fracao_bullets": round(sum(1 for l in linhas if l[:2] in ("- ", "* ", "• ")) / len(linhas), 3)
        if linhas else None,
    }


def analyze(data: bytes, kind: str, filename: str = "") -> dict:
    texto = extrair(data, kind, filename)
    palavras = len(texto.split())
    achados: list[dict] = []
    notas: list[str] = []
    sinais: dict = {"kind": kind, "palavras_extraidas": palavras}

    if palavras < MIN_PALAVRAS:
        notas.append(f"texto extraído insuficiente ({palavras} palavras): nada a avaliar no conteúdo")
        return {"name": "text", "status": "skipped", "findings": [], "signals": sinais, "notes": notas}

    # estilometria vai para `signals` e NÃO vira finding: medida em 1 positivo (lista de compras)
    # contra 80 negativos (contratos/faturas), está confundida por gênero — separa lista de prosa,
    # não IA de humano. Sem positivo e negativo do MESMO gênero não há calibração honesta.
    sinais["estilometria"] = estilometria(texto)
    sinais["texto_previa"] = texto[:300]
    ver = _synthid_text(texto)
    sinais["synthid_text"] = ver
    if ver.get("status") != "ok":
        notas.append(f"SynthID-text não avaliado: {ver.get('motivo')}")
    return {"name": "text", "status": "ok", "findings": achados, "signals": sinais, "notes": notas}


def _synthid_text(texto: str) -> dict:
    """Verificação de marca d'água de texto.

    A marca é definida nos token ids do tokenizador do FORNECEDOR: sem chave E tokenizador, não
    há detecção possível de texto de terceiro. Aqui só rodamos quando o operador fornece ambos.
    """
    import os

    if not os.environ.get("SYNTHID_TEXT_KEYS"):
        return {"status": "sem_chave", "motivo": "SYNTHID_TEXT_KEYS não configurada (a marca é "
                                                "keyed; sem a chave do fornecedor o score cai a chance)"}
    if not os.environ.get("SYNTHID_TEXT_TOKENIZER"):
        return {"status": "sem_tokenizador",
                "motivo": "a marca é definida sobre os token ids do tokenizador do fornecedor; "
                          "sem ele os ngrams não coincidem e o score cai a chance"}
    return {"status": "indisponivel",
            "motivo": "hook de tokenizador configurado mas não implementado neste PoC"}
