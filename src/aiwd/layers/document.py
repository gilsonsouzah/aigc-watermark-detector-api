"""Camada de documento: é documento? é print/scan de documento? foi alterado localmente?

Motivo de existir separado: para documento a pergunta muda. "Isto foi gerado por IA?" quase
não importa — o que importa é (a) isto é um print, foto de tela ou scan de documento, e
(b) o conteúdo foi alterado depois (valor, data, assinatura, texto). São perguntas
diferentes, com sinais diferentes, e o detector de imagem sintética responde mal as duas.

Sinais aqui são estruturais e determinísticos, não estatísticos:

  fundo claro dominante + baixa saturação        -> página
  linhas de texto (projeção de gradiente em faixas) -> documento com texto
  razão de aspecto de página (A4/Letter)         -> documento escaneado
  grade de tela (moiré) ou faixas uniformes      -> print/foto de tela, não arquivo
  múltiplos %%EOF / XObjects no PDF              -> PDF salvo incrementalmente várias vezes
"""

import io
import re

import numpy as np
from PIL import Image

from ..questions import finding

# Fração máxima de pixels de tinta para ainda ser página. Medido em 60 fotos reais: as que
# passavam pela heurística de linha tinham 4.2%-28.9% de tinta; páginas ficam em 1.3%-2.7%.
PAGINA_TINTA_MAX = 0.035

# razões de página aceitas (retrato ou paisagem)
Razoes_PAGINA = ((1.414, "A4"), (1.294, "Letter"), (1.333, "4:3"), (1.5, "3:2"))


def _pagina(gray: np.ndarray) -> dict:
    """Detecta ESTRUTURA DE TEXTO, não brilho.

    Duas tentativas anteriores falharam, as duas medidas:
      1. limiar absoluto (escuro < 100, claro > 200): funciona na página renderizada e falha no
         mesmo documento reescalado/recomprimido — o texto afina e clareia;
      2. percentil p5/p95: falha porque tinta de texto ocupa MENOS de 5% da área, então p5 cai
         no papel e o contraste medido foi 9 em vez de ~200. Pior: 3 de 6 fotos reais passaram.

    O que separa página de foto não é brilho, é a linha: numa página de texto existem faixas
    horizontais que são ~só papel com um pouco de tinta (as linhas de texto). Numa fotografia
    nenhuma faixa é simultaneamente quase toda clara e com um pouco de escuro.
    """
    tinta = float(np.percentile(gray, 1))
    papel = float(np.percentile(gray, 99))
    contraste = papel - tinta
    if contraste < 40:
        return {"fracao_clara": round(float((gray > papel - 10).mean()), 3), "fracao_escura": 0.0,
                "linhas_de_texto": 0, "bimodal": False, "contraste": round(contraste, 1)}
    e_tinta = gray < (tinta + 0.35 * contraste)
    e_papel = gray > (tinta + 0.65 * contraste)
    frac_tinta_linha = e_tinta.mean(axis=1)          # por linha: fração de tinta
    frac_papel_linha = e_papel.mean(axis=1)          # por linha: fração de papel
    # linha de texto = quase toda papel, com um pouco de tinta (nem vazia, nem cheia)
    linhas_de_texto = int(((frac_papel_linha > 0.55) & (frac_tinta_linha > 0.01)
                           & (frac_tinta_linha < 0.55)).sum())
    return {"fracao_clara": round(float(e_papel.mean()), 3),
            "fracao_escura": round(float(e_tinta.mean()), 3),
            "linhas_de_texto": linhas_de_texto,
            # A fração de tinta é o que separa, medido: página 1.3-2.7%, foto 4.2-28.9%.
            # `linhas_de_texto` sozinho NÃO separa (foto chegou a 326 "linhas de texto" contra
            # 256 de uma página) — por isso ele é só pré-requisito, não o critério.
            "bimodal": linhas_de_texto > 8 and float(e_tinta.mean()) < PAGINA_TINTA_MAX,
            "contraste": round(contraste, 1)}


def _razao_page(w: int, h: int) -> str | None:
    r = max(w, h) / max(1, min(w, h))
    return next((nome for alvo, nome in Razoes_PAGINA if abs(r - alvo) < 0.03), None)


def analyze(data: bytes, kind: str, filename: str = "", ocr_signals: dict | None = None) -> dict:
    sinais: dict = {"kind": kind}
    findings: list[dict] = []
    notas: list[str] = []

    if kind == "pdf":
        pdf = _pdf(data)
        sinais["pdf"] = pdf
        if pdf["campos_vazios"]:
            sinais["pdf_metadata_apagado"] = True

    if kind == "pdf":
        _findings_pdf(sinais["pdf"], findings, notas)
        # um PDF É um documento por definição: emitir a resposta sem depender de abrir a página.
        # (antes desta correção a análise de página nunca rodava em PDF, porque PDF não abre com
        # Image.open — medido: 0/16 páginas reconhecidas na bancada)
        findings.append(finding("is_document", 1.0, "deterministic",
                                [f"container PDF com {sinais['pdf']['paginas']} página(s)"
                                 + (" e imagem embutida" if sinais['pdf']['imagens'] else ", sem imagem embutida")],
                                answer="sim"))

    try:
        img = Image.open(io.BytesIO(data)).convert("L")
    except Exception:
        if kind != "pdf":
            return {"name": "document", "status": "skipped", "findings": [], "signals": sinais,
                    "notes": ["não decodifica como imagem"]}
        return {"name": "document", "status": "ok", "findings": findings, "signals": sinais, "notes": notas}

    w, h = img.size
    lado = min(1024, min(img.size))
    esq, topo = (w - lado) // 2, (h - lado) // 2
    gray = np.asarray(img.crop((esq, topo, esq + lado, topo + lado)), dtype=np.float32)
    pag = _pagina(gray)
    sinais.update(pag)
    razao = _razao_page(w, h)
    sinais["razao_pagina"] = razao

    # OCR é o detector de documento mais robusto que temos, e é MEDIDO: numa FOTO de carteira de
    # identidade (cartão verde sobre mesa de madeira) o critério de página falhou (is_document 0.14)
    # porque os limiares de papel/tinta foram calibrados em PÁGINA renderizada, não em foto de
    # documento. O OCR leu o documento inteiro — 20+ fragmentos — e isso não depende de fundo,
    # cor do cartão nem perspectiva.
    n_textos_ocr = int((ocr_signals or {}).get("n_textos") or 0)
    if n_textos_ocr >= 8:
        rotulos = (ocr_signals or {}).get("textos") or []
        findings.append(finding("is_document", 0.9, "strong",
                                [f"OCR leu {n_textos_ocr} fragmentos de texto na imagem "
                                 f"(ex.: {rotulos[0][:60]!r}), padrão de documento"], answer="sim"))
        notas.append("documento reconhecido pelo OCR, não pela heurística de página")

    eh_documento = pag["bimodal"] and pag["fracao_clara"] > 0.35
    if eh_documento:
        findings.append(finding("is_document", 0.85 if pag["linhas_de_texto"] > 20 else 0.7, "strong",
                                [f"página com fundo claro dominante ({pag['fracao_clara']:.0%}) e "
                                 f"{pag['linhas_de_texto']} faixas de texto"
                                 + (f"; proporção de {razao}" if razao else "")],
                                answer="sim"))
        notas.append("documento detectado: as perguntas de geração por IA valem pouco aqui; "
                     "o que importa é se é print/scan e se houve alteração local")
    else:
        findings.append(finding("is_document", 0.15, "strong", ["sem estrutura de página (fundo claro "
                                "dominante + faixas de texto)"], answer="não"))
    if eh_documento and pag["linhas_de_texto"] > 30 and pag["fracao_escura"] > 0.03:
        findings.append(finding("screenshot_or_reencoded", 0.6, "weak",
                                ["estrutura de grade de texto densa: típico de print de página, "
                                 "não de arquivo original"]))
    return {"name": "document", "status": "ok", "findings": findings, "signals": sinais, "notes": notas}


FONTES_DE_ESCRITORIO = ("Liberation", "DejaVu", "Nimbus", "Carlito", "Caladea", "FreeSans")


def _campo(data: bytes, chave: bytes) -> str | None:
    """Valor de uma chave de /Info. Devolve '' quando existe mas está vazia (o caso interessante)."""
    m = re.search(rb"/" + chave + rb"\s*\(([^)]*)\)", data)
    if m is None:
        return None
    return m.group(1).decode("latin-1", "replace")


def _findings_pdf(pdf: dict, findings: list[dict], notas: list[str]) -> None:
    """Sinais estruturais de PDF. Nenhum deles prova geração por IA — provam COMO o arquivo foi
    escrito, e o conjunto é o que separa biblioteca de script de editor de escritório."""
    if pdf["campos_vazios"] or not pdf["ferramentas_declaradas"]:
        pistas = []
        if pdf["campos_vazios"]:
            pistas.append(f"campos de ferramenta presentes e vazios ({', '.join(pdf['campos_vazios'])})")
        else:
            pistas.append("nenhuma ferramenta declarada no /Info (editores reais sempre assinam)")
        if pdf["streams_nao_comprimidos"]:
            pistas.append("nenhum stream comprimido — todo editor real usa FlateDecode")
        if not pdf["xref_stream"] and not pdf["objetos_em_stream"]:
            pistas.append(f"xref clássico e sem object streams ({pdf['versao']}), estilo de biblioteca")
        # fonte de escritório NÃO entra como pista: medido em 30 PDFs reais, é True nos dois
        # lados (todo writer em Linux embute Liberation) — não discrimina nada
        if pdf["chave_trapped"]:
            pistas.append("chave /Trapped setada à mão")
        # ESPECIFICIDADE MEDIDA, SENSIBILIDADE NÃO. Em 86 PDFs reais do mundo real (contratos,
        # apólices, faturas, certificados, prints de Chrome/Skia, iText, Amdocs, openhtmltopdf),
        # a combinação "sem ferramenta declarada + stream não comprimido" apareceu 0 vez → teto de
        # falso positivo 3,5% a 95% (regra de três). Do outro lado há exatamente 1 PDF sabidamente
        # gerado por IA, então o score é estimativa guiada por especificidade, não probabilidade
        # calibrada. Cada condição isolada existe no mundo real (a fatura da Amdocs tem stream cru;
        # 3 contratos não declaram ferramenta) — é a COMBINAÇÃO que é rara.
        forte = pdf["streams_nao_comprimidos"] and (pdf["campos_vazios"] or not pdf["ferramentas_declaradas"])
        findings.append(finding("is_ai_generated", 0.85 if forte else 0.6, "strong",
                                ["PDF sem identificação de ferramenta: " + "; ".join(pistas)
                                 + ". Conjunto típico de arquivo montado por script/biblioteca, "
                                   "não de editor de escritório — indício, não prova de IA"]))
    if not pdf["c2pa"]:
        findings.append(finding("has_provenance_marker", 0.05, "weak",
                                ["PDF sem Content Credentials (C2PA)"]))
    if pdf["incremental_updates"] > 1:
        findings.append(finding("locally_manipulated", 0.6, "weak",
                                [f"PDF com {pdf['incremental_updates']} atualizações incrementais: "
                                 "o arquivo foi alterado e salvo por cima, sem reescrever"]))
    if pdf["produtores_distintos"] > 1:
        findings.append(finding("locally_manipulated", 0.65, "weak",
                                ["PDF com múltiplos produtores/autores declarados: alterado por "
                                 "software diferente do que criou"]))
    if pdf["imagens"] == 0:
        notas.append("PDF sem imagem embutida: as camadas de pixel não têm o que analisar; "
                     "o veredito sai da estrutura do container")


def _pdf(data: bytes) -> dict:
    """Forense estrutural de PDF: quando o metadata é apagado, sobra como o arquivo foi escrito.

    O que separa PDF escrito por biblioteca de script (típico de documento gerado por IA, onde
    um agente escreve um script que monta o arquivo) de PDF de editor de escritório não é o
    conteúdo: é (a) ferramenta se identificando, (b) streams comprimidos ou não, (c) estilo de
    xref, (d) quais fontes foram embutidas.
    """
    eofs = len(re.findall(rb"%%EOF", data))
    produtor = _campo(data, b"Producer")
    criador = _campo(data, b"Creator")
    autor = _campo(data, b"Author")
    ferramentas = [v for v in (produtor, criador, autor) if v]
    vazias = [v for v in (produtor, criador, autor) if v == ""]
    filtros = sorted({m.decode() for m in re.findall(rb"/Filter\s*/(\w+)", data)})
    fontes = sorted({m.decode() for m in re.findall(rb"/BaseFont\s*/([A-Za-z0-9+\-]+)", data)})
    return {
        "incremental_updates": eofs,
        "imagens": len(re.findall(rb"/Subtype\s*/Image", data)),
        "fontes": sorted({f.split("+")[-1] for f in fontes})[:6],
        "fontes_embutidas": b"/FontFile" in data,
        "fonte_de_escritorio": any(f.split("+")[-1].startswith(FONTES_DE_ESCRITORIO) for f in fontes),
        "paginas": len(re.findall(rb"/Type\s*/Page\b", data)),
        "versao": data[:8].decode("latin-1", "replace").strip(),
        "ferramentas_declaradas": ferramentas,
        "campos_vazios": vazias or None,
        "streams_nao_comprimidos": not filtros and b"/ObjStm" not in data,
        "filtros": filtros,
        "xref_stream": b"/Type /XRef" in data or b"/Type/XRef" in data,
        "objetos_em_stream": b"/ObjStm" in data,
        "chave_trapped": b"/Trapped" in data,
        "c2pa": any(m in data for m in (b"c2pa", b"jumbf", b"urn:content-credentials")),
        "produtores_distintos": len({v for v in ferramentas if v}),
    }
