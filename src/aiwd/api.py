"""API HTTP. Documentação interativa em /docs (Swagger UI) e /redoc."""

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse

from .config import Config
from .pipeline import detect
from .questions import QUESTIONS

DESCRICAO = """
Estima se **uma imagem** — ou **um arquivo com imagens dentro** (PDF, DOCX, PPTX, XLSX, ZIP) —
foi gerada por IA. Devolve o veredito **e as evidências que o sustentam**, camada por camada.

---

## 1. Como ler a resposta

A resposta é hierárquica: **resultado → N questões → N camadas dentro de cada questão**.
Nenhuma camada decide sozinha: cada uma emite achados para as questões sobre as quais ela sabe
algo, e a questão é resolvida combinando esses achados.

```
verdict ................. resumo em uma palavra (ver §2)
ai_probability .......... probabilidade calibrada de ser IA, 0 a 1
questions[] ............. as 8 perguntas (§4), cada uma com as camadas que opinaram
  └ layers[] ............ quem opinou, com score, weight, reliability e a evidência em texto
layers[] ................ evidência BRUTA de cada camada (§5): textos lidos, espectro, dispersão
warnings[] .............. ressalvas — leia sempre (inclui aviso de veredito frágil)
```

**Leia `questions` junto com `verdict`**: o veredito é o resumo, a evidência é a resposta de
verdade. Um `likely_ai` sustentado por três camadas independentes não é o mesmo que um `likely_ai`
sustentado por uma só — e o segundo caso vem com aviso em `warnings`.

### Campos de cada achado dentro de `questions[].layers[]`

| campo | significado |
|---|---|
| `layer` | qual camada emitiu: `provenance`, `ocr`, `document`, `forensics`, `image_model`, `synthid`, `text`, `community_probe` ou `jev` |
| `score` | 0 a 1 — o que **aquela camada** acha sobre **aquela questão**. Não é probabilidade calibrada em todas as camadas: é o voto dela |
| `weight` | quanto essa camada pesa **nessa questão**. Uma camada sabe muito de uma pergunta e nada de outra (procedência pesa 1.0 em `has_provenance_marker` e nada em `locally_manipulated`) |
| `reliability` | o tier do voto (§3) — define o peso padrão e se o voto domina |
| `evidence` | frases legíveis dizendo **por que**: `"marcador de openai em metadado/nome"`, `"OCR leu 21 fragmentos de texto"` |
| `answer` | resposta explícita quando a camada tem uma (`sim`/`não`, ou a família do gerador) |

### Campos de cada questão em `questions[]`

`id`, `question` (o texto), `kind` (`boolean`/`choice`/`score`), `primary` (só a
`is_ai_generated` é primária), `answer`, `score`, `confidence`, `reliability` (o tier dominante) e
`layers[]`.

`confidence` **não** é "quão provável é IA" — é **quanto de evidência existiu**: 0.9 quando há
marcador determinístico, ~0.5 com duas ou mais camadas concordando, 0.22 com uma só, 0.05 com
nenhuma. Score alto com confiança baixa significa "o pouco que vi aponta para IA".

---

## 2. O que cada veredito significa

| veredito | regra | como ler |
|---|---|---|
| `ai` | `ai_probability` ≥ 0.90 | evidência decisiva: marcador de gerador, Content Credentials, rótulo lido por OCR, ou várias camadas independentes concordando |
| `likely_ai` | ≥ 0.65 | evidência forte, sem o grau de assinatura do caso acima |
| `inconclusive` | entre 0.35 e 0.65, **ou** confiança < 0.20 **ou** nenhuma camada opinou | **a resposta honesta em muitos casos reais.** Não é erro: é o detector dizendo que não tem base para acusar nem para absolver. Ex.: screenshot muito degradado, PDF de texto puro |
| `likely_human` | ≤ 0.35 | evidência de captura real (EXIF de câmera, ruído de sensor acoplado à luz) sem marcador de IA |
| `human` | ≤ 0.10 | evidência forte de câmera e nenhum sinal de IA |
| `unsupported` | HTTP 415 | formato não analisável: não é imagem nem container com imagem dentro |

**Não existe veredito "é IA com 100%"**, e não por modéstia: todo sinal aqui é estatístico ou
declaratório. Marcador determinístico chega a `ai`; o resto é probabilidade.

---

## 3. Os tiers de confiabilidade (e por que o peso importa)

| tier | peso padrão | significado | exemplos |
|---|---|---|---|
| `deterministic` | 1.0 | assinatura no arquivo; **domina a questão** — não é rebaixada por sinal estatístico nem pela opinião do modelo | chunk `parameters` do A1111, C2PA com `digitalSourceType=trainedAlgorithmicMedia`, rótulo AIGC/TC260, `claim_generator` literal de gerador |
| `strong` | 0.4 | forte mas copiável/falsificável ou interpretativo | EXIF de câmera (copiável), nome de arquivo, rótulo de IA lido por OCR numa legenda, software de edição |
| `model` | 0.3 | opinião do avaliador Jev sobre as evidências | — |
| `weak` | 0.12 | sinal estatístico de pixel | resolução canônica, déficit de alta frequência, dispersão entre blocos |
| `experimental` | 0.03 | sonda não validada: **aparece no relatório, nunca decide** | sonda comunitária de SynthID |

**Evidência independente soma em log-odds** (Bayes), não em média. Três sinais fracos que
concordam valem mais que cada um sozinho; **um sinal fraco sozinho não acusa.** Foi essa regra que
levou o falso positivo em foto real de 13% para 1-2% no regime limpo.

O Jev **não conta como corroboração independente**: ele é downstream, lê os escores das outras
camadas no estado que recebe. Se o `warnings` diz que o veredito se apoia em uma única camada, é
porque só uma camada que olha o artefato acusou.

---

## 4. As 8 questões

| questão | o que responde | quem costuma responder |
|---|---|---|
| `is_ai_generated` **(primária)** | o conteúdo foi gerado por IA? | todas as camadas |
| `generator_family` | qual família de gerador (openai, midjourney, stable_diffusion, google, flux, …) | procedência (marcador) e Jev |
| `has_provenance_marker` | carrega Content Credentials/C2PA verificável? | procedência — responde `não` quando não há box C2PA no arquivo |
| `watermark_declared` | declara watermark ou rótulo de IA (soft binding C2PA, AIGC/TC260, rótulo escrito na imagem)? | procedência, SynthID, OCR |
| `screenshot_or_reencoded` | os metadados foram destruídos por print/reencode/upload? | forense (espectro achatado, grade JPEG desalinhada) e Jev |
| `is_document` | é documento (página, print, scan, PDF) e não fotografia? | documento (estrutura de página) e OCR (leu muito texto) |
| `locally_manipulated` | há indício de alteração **local** (inpainting, patch) e não geração da imagem toda? | nenhuma calibrada — ver §6 |
| `evidence_strength` | quão forte é a evidência total | Jev |

---

## 5. As camadas, uma por uma

Cada linha de `layers[]` traz `name`, `status` (`ok`, `skipped`, `disabled`, `error`), `notes` e
`signals` — os números brutos que a camada mediu.

| camada | o que olha | natureza | observação |
|---|---|---|---|
| `provenance` | C2PA/Content Credentials (manifest, `digitalSourceType`, `c2pa.ai-disclosure`, `c2pa.soft-binding`), EXIF, XMP/IPTC, chunks PNG, metadados de PDF/DOCX, **nome do arquivo**, rótulo TC260/AIGC, assinatura do Grok | determinística | lê os literais **verificados em arquivo real**: `ChatGPT`, `Google C2PA Core Generator Library`, `Microsoft_Responsible_AI/1.0`, `Black Forest Labs API` |
| `ocr` | texto **dentro** da imagem: legenda, faixa, marca d'água visível, etiqueta do gerador | rápida | único sinal que sobrevive a screenshot, porque vem do pixel. `signals.textos` traz a transcrição |
| `document` | é documento? print/scan? + forense de container PDF (atualizações incrementais, produtores divergentes, streams não comprimidos, ferramenta ausente) | estrutural | em foto de documento o OCR decide; em PDF o container decide |
| `forensics` | resolução canônica de gerador, área localmente chapada, déficit de alta frequência, correlação ruído↔luminância, picos espectrais de upsampling, grade JPEG desalinhada, **dispersão entre blocos** | estatística | limiares por **percentil medido** em fotos reais, e por **regime detectado** (o próprio espectro denuncia a degradação) |
| `image_model` | detector neural ONNX: Community Forensics ViT-S/16 (MIT, 87 MB, CPU sem torch) | a mais forte | validado contra o checkpoint oficial; escore calibrado por densidade medida |
| `synthid` | declaração de SynthID no manifest C2PA; detector externo plugável; sonda local por template | condicional | o watermark de imagem não é detectável por código aberto — a API lê a *declaração* |
| `text` | extrai o texto (PDF/DOCX/TXT) e calcula estilometria | condicional | a estilometria **não** gera achado: medida em 1 positivo de gênero diferente dos 80 negativos, está confundida — ver §6 |
| `community_probe` | sonda neural comunitária em ONNX (opcional) | experimental | medida: AUC 0.31-0.59, abaixo de 0.5 justamente em imagem com SynthID. Desligada por padrão |
| `jev` | avaliador text-only: recebe as evidências com score e peso e responde as questões tipadas | modelo | **não vê a imagem** (verificado: não decodifica base64). Peso 0.3, medido |

Camada sem achado **não aparece** em `questions[].layers[]` — ausência ali é abstenção, não
discordância. Os números dela seguem em `layers[].signals`.

---

## 6. Tetos — o que esta API NÃO faz

Medido, não estimado. Cada item traz o número que o sustenta.

- **Screenshot degradado** é o pior caso para os sinais artesanais: resolução canônica cai de AUC
  0.99 para 0.50, déficit de alta frequência para 0.45. Quem decide nesse regime é o
  `image_model` (AUC 0.977 sob screenshot). No braço local, sem ele, o resultado é `inconclusive`
  em vez de chute — abstenção é a escolha deliberada.
- **"Processado com IA" ≠ "gerado por IA".** Ferramentas de editor (AI Denoise, Super Resolution,
  Generative Fill) produzem a mesma assinatura que geração para o detector neural. Uma foto real
  retocada pode sair como `likely_ai` — aconteceu com uma foto oficial de imprensa, cujo escore
  bruto (0.44) ficou 265× acima do máximo em foto real da calibração (0.001). Quando o veredito se
  apoia em uma única camada, `warnings` avisa.
- **Watermark invisível de terceiro não é detectável por código aberto.** SynthID de imagem não
  tem detector público; todos os candidatos medidos caem na chance. A API lê a **declaração** no
  manifest C2PA, não o watermark. O SynthID de **texto** tem verificador implementado e provado,
  mas a marca é definida sobre os token ids do tokenizador do fornecedor: sem chave **e**
  tokenizador, texto de terceiro é indetectável — limitação do esquema, não da implementação.
- **Alteração LOCAL (inpainting/patch) não é detectada.** Duas abordagens medidas e reprovadas:
  dispersão entre blocos (AUC ~0.5, chance) e detector neural aplicado a blocos (1,7% de
  detecção). Precisa de modelo treinado para localização.
- **PDF de texto puro** com metadados apagados: a anomalia estrutural é encontrada (0 de 85 PDFs
  reais produziram o padrão), mas as classes se sobrepõem — o PDF gerado por IA pontua 0.60 e PDFs
  reais também. Um sinal forte sozinho não cruza o limiar de acusação, por desenho.
- **Estilometria não é usada** para decidir: medidas as features em 1 PDF gerado por IA contra 80
  reais, a separação é de **gênero** (lista de compras vs contrato), não de autoria.
- **Amostra pequena.** 6 geradores, 360 imagens, fotos reais só do COCO (amadoras, sem celular
  moderno). Os limiares valem para esse regime; refaça a medição antes de confiar em outro domínio.
- Vídeo, áudio e HEIC/AVIF sem plugin de decode não são suportados.

---

## 7. Calibração

`ai_probability` é o score **calibrado** (τ por regime, ajuste de Youden sobre amostra rotulada);
`ai_probability_raw` é o bruto e `calibration` diz se foi aplicada e com quais números. Evidência
determinística **não** é recalibrada — já é a resposta.

O escore do detector neural passa por mapeamento por densidade (Bayes) antes de entrar: o sigmoide
cru dele não é probabilidade (a mesma imagem gerada vai de 0.97 para 0.009 só por
reescalonamento+recompressão, e o corte real fica em 0.00123, não em 0.5).

---

**Prova de conceito, uso não comercial** (ver `LICENSE`). Nenhum número desta página é escolhido de
cabeça: todos vêm de medição reproduzível pelos scripts em `tools/`, contra amostra rotulada — com
os modelos ONNX e o conjunto baixados por `tools/fetch_bench_set.py`.
"""

EXEMPLO = {
    "verdict": "ai",
    "ai_probability": 0.93,
    "ai_probability_raw": 0.93,
    "calibration": {"calibrated": False, "reason": "evidência determinística não é recalibrada"},
    "confidence": 0.9,
    "confidence_label": "muito forte",
    "generator_guess": "openai",
    "kind": "image",
    "filename": "ChatGPT Image.png",
    "images_analyzed": 1,
    "questions": [{
        "id": "is_ai_generated",
        "question": "O conteúdo foi gerado por IA generativa?",
        "kind": "boolean",
        "primary": True,
        "answer": "sim",
        "score": 0.93,
        "confidence": 0.9,
        "reliability": "deterministic",
        "layers": [
            {"layer": "provenance", "score": 0.93, "weight": 1.0, "reliability": "deterministic",
             "evidence": ["marcador de openai em metadado/nome"]},
            {"layer": "jev", "score": 0.96, "weight": 0.6, "reliability": "model",
             "evidence": ["avaliador Jev: P=0.96"]},
        ],
    }],
    "judge": {"model": "typesafe-ai/jev", "used": True},
    "elapsed_ms": 642,
    "warnings": [],
}

app = FastAPI(
    title="AIGC Watermark Detector API",
    version="0.1.0",
    summary="Detecta conteúdo gerado por IA em imagens e arquivos, com a evidência de cada camada.",
    description=DESCRICAO,
    contact={"name": "wehandle"},
    license_info={"name": "PoC — uso não comercial"},
    openapi_tags=[
        {"name": "detecção", "description": "Analisa um arquivo e devolve o veredito com as evidências."},
        {"name": "serviço", "description": "Saúde e metadados da API."},
    ],
)
CONFIG = Config.from_env()


@app.get("/", include_in_schema=False)
def raiz() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/healthz", tags=["serviço"], summary="Saúde da API e o que está configurado")
def healthz() -> dict:
    from .calibrate import load as carregar_calibracao

    calib = carregar_calibracao()
    return {
        "ok": True,
        "judge_configured": bool(CONFIG.key),
        "judge_model": CONFIG.judge_model,
        "max_upload_mb": CONFIG.max_upload_mb,
        "calibration_loaded": bool(calib),
        "calibration_regimes": sorted((calib or {}).get("regimes", {})),
    }


@app.get("/v1/questions", tags=["serviço"], summary="As questões que a API responde")
def listar_questoes() -> dict:
    return {"questions": [{"id": q.id, "kind": q.kind, "question": q.text, "primary": q.primary}
                          for q in QUESTIONS]}


@app.post(
    "/v1/detect",
    tags=["detecção"],
    summary="Analisa um arquivo e devolve o veredito com as evidências de cada camada",
    responses={
        200: {"description": "Análise concluída", "content": {"application/json": {"example": EXEMPLO}}},
        413: {"description": "Arquivo maior que o limite configurado"},
        415: {"description": "Formato não suportado"},
    },
)
async def detect_endpoint(
    file: UploadFile = File(..., description="Imagem (JPEG/PNG/WebP/TIFF/…) ou container "
                                             "com imagens dentro (PDF, DOCX, PPTX, XLSX, ZIP)"),
) -> JSONResponse:
    data = await file.read()
    limite = CONFIG.max_upload_mb * 1024 * 1024
    if len(data) > limite:
        raise HTTPException(413, f"arquivo maior que {CONFIG.max_upload_mb} MB")
    if not data:
        raise HTTPException(400, "arquivo vazio")
    resultado = detect(data, file.filename or "upload", CONFIG)
    return JSONResponse(resultado, status_code=200 if resultado["verdict"] != "unsupported" else 415)
