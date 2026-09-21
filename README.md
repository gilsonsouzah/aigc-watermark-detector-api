# aigc-watermark-detector-api

API que estima se **uma imagem** — ou **um arquivo com imagens dentro** (PDF, DOCX, PPTX, XLSX, ZIP) —
foi gerada por IA. Um endpoint, resposta hierárquica: **resultado → N questões → N camadas com score
e peso dentro de cada questão**.

> **Prova de conceito, uso não comercial** (ver `LICENSE`). Os números abaixo foram medidos nesta
> máquina, não copiados de papers; onde não medi, está escrito que não medi.

```
POST /v1/detect      multipart/form-data, campo `file`
GET  /healthz
```

## Como decide

| camada | o que olha | natureza |
|---|---|---|
| `provenance` | C2PA/Content Credentials (manifest, `digitalSourceType`, `c2pa.ai-disclosure`, `c2pa.soft-binding`), EXIF, XMP/IPTC, chunks PNG, metadados de PDF/DOCX, nome do arquivo, rótulo TC260/AIGC, assinatura do Grok | determinística |
| `synthid` | declaração de SynthID no manifest; detector externo plugável; sonda local por template (desligada por padrão) | determinística + experimental |
| `forensics` | resolução canônica de gerador, área localmente chapada, déficit de alta frequência, correlação ruído↔luminância, picos espectrais de upsampling, grade JPEG desalinhada, pistas de screenshot, **dispersão entre blocos** (alteração local) | estatística, limiares calibrados |
| `ocr` | lê o texto DENTRO da imagem (rótulo de IA, legenda, marca visível) — único sinal que sobrevive a screenshot | rápida, medida |
| `text` | extrai o texto (PDF/DOCX/TXT), calcula estilometria e verifica SynthID-TEXT quando há chave+tokenizador | condicional |
| `document` | é documento (fundo claro + faixas de texto + proporção de página), print/scan, forense de container PDF (atualizações incrementais, produtores divergentes) | estrutural |
| `image_model` | detector neural em ONNX: Community Forensics ViT-S/16 (MIT, 87 MB fp32, CPU sem torch), validado contra o checkpoint oficial | **o mais forte** |
| `community_probe` | sonda neural comunitária em ONNX (opcional, desligada sem `AIGC_PROBE_ONNX`) | experimental |
| `jev` | recebe as evidências como estado textual e responde cada questão (boolean/choice/score) | modelo avaliador |

Cada camada emite **findings por questão**, com um tier de confiabilidade que vira peso:

```
deterministic 1.0   assinatura no arquivo (C2PA, chunk de gerador, rótulo legal)
strong        0.4   forte mas falsificável/copiável (EXIF de câmera, nome de arquivo)
model         0.3   opinião do avaliador Jev
weak          0.12  sinal estatístico de pixel
experimental  0.03  sonda não calibrada — entra no relatório, nunca decide
```

**Evidência determinística domina a questão** — um marcador explícito não é rebaixado por sinal
estatístico nem pela opinião do modelo (testado nos dois sentidos: Jev dizendo 0.2 e 0.99 não move
um `deterministic` 0.93).

## Rodar

### Bancada em sandbox (medição pesada)

Medição de CPU sempre dentro de container com limite, para não travar a máquina:

```bash
bench/run.sh python tools/fetch_bench_set.py --por-gerador 60   # baixa o conjunto
bench/run.sh python -m tools.measure --n-reais 300              # AUC por sinal e regime
BENCH_CPUS=8 BENCH_MEM=8g bench/run.sh python -m tools.measure_cf
```

`bench/run.sh` aplica `--cpus=6 --memory=6g` (ajustável por `BENCH_CPUS`/`BENCH_MEM`); a cota de
cgroup foi verificada dentro do container (`cpu.max=600000 100000`, `memory.max=6 GiB`). O conjunto
e os modelos ficam em `bench/` (no `.gitignore`) — não em `/tmp`, que não sobrevive a restart.


```bash
uv venv && uv pip install -e .          # + ".[onnx]" para a sonda neural opcional
cp .env.example .env                    # preencha VERCEL_AI_GATEWAY_KEY (vck_...)
uv run uvicorn aiwd.api:app --port 8000 --reload
```

| rota | o que faz |
|---|---|
| **`/docs`** | **Swagger UI — testa por aqui**: `POST /v1/detect` → *Try it out* → escolhe o arquivo → *Execute* |
| `/redoc` | a mesma documentação, em leitura |
| `POST /v1/detect` | upload do arquivo, devolve o veredito com a evidência de cada camada |
| `GET /healthz` | diz se o Jev está configurado e quais regimes de calibração carregaram |
| `GET /v1/questions` | lista as questões que a API responde |

```bash
curl -s -X POST localhost:8000/v1/detect -F file=@imagem.png | jq
uv run aiwd imagem.png --verbose        # mesmo pipeline, com score e peso por camada
uv run python tests/selfcheck.py        # 11 checks, inclui endpoint e gateway real
```

Sem chave do gateway a camada `jev` é pulada e o resultado sai só das camadas locais, com aviso em
`warnings`. A API nunca devolve 5xx por causa do modelo.

## Como ler a resposta

**A referência completa é o próprio `/docs`** — ele tem a tabela de vereditos, as 8 questões, as
camadas uma por uma e os tetos, tudo com descrição de campo. Aqui só o essencial para não
divergir (esta seção já apontou um `layers` que não existia mais):

```
verdict ............... resumo em uma palavra: ai · likely_ai · inconclusive · likely_human · human · unsupported
ai_probability ........ probabilidade calibrada (ai_probability_raw traz o bruto; calibration diz se foi aplicada)
confidence ............ quanto de EVIDÊNCIA existiu, não quão provável é IA (0.9 com marcador, 0.22 com uma camada só)
questions[] ........... as 8 perguntas, cada uma com as camadas que opinaram (score, weight, reliability, evidence)
layers[] .............. evidência BRUTA de cada camada: textos lidos pelo OCR, espectro, dispersão, estilometria
warnings[] ............ sempre leia — inclui aviso quando o veredito se apoia em uma única camada
```

Hierarquia: **resultado → questões → camadas**. Uma camada que abstém **não aparece** em
`questions[].layers[]` (ausência é abstenção, não discordância). `inconclusive` é resposta
frequente e honesta, não falha: é o detector dizendo que não tem base para acusar nem absolver.

## O que foi medido

Bancada: **360 imagens de 6 geradores 2025-26** (GPT-Image-1.5, imagen-4, FLUX.1-dev, SD-3.5-Large,
Nano-Banana, Seedream-4, do T2I-CoReBench, Apache-2.0) contra **300 fotos reais** (COCO val2017),
em 4 regimes de degradação. Reproduzir: `bench/run.sh python -m tools.measure` e
`bench/run.sh python -m tools.evaluate_pipeline` (em sandbox, com limite de CPU).

AUC por sinal isolado (fakes vs fotos reais):

| sinal | limpo | JPEG q75 | resize 0.5 | screenshot |
|---|---|---|---|---|
| resolução canônica de gerador | **0.99** | 0.99 | 1.00 | 0.50 |
| déficit de alta frequência | **0.87** | 0.82 | 0.67 | 0.45 |
| fração de área chapada | 0.79 | 0.80 | 0.77 | **0.72** |
| correlação ruído↔luminância | 0.71 | 0.72 | 0.70 | **0.70** |
| picos espectrais de upsampling | 0.56 | 0.57 | 0.55 | 0.52 |
| σ do ruído em área plana | inverte por gerador (0.07 no FLUX, 0.98 no Seedream) | | | |

Veredito final do pipeline, com todas as correções medidas aplicadas
(`bench/run.sh python -m tools.evaluate_pipeline --n-fakes 25 --n-reais 100`, 150 geradas vs 100 fotos):

| regime | acurácia balanceada | recall | falso positivo em foto real | abstenção em foto |
|---|---|---|---|---|
| limpo | **0.990** | 1.000 (6/6 geradores) | **2,0%** | 98,0% |
| JPEG q75 | **0.970** | 1.000 | 6,0% | 94,0% |
| screenshot | 0.753¹ | 0.507 | **0%** | 100% |

¹ Medição do pipeline **só com as camadas locais** (sem o detector neural, que exige o ONNX baixado
à parte). O `screenshot` do braço local fica em 0.753 porque abstém em 100% das fotos reais em vez
de arriscar — zero falso positivo é a escolha deliberada. Com o detector neural ligado — que mede
**AUC 0.977 justamente sob screenshot**, contra ~0.5 dos sinais artesanais no mesmo regime — os
vereditos ficam decisivos: p=0.992 e 0.930 nas imagens geradas e p=0.0003 na foto real (tabela de
casos acima).

No regime screenshot o detector **abstém** em vez de chutar: acerta FLUX/GPT-Image/imagen-4 com
recall 1.0, responde `inconclusive` em Nano-Banana/SD-3.5, e não acusa nenhuma foto real. Recall
0.513 é o preço honesto dessa postura.

**A medição mudou o código.** O detector de picos espectrais tinha score 0.8 por intuição; medido, dá
AUC 0.56 em gerador moderno — o comb clássico de upsampling praticamente sumiu nos decoders atuais.
O σ do ruído saiu do veredito porque a direção inverte entre geradores. A correlação
ruído↔luminância estava com a **direção invertida** no código: a imagem gerada tem correlação
*maior* (mediana 0.14) que a foto real (−0.14), o oposto do que "sensor acopla ruído e luz" sugere.

Os limiares vêm de **percentil das fotos reais** (p95/p5) **e são por regime**, não corte no zero:
AUC alto não autoriza limiar arbitrário porque as distribuições se sobrepõem, e a degradação
desloca as duas classes juntas (sob screenshot, `hf_minus_mid_db` cai para −9.1 na foto real e
−8.9 no fake). O pipeline detecta o regime pelo achatamento do espectro e troca de limiar.

Progressão medida da acurácia balanceada no regime limpo:

| versão | BA | falso positivo em foto real |
|---|---|---|
| limiares por intuição | 0.740 | **40,8%** |
| percentil global | 0.921 | 15,8% (mas screenshot com 100% de FP) |
| **percentil por regime** | **0.935** | **13,0%** |

A calibração de saída (`calibration.json`, τ por regime por máximo de Youden) foi medida e muda
o resultado em −0,003: os τ ajustados (0.02 a 0.19 em logit) são pequenos demais para mover casos
através dos cortes de veredito. Ela afina a probabilidade reportada em `ai_probability`, não o
veredito — o ganho veio dos limiares dentro das camadas. `ai_probability_raw` traz o score bruto
e `calibration` diz se foi aplicada (evidência determinística não é recalibrada).

## O detector neural: AUC excelente, escore que precisou de calibração

Medição que mudou a decisão, e vale pelo método: o Community Forensics tem **AUC 0.977-0.999**
por regime (tabela abaixo), mas o **sigmoide cru não é probabilidade**. Medido:

```
imagen-4/0009.png  original            -> 0.9733
                   reescalonado + JPEG -> 0.0093   (vira "não é IA")
imagen-4/0001 -> 0.0001   0004 -> 0.0065   (mesmo gerador, sem degradação nenhuma)
```

A AUC fica altíssima porque **AUC é ranking**: os gerados continuam acima das fotos reais (que ficam
em ~0.0000), mesmo quando caem para 0.0001. Mas a agregação usa o valor como probabilidade, e no
limiar 0.5 esses casos viram "não IA". Efeito end-to-end medido — o detector neural **ligado sem
calibrar piora**:

| regime | sem detector neural | com detector neural (escore cru) |
|---|---|---|
| limpo | BA 1.000 · **recall 1.000** · abstenção 100% | BA 0.871 · **recall 0.742** · abstenção 0% |
| JPEG q75 | BA 0.975 · recall 1.000 | BA 0.796 · recall 0.592 |
| screenshot | BA 0.750 · recall 0.500 | BA 0.788 · recall 0.575 |

É a **terceira vez** que o mesmo erro aparece neste projeto, em três camadas diferentes: usar escore
como probabilidade. Nos limiares de pixel foi percentil; no limiar de veredito foi varredura; aqui é
calibração de modelo.

**Correção, e o segundo erro dentro da correção:** a primeira tentativa foi Platt scaling — e ela
piorou (BA 0.967 contra 0.992 do corte direto), porque Platt otimiza probabilidade (log-loss), não
decisão: o ponto onde ele cruza 0.5 não é o corte ótimo, e uma foto real com escore 0.001 saía com
**p=0.85** e viraria falso positivo. O que funciona é estimar as densidades f(escore|IA) e
f(escore|real) por histograma no eixo log10 e devolver a posterior de Bayes — aí **p=0.5 É o corte**:

| regime | BA no limiar cru 0.5 | BA com o mapeamento medido | corte real |
|---|---|---|---|
| limpo | 0.750 | **0.996** | 0.00123 |
| screenshot | 0.604 | **0.875** | 0.00030 |

O corte real é **três ordens de grandeza abaixo de 0.5**, que era o valor que eu estava usando.
Efeito nos casos que falhavam — print de imagem gerada, reescalonado e recomprimido:

| degradação | antes (escore cru) | depois (calibrado) |
|---|---|---|
| print 1.0x JPEG85 | `likely_human` 0.128 | **`ai` 0.916** |
| print 0.75x JPEG75 | `likely_human` 0.130 | **`ai` 0.908** |
| print 0.5x JPEG60 | `likely_human` 0.129 | **`ai` 0.910** |
| print 0.35x JPEG50 | — | `inconclusive` 0.382 (extremo, abstém) |

`tools/calibrate_model.py` gera `bench/model_calibration.json`; a camada lê por
`AIGC_MODEL_CALIBRATION` e, sem o arquivo, devolve o escore cru **avisando em `notes`**.


Medido em 360 imagens de 6 geradores contra 200 fotos reais (`tools/measure_cf.py`, dentro do
sandbox), AUC por regime:

| regime | AUC média | pior gerador | melhor |
|---|---|---|---|
| limpo | **0.999** | imagen-4 1.00 | todos 1.00 |
| JPEG q75 | 0.974 | imagen-4 0.94 | GPT-Image 0.99 |
| resize 0.5 | 0.989 | imagen-4 0.97 | vários 1.00 |
| **screenshot** | **0.977** | imagen-4 0.95 | Seedream 1.00 |

Contraste com os sinais artesanais, que **colapsam** sob screenshot: resolução canônica cai de
AUC 0.99 para 0.50, déficit de alta frequência para 0.45, picos espectrais para 0.52. Não existe
limiar que conserte isso — é falta de sinal no domínio de pixel depois da degradação, e o modelo
treinado é o que repõe. **É o argumento medido para a camada neural existir.**

Custo: ~540 ms por imagem a 384² em 4 núcleos (medido no container com limite). O ONNX `int8`
publicado no mesmo repo **não reproduz o checkpoint** (Δ 0.42) e está reprovado em
`tools/validate_cf_onnx.py` — só o fp32 entra.

## A/B do detector neural calibrado (150 geradas vs 60 fotos)

| regime | **com** neural calibrado | sem neural |
|---|---|---|
| limpo | BA 0.983 · recall 0.983 · FP 1,7% · **abstenção 35%** | BA 1.000 · recall 1.000 · FP 0% · **abstenção 100%** |
| JPEG q75 | BA 0.917 · recall 0.917 · FP 8,3% · abstenção 53% | BA 0.975 · recall 1.000 · FP 5,0% · abstenção 95% |
| **screenshot** | **BA 0.917 · recall 0.850** · abstenção 45% | BA 0.750 · recall 0.500 · abstenção 100% |

Ler com cuidado: no regime limpo o braço sem neural mostra BA 1.000 — mas **abstendo em 100% das
fotos reais**, ou seja, sem responder. Com o neural a abstenção cai para 35% e o regime que importa
(screenshot) sobe de 0.750 para **0.917**, com recall de 0.500 para **0.850**. As diferenças de 1-2
pontos no regime limpo são 1 imagem em 60: dentro do ruído desta amostra.

## Quem decide: agregação determinística, não o modelo

Pergunta testada em A/B com as MESMAS evidências nos dois braços (`tools/measure_jev_judge.py`),
100 arquivos rotulados:

| braço | recall | falso positivo | acurácia balanceada |
|---|---|---|---|
| **agregação determinística** (log-odds dos pesos) | 1.000 | **0.000** | **1.000** |
| Jev decidindo no final, sobre os scores | 1.000 | 0.176 | 0.912 |

Deixar a decisão final a cargo do Jev **piora**: ele é um avaliador text-only que hedgeia para
perto de 0.5 e não está calibrado nos limiares, então introduz falso positivo onde a agregação não
tinha nenhum. Ressalva declarada: 49 das 100 chamadas ao juiz falharam por rate limit do provedor
(HTTP 429), então o braço B saiu de ~51 casos contra 100 do braço A. O peso do Jev em
`is_ai_generated` é ajustável (`AIGC_WEIGHT_JEV`) e varrido por `tools/sweep_jev_weight.py` sobre
um cache congelado (`tools/cache_jev_run.py`), para poder ser medido sem gastar chamada de API.

## Resultado nos casos de teste (servidor vivo)

`bench/casos/`, com o detector neural ligado — `curl -X POST localhost:8000/v1/detect -F file=@...`:

| arquivo | veredito | p | camada que decidiu |
|---|---|---|---|
| `01-imagem-gerada-imagen4.png` | **ai** | 0.992 | `image_model` 0.961 |
| `02-imagem-gerada-flux.png` | **ai** | 0.930 | `image_model` 0.900 |
| `03-foto-real-coco.jpg` | **likely_human** | 0.185 | `image_model` 0.125 |
| `IA-lista-compras.pdf` (gerado por IA) | inconclusive | 0.621 | `document` 0.85 — limitação medida, ver Tetos |

Medido também: **print de imagem gerada** (reescalonado, desfocado e recomprimido em JPEG) é
detectado como `ai` de 1.0x até 0.5x de escala, com p=0.91-0.92.

## Texto: SynthID-TEXT funciona (com a chave), estilometria não foi validada

**O padrão de TEXTO do Google tem código aberto** — diferente do de imagem. `src/aiwd/synthid_text.py`
porta o verificador oficial (`hashing_function`, derivação de g-values, média e média ponderada) para
inteiros puros, sem torch/jax/numpy, emulando o wrap-around de int64 de que o algoritmo depende.
Validado pelo mesmo critério do repositório oficial (que não fixa valores, fixa propriedade):

| teste | resultado |
|---|---|
| tokens aleatórios (null) | média de g = 0.496 / 0.501 / 0.498 — oficial espera 0.5 |
| sequência marcada, chave **certa** | **0.816** (dispara) |
| mesma sequência, chave **errada** | 0.494 (chance) |

**Mas a marca é keyed ao tokenizador**: ela é definida sobre os *token ids* do fornecedor. Sem a
chave secreta E o tokenizador exato, detectar texto de terceiro é impossível — limitação do
esquema, não da implementação. A camada reporta `unavailable` com o motivo em vez de fingir.

**Jev avaliando "este texto é de IA?": medido e REJEITADO.** Em 30 PDFs reais ele acusou **23**
(mediana 0.545) e no PDF gerado por IA deu **0.22** — acusa a maioria dos documentos reais e não
pega o de IA. Como sinal, é pior que inútil. A pergunta `text_ai_generated` foi **removida do
contrato**: pergunta sem fonte válida só polui a saída. O que a camada `text` faz é extrair o
conteúdo (que passa a alimentar o estado do Jev para as outras perguntas) e verificar SynthID-text
quando houver chave e tokenizador.

**Estilometria: medida e NÃO validada.** Em 1 PDF gerado por IA contra 80 PDFs reais as features
parecem separar tudo — palavras/frase 180 vs 25, fração de bullets 0.85 vs 0.00, razão tipo/token
0.84 vs 0.27 — mas o positivo é uma **lista de compras** e os negativos são **contratos e faturas**.
Isso separa lista de prosa, não IA de humano. Com 1 positivo de gênero diferente não há como
calibrar, e um limiar escolhido aqui seria overfitting ao arquivo em mãos. Por isso as features
aparecem em `signals` e **não geram finding**.

## OCR: o único sinal que sobrevive a screenshot

Escolha medida, não por gosto: **OCR clássico resolve, VLM pequeno não é necessário.** `rapidocr`
(ONNX, mesmo runtime do serviço, 31,7 MB de modelo já embutido) lê a 0,2-0,4 s por imagem:

| caso | leitura |
|---|---|
| rótulo em texto claro | `"Exemplo de imagem gerada por uma IA generativa"` + `"[Fonte: Midjourney v5]"` |
| marca d'água cinza-claro (215,215,215) | `"Gerado por IA · SynthID"` |
| 40 fotos reais | 13 com algum texto lido, **0 com rótulo de IA** |

Efeito no caso original — print de imagem gerada **com a legenda que ele carregava**:

| | antes | com OCR |
|---|---|---|
| veredito | `inconclusive` 0.60 | **`likely_ai` 0.707** |
| `watermark_declared` | 0.32 | **0.890** (`ocr` 0.9) |

O achado de OCR entra como `strong` e **não** como `deterministic`: o rótulo pode ser legenda de
artigo *falando sobre* a imagem — indício forte, não assinatura do gerador. Erro de OCR medido e
tratado: `"gerada por uma lAgenerativa"` (o `I` lido como `l`) virou variante explícita na regex.

**Sobre VLM pequeno:** pesquisado e **não adicionado**, com base em benchmark — modelos <1B dão
MMMU 29-34 (aleatório ~25%) e entrada de 384-512 px, que destrói o sinal de alta frequência que
sustenta o detector atual (AUC 0.996). O caminho certo para descrição forense é mandar **números
estruturados** ao Jev, não prosa gerada.

## Tetos — o que esta API **não** faz

- **Watermark invisível de terceiro não é detectável.** SynthID não tem detector público (só portal
  com waitlist). Todos os detectores comunitários foram medidos por auditoria independente e caem na
  chance: AUC pareada 0.517 no mais citado (4,9k★), 0/400 em outro, 214/600 fotos reais aceitas num
  terceiro; o teto teórico por imagem é ~0.75 AUC mesmo com template perfeito. O paper oficial do
  SynthID **não testa screenshot nem recapture** — a robustez de 99,97% é contra crop/resize/JPEG.
  Esta API lê a **declaração** de watermark no manifest; não alega detectar o watermark.
- **Screenshot é o pior caso e continua sendo.** Ele destrói metadado e C2PA, e apaga o sinal forense
  mais forte (resolução canônica cai de AUC 0.99 para 0.50). Sobram os sinais fracos.
- **`generator_guess` só é confiável quando vem de metadado.** Sem marcador, a forense não sabe qual
  gerador foi — e o Jev chuta pelo padrão dos sinais.
- **Amostra pequena.** 6 geradores, 360 imagens, fotos reais só do COCO (fotografia de cena, sem
  celular moderno, sem HDR computacional). Os limiares valem para esse regime; refaça a medição
  antes de confiar em outro domínio.
- **Alteração LOCAL em foto (inpainting, patch colado): a abordagem por dispersão de blocos foi
  medida e REPROVADA.** AUC de 0.499 a 0.601 contra splice real e patch gerado, com recall de 2% a
  13% a 5% de falso positivo (`tools/measure_tamper.py`) — ou seja, chance. O motivo: o patch com
  borda suavizada e recomprimido herda a estatística de ruído do resto da imagem. Por isso
  `LIMIARES_LOCAL` está **vazio**: as features de dispersão aparecem em `signals` para inspeção e
  **não acusam nada**. Sinal sem calibração não vira veredito — e sinal que mede chance não é
  detector.
  A **segunda tentativa também foi medida e reprovada**: usar o detector neural (o mesmo que dá
  AUC 0.999 na imagem inteira) recortando blocos e olhando onde ele pontua alto — pensando "se o
  pedaço foi gerado, aquele bloco pontua como IA". Resultado em 60 fotos com patch gerado colado
  (`tools/measure_tile_patch.py`): score no bloco do patch 0.020 contra 0.002 no controle, acerto
  pareado 0.783, e **1,7% dos patches passariam do limiar**. O modelo é treinado para classificar a
  imagem inteira e não localiza — um recorte com o miolo gerado continua parecendo foto.
  Conclusão medida: **alteração local precisa de modelo treinado para localização** (a literatura
  da área é de rede dedicada a splicing/inpainting), que está fora do escopo atual e é uma decisão
  a tomar — não algo que se resolva calibrando sinal.
- Vídeo, áudio e HEIC/AVIF sem plugin de decode não são suportados. Não há detecção de texto.

## Marcação legal (EU AI Act)

O Art. 50(2) do Reg. (UE) 2024/1689 exige que conteúdo sintético seja marcado em formato legível por
máquina; aplicável desde 2/ago/2026, com prazo até 2/dez/2026 para sistemas já no mercado
(Reg. (UE) 2026/1744, Art. 111(4)). O Código de Prática de transparência (10/jun/2026, ~190
signatários incluindo OpenAI, Google, Meta, Microsoft, BFL) prevê, na Medida 3.4, **API pública de
interoperabilidade de detecção até 2/fev/2027** — é a interface que a camada `synthid` já espera em
`SYNTHID_DETECTOR_URL`. A camada `provenance` lê hoje o que os signatários escrevem no arquivo.

## Estrutura

```
src/aiwd/questions.py    as 8 questões + peso de cada camada dentro de cada questão
src/aiwd/aggregate.py    combina as camadas DENTRO de cada questão (log-odds; determinístico domina)
src/aiwd/layers/
    provenance.py        C2PA/manifest, EXIF/XMP/IPTC, marcadores de gerador, nome de arquivo
    forensics.py         sinais de pixel com limiar por regime (o próprio espectro detecta o regime)
    document.py          é documento? PDF/print/scan + forense estrutural de PDF
    image_model.py       detector neural ONNX (Community Forensics), validado contra o checkpoint
    synthid.py           declaração de SynthID + detector externo plugável + sonda por template
    community_probe.py   sonda comunitária (medida: AUC 0.31-0.59, desligada por padrão)
src/aiwd/judge.py        perguntas tipadas ao Jev (único modelo; peso medido)
src/aiwd/describe.py     descrição forense plugável (AIGC_VLM_URL) para alimentar o Jev
src/aiwd/calibrate.py    deslocamento de logit com τ medido por regime
src/aiwd/pipeline.py     orquestra e monta a resposta hierárquica
src/aiwd/api.py          FastAPI (/docs, /redoc, /v1/detect, /v1/questions, /healthz)

Dockerfile.bench         sandbox da bancada
bench/run.sh             roda ferramenta com --cpus/--memory limitados (BENCH_CPUS, BENCH_MEM)
bench/                   conjunto, modelos, casos de teste e logs (no .gitignore, 2 GB)

tools/fetch_bench_set.py      baixa o conjunto (streaming do tar.gz de 7 GB por gerador)
tools/measure.py              AUC por sinal e por regime
tools/measure_cf.py           AUC do detector neural por família e regime
tools/measure_tamper.py       alteração local e documento (as duas abordagens reprovadas)
tools/measure_tile_patch.py   detector neural aplicado a blocos (também reprovado)
tools/measure_pdf.py          PDFs reais como controle, com rótulo por prefixo IA-
tools/evaluate_pipeline.py    veredito final do pipeline
tools/measure_jev_judge.py    A/B: agregação determinística vs Jev decidindo
tools/cache_jev_run.py        congela uma passada com o Jev para varrer pesos sem gastar API
tools/sweep_jev_weight.py     varre o peso do Jev
tools/sweep_threshold.py      varre o limiar de veredito
tools/fit_calibration.py      ajusta τ e escreve calibration.json
tools/validate_cf_onnx.py     valida o ONNX do detector neural contra o checkpoint oficial
tools/validate_onnx_probe.py  idem para a sonda comunitária
tools/export_synthid_surrogates.py / _probe_torch_logits.py / _cf_torch_logits.py
tests/selfcheck.py            13 checks, sem framework (inclui endpoint e gateway real)
```
