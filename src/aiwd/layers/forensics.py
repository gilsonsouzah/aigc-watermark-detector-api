"""Camada 2 — forense de pixel/sinal, offline e sem modelo.

Teto conhecido: nenhum destes sinais prova geração por IA sozinho. Eles separam
"imagem de câmera" de "imagem sintetizada/processada" com confiança baixa-média e
sobrevivem em parte a reencode (inclusive screenshot). Peso baixo no veredito.
"""

import io

import numpy as np
from PIL import Image

from ..questions import finding

CROP = 1024  # recorte central no tamanho nativo: nunca reamostra (reamostrar destrói ruído e espectro)
GEN_SIZES = {256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1216, 1280, 1344, 1536, 1792, 2048}
# resoluções clássicas de câmera/monitor (4:3, VGA/XGA/SXGA...) que também caem em múltiplos de 64
DISPLAY_SIZES = {(640, 480), (800, 600), (1024, 768), (1280, 960), (1280, 1024), (1600, 1200), (2048, 1536),
                 (1152, 864), (1400, 1050), (1440, 1080), (720, 576), (1920, 1440)}
SCREEN_SIZES = {(1080, 1920), (1920, 1080), (1170, 2532), (2532, 1170), (1284, 2778), (2778, 1284),
                (1440, 3200), (3200, 1440), (1080, 2340), (2340, 1080), (750, 1334), (1334, 750),
                (828, 1792), (1792, 828), (1366, 768), (2560, 1440), (1440, 2560), (3840, 2160), (2160, 3840)}
AXIS_FREQS = [1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 8, 1 / 10, 1 / 12, 1 / 16]
# Limiares NÃO calibrados: não existe valor público calibrado para este sinal.
# Calibrar contra dataset rotulado antes de confiar; conservador de propósito (prefere perder
# caso comprimido a acusar ruído). Verificado: não dispara em 10 controles de ruído branco e
# dispara em grade periódica injetada (mutações amp=2 e amp=6) e no fixture sintético.
# Limiares derivados de PERCENTIL medido, não de intuição: cada corte é o p95/p5 da
# distribuição em 200 fotos reais (COCO val2017), de modo que o falso positivo em foto
# legítima fique ~5% por sinal. AUC alto não autoriza corte no zero — a distribuição das
# duas classes se sobrepõe, e foi assim que a primeira versão acusou 40% das fotos reais.
# Reproduzir com: python -m tools.measure && python -m tools.evaluate_pipeline
# Limiar por REGIME, não global: a degradação desloca a distribuição inteira. Sob screenshot,
# hf_minus_mid_db cai para -9.1 na foto real e -8.9 no fake — as duas classes cruzam juntas o
# corte do regime limpo (-2.3), e um limiar fixo acusa 100% das fotos. Medido em 80 fotos COCO
# e 84 imagens de gerador por regime.
LIMIARES = {
    "limpo":      {"flat_p95": 0.498, "hf_p5": -2.82, "corr_p95": 0.346, "peak_p95": 12.35},
    "degradado":  {"flat_p95": 0.639, "hf_p5": -11.04, "corr_p95": 0.341, "peak_p95": 12.35},
}
# hf médio da foto real por regime: serve para descobrir em que regime a imagem está
HF_REGIME_LIMIAR = -5.0
# Dispersão entre blocos (alteração local). Preenchido por medição: enquanto a chave não
# existir, a camada reporta o número em `signals` mas NÃO acusa — sinal sem calibração não
# vira veredito (`tools/measure_tamper.py`).
LIMIARES_LOCAL: dict[str, dict] = {}
PEAK_DB_MIN = 6.0
PEAK_TOP_DB = 12.0
PEAK_HARMONIC_DB = 8.0
PEAK_SOLO_DB = 20.0  # pico isolado: em 10 controles de ruído branco o máximo foi 10.5 dB


def _median3(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    return np.median(np.stack([p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)]), axis=0)


def _box3(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    return sum(p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)) / 9.0


def _local_std(a: np.ndarray, k: int = 3) -> np.ndarray:
    p = np.pad(a, k // 2, mode="edge")
    return np.stack([p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(k) for j in range(k)]).std(axis=0)


def _peaks(profile: np.ndarray, fft_len: int) -> list[dict]:
    """profile = meia-linha do espectro (bins 0..Nyquist); fft_len = largura total do FFT.
    O índice do bin é f*fft_len, NUNCA f*len(profile): o profile tem metade dos bins."""
    n = len(profile)
    out = []
    for f in AXIS_FREQS:
        i = int(round(f * fft_len))
        if i < 4 or i > n - 2:
            continue
        neigh = np.concatenate([profile[max(0, i - 4):max(0, i - 1)], profile[min(n, i + 2):min(n, i + 5)]])
        if neigh.size < 2:
            continue
        base = float(np.median(neigh))
        # profile está em log1p(magnitude): diferença no log já É a razão, converte para dB
        db = float((float(profile[i]) - base) * 20 / np.log(10))
        if db >= PEAK_DB_MIN:
            out.append({"freq": round(f, 4), "prominence_db": round(db, 2)})
    return sorted(out, key=lambda d: -d["prominence_db"])[:3]


def _has_harmonic_grid(peaks: list[dict]) -> bool:
    if not peaks:
        return False
    top = peaks[0]
    if top["prominence_db"] >= PEAK_SOLO_DB:
        return True
    if top["prominence_db"] < PEAK_TOP_DB:
        return False
    for p in peaks[1:]:
        if p["prominence_db"] < PEAK_HARMONIC_DB:
            continue
        # o harmônico pode estar abaixo do pico (fundamental 1/16, 2º harmônico 1/8)
        for hi, lo in ((p["freq"], top["freq"]), (top["freq"], p["freq"])):
            ratio = hi / lo
            if ratio >= 1.95 and abs(ratio - round(ratio)) <= 0.08 * round(ratio):
                return True
    return False


def _radial(logmag: np.ndarray) -> dict:
    """Perfil radial em ciclos/pixel: separa espectro de ruído branco de decoder super-suave."""
    h, w = logmag.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    out = {}
    for name, lo, hi in (("dc", 0.0, 0.02), ("low", 0.02, 0.15), ("mid", 0.15, 0.25), ("high", 0.25, 0.5)):
        m = (r >= lo) & (r < hi)
        out[name] = float(logmag[m].mean()) if m.any() else None
    if out["mid"] and out["high"]:
        out["hf_minus_mid_db"] = round(float((out["high"] - out["mid"]) * 20 / np.log(10)), 2)
    return out


def _dispersao(gray: np.ndarray, partes: int = 4) -> dict:
    """Estatística POR BLOCO — o sinal de alteração local, que score global não pega.

    Se um pedaço da imagem foi regenerado (inpainting/Generative Fill) ou colado de outro
    arquivo, o ruído e a grade de compressão daquele bloco deixam de combinar com o resto.
    Medir a média da imagem inteira apaga isso; medir a DISPERSÃO entre blocos não.

    Teto: cena natural também varia muito (céu liso ao lado de folhagem). Por isso o limiar
    tem que vir de percentil medido em fotos reais — ver LIMIARES_LOCAL.
    """
    h, w = gray.shape
    th, tw = h // partes, w // partes
    if min(th, tw) < 48:
        return {}
    sigmas, hfs, blks = [], [], []
    for i in range(partes):
        for j in range(partes):
            t = gray[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
            lstd = _local_std(t)
            plano = lstd < 1.5
            if plano.sum() > 200:
                sigmas.append(float((t - _median3(t))[plano].std()))
            logmag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2((t - _box3(t)) * np.hanning(th)[:, None] * np.hanning(tw)[None, :]))))
            radial = _radial(logmag)
            if radial.get("hf_minus_mid_db") is not None:
                hfs.append(radial["hf_minus_mid_db"])
            blk = _blockiness(t)
            if blk.get("ratio"):
                blks.append(blk["ratio"])
    if len(sigmas) < 4 or len(hfs) < 4:
        return {"n_blocos": partes * partes, "blocos_com_amostra": len(hfs)}
    def cv(v):
        """Coeficiente de variação — só para grandeza SEMPRE positiva (σ, razão de grade)."""
        m = float(np.mean(v))
        return round(float(np.std(v) / m), 3) if m else None

    def mad(v):
        """Desvio absoluto mediano, nas MESMAS unidades — para grandeza sinalizada (dB), onde
        CV não faz sentido: hf_minus_mid_db pode ser negativo e dividir pela média dá número
        sem significado (chegou a dar CV = -0.748 antes desta correção)."""
        med = float(np.median(v))
        return round(float(np.median(np.abs(np.asarray(v) - med))), 3)

    def destaque(v):
        """Z-score robusto do bloco mais extremo: |x - mediana| / MAD. Sempre positivo e
        invariante de escala. Normalizar pela MEDIANA estava errado para grandeza sinalizada
        (dB pode ser negativo, e dividir por mediana negativa invertia o sinal do resultado)."""
        a = np.asarray(v, dtype=float)
        med = float(np.median(a))
        desvio_robusto = float(np.median(np.abs(a - med)))
        if desvio_robusto <= 1e-9:
            return 0.0
        return round(float(np.max(np.abs(a - med)) / desvio_robusto), 3)

    return {"n_blocos": partes * partes, "blocos_com_amostra": len(hfs),
            "sigma_cv": cv(sigmas), "hf_mad": mad(hfs), "hf_destaque": destaque(hfs),
            "grade_cv": cv(blks) if len(blks) >= 4 else None,
            "sigma_destaque": destaque(sigmas),
            "sigma_min": round(min(sigmas), 3), "sigma_max": round(max(sigmas), 3)}


def _blockiness(gray: np.ndarray) -> dict:
    d = np.abs(np.diff(gray, axis=1))
    offs = {}
    for off in range(8):
        cols = np.arange(off, d.shape[1], 8)
        if cols.size >= 4:
            offs[off] = float(d[:, cols].mean())
    if not offs:
        return {}
    interior = float(np.median(list(offs.values())))
    return {"strongest_offset": int(max(offs, key=offs.get)), "ratio": round(max(offs.values()) / interior, 3) if interior else None}


def analyze(data: bytes, suffix: str) -> dict:
    notes: list[str] = []
    findings: list[dict] = []

    try:
        img = Image.open(io.BytesIO(data))
        fmt, size = img.format, img.size
        rgb = img.convert("RGB")
    except Exception as e:
        return {"name": "forensics", "status": "skipped", "findings": [], "signals": {},
                "notes": [f"não é imagem decodificável ({type(e).__name__})"]}

    w, h = size
    if min(size) < 64:
        return {"name": "forensics", "status": "skipped", "findings": [], "signals": {"size": list(size)},
                "notes": ["imagem pequena demais para análise de sinal"]}

    left, top = max(0, (w - CROP) // 2), max(0, (h - CROP) // 2)
    arr = np.asarray(rgb.crop((left, top, min(w, left + CROP), min(h, top + CROP))), dtype=np.float32)
    gray = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    signals: dict = {"format": fmt, "size": [w, h], "analyzed_crop": list(arr.shape[:2])}

    dims = (w, h) if w <= h else (h, w)
    square = w == h and w in GEN_SIZES
    gen_res = (square or (w in GEN_SIZES and h in GEN_SIZES)) and dims not in DISPLAY_SIZES and dims[::-1] not in DISPLAY_SIZES
    signals["generator_resolution"] = bool(gen_res)
    if gen_res:
        # MEDIDO: AUC 0.99 em todos os 6 geradores (limpo, JPEG e resize) — o sinal forense mais
        # forte que existe aqui. Cai para 0.50 sob screenshot, que muda a resolução.
        findings.append(finding("is_ai_generated", 0.85 if square else 0.7, "strong",
                                [f"{w}x{h} é resolução canônica de gerador (AUC medido 0.99)"]))
    if (w, h) in SCREEN_SIZES or (h, w) in SCREEN_SIZES:
        signals["device_resolution"] = True

    signals["clipped_fraction"] = round(float(((arr <= 0) | (arr >= 255)).mean()), 4)

    res = gray - _median3(gray)
    lstd = _local_std(gray)
    flat = lstd < 1.5
    flat_frac = float(flat.mean())
    signals["flat_fraction"] = round(flat_frac, 4)
    # (limiar aplicado adiante, quando o regime já foi determinado pelo espectro)

    noise = float(res[flat].std()) if flat.sum() > 500 else None
    signals["noise_sigma_flat"] = round(noise, 3) if noise is not None else None
    # σ do ruído NÃO vira finding: a direção inverte por gerador (AUC medido 0.07 no FLUX,
    # 0.98 no Seedream-4), então como sinal de direção única ele é pior que nada. Fica no
    # relatório como número bruto para inspeção.
    if noise is not None:
        if flat.sum() > 500 and float(lstd[flat].std()) > 1e-6 and float(gray[flat].std()) > 1e-6:
            corr = float(np.corrcoef(gray[flat].ravel(), lstd[flat].ravel())[0, 1])
            signals["noise_luma_corr"] = round(corr, 3)
            # ATENÇÃO à direção: medido em 200 fotos COCO vs 240 imagens de gerador, a imagem
            # GERADA tem correlação ruído-luminância MAIOR (mediana 0.14) que a foto real
            # (mediana -0.14) — o inverso do que a intuição de "sensor acopla ruído e luz" sugere.
            signals["noise_luma_corr"] = round(corr, 3)

    res_hp = gray - _box3(gray)
    win = np.hanning(arr.shape[0])[:, None] * np.hanning(arr.shape[1])[None, :]
    logmag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(res_hp * win))))
    cy, cx = logmag.shape[0] // 2, logmag.shape[1] // 2
    raw = _peaks(logmag[cy, cx:], logmag.shape[1]) + _peaks(logmag[cy:, cx], logmag.shape[0])
    seen: set[float] = set()
    peaks = [p for p in sorted(raw, key=lambda d: -d["prominence_db"]) if not (p["freq"] in seen or seen.add(p["freq"]))]
    signals["spectral_peaks"] = peaks[:3]
    signals["spectral_peak_db"] = peaks[0]["prominence_db"] if peaks else 0.0
    radial = _radial(logmag)
    signals["spectrum_db"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in radial.items()}
    hf = radial.get("hf_minus_mid_db")
    # o próprio déficit de alta frequência denuncia o regime: reescalonamento/blur derruba a
    # banda alta de TODA imagem, real ou gerada
    regime = "degradado" if (hf is not None and hf < HF_REGIME_LIMIAR) else "limpo"
    lim = LIMIARES[regime]
    signals["regime"] = regime

    if _has_harmonic_grid(peaks):
        # MEDIDO (480 fakes de 6 geradores 2025-26 vs 300 fotos COCO): AUC 0.56 no regime limpo,
        # 0.52 sob screenshot. O detector funciona (pega grade injetada no teste de mutação), mas
        # quase não discrimina gerador moderno — decoders novos não deixam mais o comb clássico.
        forte = signals["spectral_peak_db"] > lim["peak_p95"]
        findings.append(finding("is_ai_generated", 0.62 if forte else 0.55, "weak",
                                [f"grade espectral de upsampling: picos em {[p['freq'] for p in peaks[:3]]} "
                                 f"({peaks[0]['prominence_db']} dB)"
                                 + (", acima do p95 das fotos reais" if forte else " — AUC medido 0.56, sinal fraco")]))
    if hf is not None:
        # Melhor sinal isolado no regime limpo (recall 0.54 com 5% de FP). Sob degradação a
        # separação some (AUC 0.45): o limiar do regime degradado quase nunca dispara, de propósito.
        if hf < lim["hf_p5"]:
            findings.append(finding("is_ai_generated", 0.75 if regime == "limpo" else 0.6, "weak",
                                    [f"déficit de alta frequência ({hf} dB, abaixo do p5 das fotos reais "
                                     f"no regime {regime}): textura suavizada por decoder"]))
    if flat_frac > lim["flat_p95"]:
        findings.append(finding("is_ai_generated", 0.72 if regime == "limpo" else 0.62, "weak",
                                [f"{flat_frac:.0%} da imagem é localmente chapada, acima do p95 das fotos "
                                 f"reais no regime {regime}"]))
    corr_medido = signals.get("noise_luma_corr")
    if corr_medido is not None and corr_medido > lim["corr_p95"]:
        findings.append(finding("is_ai_generated", 0.68 if regime == "limpo" else 0.6, "weak",
                                [f"correlação ruído-luminância r={corr_medido} acima do p95 das fotos reais "
                                 f"no regime {regime}"]))
    if regime == "degradado":
        findings.append(finding("screenshot_or_reencoded", 0.7, "weak",
                                [f"espectro achatado (hf={hf} dB): imagem reescalonada, recomprimida ou "
                                 "capturada de tela — sinais forenses perdem poder aqui"]))

    if fmt == "JPEG":
        try:
            signals["jpeg_quant_tables"] = {str(k): list(v)[:6] for k, v in (img.quantization or {}).items()}
        except Exception:
            pass
    blk = _blockiness(gray)
    signals["blockiness"] = blk
    if (blk.get("ratio") or 0) > 1.08 and blk.get("strongest_offset") != 0:
        signals["unaligned_jpeg_grid"] = True
        findings.append(finding("screenshot_or_reencoded", 0.6, "weak",
                                ["grade JPEG desalinhada: recompressão de segunda geração "
                                 "(típico de screenshot ou upload em rede social)"]))

    signals["sharpness"] = round(float(np.var(np.diff(gray, axis=0)) + np.var(np.diff(gray, axis=1))), 2)
    uniform_rows = int((np.abs(np.diff(gray, axis=0)).max(axis=1) < 0.5).sum())
    signals["uniform_row_fraction"] = round(uniform_rows / max(1, gray.shape[0]), 3)
    if signals.get("device_resolution") and signals["uniform_row_fraction"] > 0.02:
        signals["screenshot_cues"] = ["resolução de tela", "faixas horizontais uniformes"]
        findings.append(finding("screenshot_or_reencoded", 0.75, "weak",
                                ["resolução de dispositivo de tela + faixas horizontais uniformes (chrome de UI)"]))

    disp = _dispersao(gray)
    if disp:
        signals["dispersao_blocos"] = disp
        lim_local = LIMIARES_LOCAL.get(regime)
        if lim_local:
            for chave, limiar, texto in (
                ("sigma_destaque", lim_local.get("sigma_destaque"),
                 "um bloco tem ruído muito acima do resto da própria imagem"),
                ("hf_destaque", lim_local.get("hf_destaque"),
                 "um bloco destoa no espectro de alta frequência em relação ao resto"),
                ("grade_cv", lim_local.get("grade_cv"),
                 "grade de compressão inconsistente entre blocos"),
            ):
                valor = disp.get(chave)
                if limiar is not None and valor is not None and valor > limiar:
                    findings.append(finding("locally_manipulated", 0.68, "weak",
                                            [f"{texto} ({chave}={valor}, acima do p95 medido em fotos "
                                             "reais): indício de alteração local, não de geração total"]))
                    break
    if not findings:
        notes.append("nenhum sinal forense distintivo")
    return {"name": "forensics", "status": "ok", "findings": findings, "signals": signals, "notes": notes}
