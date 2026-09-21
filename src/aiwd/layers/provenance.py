"""Camada 1 — procedência: Content Credentials (C2PA), metadados e marcadores de gerador.

Determinística, sem modelo. Marcador forte aqui é assinatura; ausência de metadado não é
evidência de nada (screenshot, reencode e toda rede social apagam).

Os literais de provedor abaixo foram lidos de arquivos reais (fixtures de conformidade da CAI
e amostras coletadas), não inferidos de documentação — ver README. Onde o rótulo existe no
arquivo mas não tem definição normativa publicada, está marcado como empírico.
"""

import io
import json
import re

from PIL import ExifTags, Image

from ..questions import finding

try:
    import c2pa
except ImportError:
    c2pa = None

Image.MAX_IMAGE_PIXELS = 80_000_000

# família -> padrões que aparecem em METADADO (nome de arquivo, XMP, EXIF, chunk PNG, /Info de PDF)
GENERATORS = {
    "openai": (r"\bdall[\-·.]?e\b", r"\bopenai\b", r"chatgpt", r"\bsora\b", r"gpt-image"),
    "midjourney": (r"\bmidjourney\b", r"\bmj[_ ]", r"job id", r"--(?:ar|v|stylize|chaos|seed|niji)\b"),
    "stable_diffusion": (
        r"stable diffusion", r"negative[_ ]prompt", r"cfg scale", r"model hash", r"automatic1111",
        r"comfyui", r"invoke ?ai", r"\bsdxl\b", r"\bsd ?1\.5\b", r"euler a", r"steps: \d+, sampler",
        r"fooocus", r"\bnovelai\b", r"stable image",
    ),
    "firefly": (r"\bfirefly\b", r"adobe express", r"generative fill", r"adobe imaging"),
    "google": (r"gemini_?generated", r"\bimagen\b", r"made with google ai", r"synthid",
               r"google ai studio", r"google c2pa core generator"),
    "microsoft": (r"bing image creator", r"microsoft designer", r"microsoft_responsible_ai",
                  r"responsible ai image provenance", r"image creator from designer", r"\bcopilot\b"),
    "flux": (r"black ?forest ?labs", r"\bflux\b"),
    "xai": (r"\bgrok\b", r"grok imagine"),
    "stability": (r"stability ?ai", r"dreamstudio"),
    "canva": (r"\bcanva\b",),
    "ideogram": (r"ideogram",),
    "recraft": (r"recraft",),
    "leonardo": (r"leonardo\.?ai",),
    "playground": (r"playground ?ai",),
    "krea": (r"\bkrea\b",),
    "runway": (r"\brunway(?:ml)?\b",),
}
GENERATOR_RE = {fam: re.compile("|".join(p), re.IGNORECASE) for fam, p in GENERATORS.items()}
FILENAME_RE = re.compile(r"|".join(f"({p})" for p in (
    r"chatgpt[ _-]?image", r"dall[\-·.]?e", r"gemini[ _-]?generated", r"imagen", r"midjourney",
    r"stable[ _-]?diffusion", r"sd[ _-]?xl", r"firefly", r"flux", r"grok[ _-]?image", r"leonardo",
    r"ideogram", r"krea", r"recraft", r"generated[ _-]?(?:image|by)",
)), re.IGNORECASE)
EDITORS_RE = re.compile(r"adobe photoshop|lightroom|capture one|\bgimp\b|darktable|affinity photo|figma", re.IGNORECASE)
CAMERA_TAGS = ("Make", "Model", "ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength",
               "LensModel", "DateTimeOriginal", "ExposureProgram", "MeteringMode")
# URIs IPTC de c2pa.actions.digitalSourceType e de Iptc4xmpExt:DigitalSourceType
DIGITAL_SOURCE = {
    "trainedalgorithmicmedia": (0.99, "gerado por IA generativa (trainedAlgorithmicMedia)"),
    "compositewithtrainedalgorithmicmedia": (0.90, "IA generativa combinada com captura (inpainting)"),
    "trainedalgorithmicdata": (0.85, "dado gerado por modelo treinado"),
    "algorithmicallyenhanced": (0.55, "aperfeiçoado por algoritmo (pode não ser IA generativa)"),
    "algorithmicmedia": (0.5, "mídia gerada por algoritmo (ambíguo: CGI sintético ou IA generativa)"),
    "computationalcapture": (0.15, "captura computacional (HDR de celular) — NÃO é IA"),
    "digitalcapture": (0.05, "captura de câmera"),
}
# assert    ions CAWG/C2PA 2.4 que declaram IA ou watermark
AI_ASSERTIONS = ("c2pa.ai-disclosure", "c2pa.ai_generated_content")
WATERMARK_ASSERTIONS = ("c2pa.soft-binding", "c2pa.watermarked.bound", "c2pa.watermarked.unbound")
TRAINING_LABELS = ("cawg.training-mining", "c2pa.training-mining")
# rótulos de watermark registrados no c2pa-org/softbinding-algorithm-list (CC-BY-4.0)
SOFT_BINDING_NAMES = {
    "com.digimarc.validate.1": "Digimarc",
    "com.imatag.lamark.v1": "IMATAG",
    "ai.steg.api": "Steg.AI",
    "com.adobe.trustmark.q": "Adobe TrustMark Q",
    "com.adobe.trustmark.c": "Adobe TrustMark C",
    "com.adobe.trustmark.p": "Adobe TrustMark P",
    "com.microsoft.invismark.1": "Microsoft InvisMark",
    "com.microsoft.wavmark.1": "Microsoft WavMark",
    "ai.contentlens.image": "Meta Content Seal / VideoSeal",
}
SYNTHID_RE = re.compile(r"synthid", re.IGNORECASE)
GROK_SIG_RE = re.compile(r"^Signature:\s*[A-Za-z0-9+/=]{20,}")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
AIGC_MARK_RE = re.compile(r"TC260:AIGC|[\"']AIGC[\"']|AIGC\s*[:=]\s*\{", re.IGNORECASE)
AIGC_FIELDS_RE = re.compile(r"contentproducer|produceid|contentpropagator|propagateid|serviceprovider|serviceuser", re.IGNORECASE)
C2PA_BOX_MARKERS = (b"c2pa", b"jumbf", b"urn:content-credentials", b"BEGIN C2PA MANIFEST")
HIDDEN_C2PA_RE = re.compile(rb"-----BEGIN C2PA MANIFEST-----")
VAR_SELECTOR_RUN = re.compile(rb"(?:[\xef\xb8\x80-\xef\xb8\x8f]|[\xf3\xa0\x81\x80-\xf3\xa0\x81\xbf]){16,}")
SOURCE_IN_TEXT_RE = re.compile(r"digitalsourcetype/([A-Za-z]+)")


def _image_texts(data: bytes) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        return out
    for k, v in (img.info or {}).items():
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace") if k in ("xmp", "XML:com.adobe.xmp", "comment") else repr(v[:4_000])
        out[str(k)] = str(v)[:20_000]
    try:
        for k, v in (img.getexif() or {}).items():
            out[f"exif:{ExifTags.TAGS.get(k, k)}"] = str(v)[:2_000]
        for k, v in ((img.getexif() or {}).get_ifd(ExifTags.IFD.Exif) or {}).items():
            out[f"exif:{ExifTags.TAGS.get(k, k)}"] = str(v)[:2_000]
    except Exception:
        pass
    try:
        out.update({f"xmp:{k}": str(v)[:4_000] for k, v in (img.getxmp() or {}).items()})
    except Exception:
        pass
    for k, v in (getattr(img, "text", None) or {}).items():
        out[f"png:{k}"] = str(v)[:20_000]
    return out


def _raw_texts(data: bytes) -> dict[str, str]:
    """XMP/JUMBF/IPTC vivem em claro no container, fora do alcance do Pillow."""
    out: dict[str, str] = {}
    for m in re.finditer(rb"(?:<x:xmpmeta|<\?xpacket)[\x20-\x7e\n\r\t]{0,8000}", data[:8_000_000]):
        out[f"raw-xmp:{len(out)}"] = m.group(0).decode("utf-8", "replace")
    iptc = data.find(b"Photoshop 3.0")
    if iptc >= 0:
        out["iptc"] = bytes(b for b in data[iptc:iptc + 8_192] if 32 <= b < 127).decode("ascii", "replace")
    return out


def container_texts(data: bytes, kind: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if kind == "pdf":
        try:
            from pypdf import PdfReader

            meta = PdfReader(io.BytesIO(data)).metadata or {}
            out.update({f"pdf:{k.lstrip('/')}": str(v)[:2_000] for k, v in dict(meta).items()})
        except Exception as e:
            out["pdf:error"] = f"{type(e).__name__}: {e}"[:200]
    elif kind in ("docx", "pptx", "xlsx", "odt"):
        import zipfile

        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for name in ("docProps/app.xml", "docProps/core.xml", "meta.xml"):
                    if name in z.namelist():
                        out[f"zip:{name}"] = z.read(name).decode("utf-8", "replace")[:4_000]
        except Exception as e:
            out["zip:error"] = f"{type(e).__name__}: {e}"[:200]
    return out


def _c2pa(data: bytes, suffix: str) -> dict:
    """Lê o manifest ativo. Ausência de manifest é caso normal, não erro."""
    res: dict = {"present": any(m in data[:8_000_000] for m in C2PA_BOX_MARKERS),
                 "hidden_in_text": bool(HIDDEN_C2PA_RE.search(data) or VAR_SELECTOR_RUN.search(data)),
                 "read": False, "error": None, "claim_generator": None, "source_type": None,
                 "actions": [], "assertions": [], "soft_binding": None, "ai_disclosure": None,
                 "watermark_declarations": [], "training_mining": None, "validation_state": None}
    if not res["present"] or c2pa is None:
        return res
    try:
        reader = c2pa.Reader.try_create(suffix.lstrip(".") or None, stream=io.BytesIO(data))
        if reader is None:
            res["error"] = "box c2pa presente mas manifest ilegível"
            return res
        store = json.loads(reader.json())
        res["read"] = True
        res["validation_state"] = store.get("validation_state")
        manifest = (store.get("manifests") or {}).get(store.get("active_manifest")) or {}
        info = manifest.get("claim_generator_info") or manifest.get("claim_generator")
        res["claim_generator"] = (info if isinstance(info, str) else json.dumps(info, ensure_ascii=False))[:400]
        for a in manifest.get("assertions", []):
            label = a.get("label", "")
            res["assertions"].append(label)
            detail = a.get("data") or {}
            if label.startswith("c2pa.actions"):
                for act in detail.get("actions", []):
                    entry = {"action": act.get("action"), "digitalSourceType": act.get("digitalSourceType"),
                             "description": act.get("description"),
                             "agent": (act.get("softwareAgent") or {}).get("name") if isinstance(act.get("softwareAgent"), dict) else act.get("softwareAgent")}
                    res["actions"].append(entry)
                    if entry["digitalSourceType"]:
                        res["source_type"] = entry["digitalSourceType"].rsplit("/", 1)[-1]
            if label.startswith("c2pa.soft-binding"):
                res["soft_binding"] = detail.get("alg") or detail.get("soft-binding-map", {}).get("alg")
            if label.startswith("c2pa.ai-disclosure"):
                model = (detail.get("ai-model-disclosure-map") or detail)
                res["ai_disclosure"] = {k: model.get(k) for k in ("modelType", "modelName", "modelIdentifier", "contentProfile")}
            if any(t in label for t in TRAINING_LABELS):
                res["training_mining"] = label
        blob = json.dumps(manifest, ensure_ascii=False)
        if res["source_type"] is None:
            m = SOURCE_IN_TEXT_RE.search(blob)
            res["source_type"] = m.group(1) if m else None
        res["watermark_declarations"] = [a["description"] for a in res["actions"] if a.get("description") and SYNTHID_RE.search(a["description"])]
        res.pop("manifests", None)
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"[:300]
    return res


def analyze(data: bytes, suffix: str, kind: str, filename: str = "") -> dict:
    notes: list[str] = []
    findings: list[dict] = []

    texts = _image_texts(data) if kind == "image" else {}
    texts.update(container_texts(data, kind))
    texts.update(_raw_texts(data))
    texts["filename"] = filename
    signals: dict = {"metadata_keys": sorted(k for k in texts if k != "filename")[:60], "bytes": len(data)}
    blob = " ".join(texts.values())

    hit = FILENAME_RE.search(filename)
    if hit:
        signals["filename_signal"] = hit.group(0)
        findings.append(finding("is_ai_generated", 0.8, "strong",
                                [f"nome de arquivo típico de gerador: {hit.group(0)!r}"]))

    c2 = _c2pa(data, suffix)
    signals["c2pa"] = c2
    _c2pa_findings(c2, findings, notes, kind)

    # fonte declarada em XMP/IPTC comum (Meta usa Iptc4xmpExt:DigitalSourceType sem C2PA)
    for st in {m.group(1).lower() for m in SOURCE_IN_TEXT_RE.finditer(blob)}:
        mapped = next((v for k, v in DIGITAL_SOURCE.items() if st.startswith(k)), None)
        if mapped and not c2.get("source_type"):
            signals["xmp_digital_source_type"] = st
            findings.append(finding("is_ai_generated", mapped[0], "deterministic",
                                    [f"XMP/IPTC DigitalSourceType={st}: {mapped[1]}"]))

    fams = sorted({fam for fam, rx in GENERATOR_RE.items() if any(rx.search(v) for v in texts.values())})
    if fams:
        signals["generator_markers"] = fams
        findings.append(finding("is_ai_generated", 0.93, "deterministic",
                                [f"marcador de {f} em metadado/nome" for f in fams]))
        findings.append(finding("generator_family", 0.9, "deterministic", ["marcador de gerador no arquivo"],
                                answer=fams[0] if len(fams) == 1 else "|".join(fams)))

    if AIGC_MARK_RE.search(blob) and AIGC_FIELDS_RE.search(blob):
        signals["tc260_aigc"] = True
        findings.append(finding("is_ai_generated", 0.9, "deterministic",
                                ["rótulo TC260 AIGC (rotulagem obrigatória chinesa): Label/ContentProducer/ProduceID"]))
        findings.append(finding("watermark_declared", 0.85, "deterministic", ["rótulo TC260 AIGC presente"]))

    grok = _grok_signature(texts)
    if grok:
        signals["grok_signature"] = True
        findings.append(finding("is_ai_generated", 0.85, "strong",
                                ["assinatura de imagem do Grok/xAI no EXIF (Artist=UUID + ImageDescription 'Signature: …'), "
                                 "formato não verificável offline"]))

    camera = [k for k in texts if k.startswith("exif:") and k.split(":", 1)[1] in CAMERA_TAGS]
    if camera:
        signals["camera_exif"] = sorted(camera)
        findings.append(finding("is_ai_generated", 0.12, "strong",
                                [f"EXIF de captura presente: {', '.join(sorted(camera)[:4])}"
                                 " (copiável, não é prova)"]))
    editors = [k for k, v in texts.items() if EDITORS_RE.search(v)]
    if editors:
        signals["editor_software"] = editors[:5]
        findings.append(finding("is_ai_generated", 0.35, "strong",
                                [f"software de edição no metadado: {', '.join(editors[:3])}"
                                 " (Photoshop também edita foto real e tem Generative Fill)"]))
    if not camera and not editors and kind == "image":
        findings.append(finding("screenshot_or_reencoded", 0.55, "weak",
                                ["nenhum EXIF de captura: típico de screenshot, reencode ou export sem metadado"]))
    if not findings:
        notes.append("nenhum metadado de procedência utilizável")

    return {"name": "provenance", "status": "ok", "findings": findings, "signals": signals, "notes": notes}


def _c2pa_findings(c2: dict, findings: list[dict], notes: list[str], kind: str) -> None:
    def add(q, score, rel, ev, answer=None):
        findings.append(finding(q, score, rel, ev, answer=answer))

    if not c2["present"]:
        if kind == "image":
            add("has_provenance_marker", 0.05, "weak", ["nenhum box C2PA/JUMBF no arquivo"])
        return
    add("has_provenance_marker", 1.0, "deterministic",
        [f"box C2PA/JUMBF presente (validation_state={c2['validation_state']})"])
    if c2["hidden_in_text"]:
        add("has_provenance_marker", 0.9, "strong",
            ["C2PA embutido em texto (manifest em claro ou variation selectors)"])

    # claim_generator literal: pega gerador que marca sem digitalSourceType (ex.: Microsoft)
    cg = (c2.get("claim_generator") or "").lower()
    vendor = next((fam for fam, rx in GENERATOR_RE.items() if rx.search(cg)), None)
    if vendor:
        add("is_ai_generated", 0.95, "deterministic", [f"claim_generator do C2PA: {c2['claim_generator']!r}"])
        add("generator_family", 0.95, "deterministic", ["claim_generator literal do manifest"], answer=vendor)

    for label in c2.get("assertions", []):
        if label.startswith(AI_ASSERTIONS):
            extra = c2.get("ai_disclosure") or {}
            model = extra.get("modelName") or extra.get("modelIdentifier") or extra.get("modelType")
            rel = "deterministic" if label.startswith("c2pa.ai-disclosure") else "strong"
            ev = [f"assertion {label}" + (f" (modelo: {model})" if model else "")]
            if extra.get("contentProfile"):
                ev.append(f"contentProfile: {extra['contentProfile']}")
            add("is_ai_generated", 0.97 if rel == "deterministic" else 0.85, rel, ev)
        if label.startswith(WATERMARK_ASSERTIONS):
            alg = c2.get("soft_binding") or ""
            nome = next((v for k, v in SOFT_BINDING_NAMES.items() if alg.lower().startswith(k)), alg or "algoritmo não declarado")
            add("watermark_declared", 0.9, "deterministic", [f"{label} → watermark declarado: {nome}"])

    if c2["source_type"]:
        st = c2["source_type"].lower()
        mapped = next((v for k, v in DIGITAL_SOURCE.items() if st.startswith(k)), None)
        if mapped:
            add("is_ai_generated", mapped[0], "deterministic",
                [f"c2pa.actions digitalSourceType={c2['source_type']}: {mapped[1]}"])
        else:
            notes.append(f"digitalSourceType não mapeado: {c2['source_type']}")
    elif c2["read"]:
        notes.append("manifest C2PA sem digitalSourceType: c2pa.created sozinho NÃO prova geração por IA")

    if c2["watermark_declarations"]:
        add("watermark_declared", 0.95, "deterministic",
            [f"manifest declara watermark: {d}" for d in c2["watermark_declarations"][:2]])
    if c2["training_mining"]:
        notes.append(f"{c2['training_mining']}: declara PERMISSÃO de treino, não geração por IA")
    if c2["error"]:
        notes.append(f"manifest C2PA não lido: {c2['error']}")


def _grok_signature(texts: dict) -> bool:
    artist = texts.get("exif:Artist", "")
    desc = texts.get("exif:ImageDescription", "")
    return bool(UUID_RE.match(artist.strip()) and GROK_SIG_RE.match(desc.strip()))
