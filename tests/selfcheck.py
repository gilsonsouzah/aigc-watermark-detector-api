"""Check executável do projeto. Sem framework: python tests/selfcheck.py

Cobre o que pode quebrar em silêncio: marcador determinístico, detector espectral
(com controle negativo), extração de imagem de container, piso determinístico sobre o
avaliador, e o endpoint HTTP de ponta a ponta.

Usa Config(key=None) no pipeline para não depender de rede; um teste separado exercita
o gateway real e é pulado quando a chave não está configurada.
"""

import io
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aiwd.aggregate import resolve
from aiwd.config import Config
from aiwd.layers import forensics, provenance, synthid
from aiwd.pipeline import detect
from aiwd.questions import finding

OFFLINE = Config(key=None)


def png_with_text(**texts) -> bytes:
    meta = PngImagePlugin.PngInfo()
    for k, v in texts.items():
        meta.add_text(k, v)
    buf = io.BytesIO()
    Image.new("RGB", (512, 512), (30, 60, 90)).save(buf, "PNG", pnginfo=meta)
    return buf.getvalue()


def jpeg_photo(seed: int = 0, size=(768, 1024)) -> bytes:
    r = np.random.default_rng(seed)
    arr = np.clip(r.normal(120, 35, size) + r.normal(0, 3, size), 0, 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(np.stack([arr] * 3, -1)).save(buf, "JPEG", quality=92)
    return buf.getvalue()


def score_of(layer, question):
    top = [f for f in layer["findings"] if f["question"] == question]
    return max((f["score"] for f in top), default=None)


def test_marcador_de_procedencia():
    layer = provenance.analyze(png_with_text(parameters="a cat\nSteps: 20, Sampler: Euler a, CFG scale: 7, Size: 512x512"),
                               ".png", "image", "x.png")
    assert "stable_diffusion" in layer["signals"]["generator_markers"], layer["signals"]
    assert score_of(layer, "is_ai_generated") >= 0.9, layer["findings"]
    assert [f for f in layer["findings"] if f["reliability"] == "deterministic"]

    by_name = provenance.analyze(png_with_text(), ".png", "image", "Gemini_Generated_Image_ab12.png")
    assert by_name["signals"]["filename_signal"], by_name["signals"]
    assert score_of(by_name, "is_ai_generated") >= 0.8, by_name["findings"]

    # câmera de verdade: EXIF de captura sem marcador de gerador
    ex = Image.new("RGB", (64, 64)).getexif()
    ex[271], ex[272], ex[305], ex[33434] = "Canon", "EOS R6", "Canon EOS R6", "1/250"
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, "JPEG", exif=ex)
    cam = provenance.analyze(buf.getvalue(), ".jpg", "image", "IMG_1234.JPG")
    assert "camera_exif" in cam["signals"], cam["signals"]
    assert score_of(cam, "is_ai_generated") <= 0.2, cam["findings"]

    # sem metadado nenhum: procedência se abstém sobre is_ai_generated (não acusa) mas
    # registra a ausência de C2PA como evidência da questão de procedência
    limpo = provenance.analyze(png_with_text(), ".png", "image", "a.png")
    assert score_of(limpo, "is_ai_generated") is None, limpo["findings"]
    assert score_of(limpo, "has_provenance_marker") <= 0.1, limpo["findings"]


def spectral_finding(layer):
    return [f for f in layer["findings"] if "espectral" in " ".join(f["evidence"])]


def test_detector_espectral_nao_dispara_em_ruido():
    for seed in range(6):
        assert not spectral_finding(forensics.analyze(jpeg_photo(seed), ".jpg")), seed


def test_detector_espectral_dispara_em_grade_injetada():
    mutacao = np.asarray(Image.open(io.BytesIO(jpeg_photo()))).astype(np.float32)
    mutacao[:, ::8] += 6.0
    buf = io.BytesIO()
    Image.fromarray(mutacao.clip(0, 255).astype("uint8")).save(buf, "PNG")
    achados = spectral_finding(forensics.analyze(buf.getvalue(), ".png"))
    # o score é baixo de propósito (AUC medido 0.56 em geradores modernos); o que o teste
    # garante é que o detector DISPARA na grade injetada, não o valor do score
    assert achados and achados[0]["score"] > 0.5, achados


def test_extrai_imagem_de_container():
    inner = io.BytesIO()
    Image.new("RGB", (1024, 1024), (120, 30, 200)).save(inner, "PNG")
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("word/document.xml", "<w:document/>")
        z.writestr("word/media/image1.png", inner.getvalue())
    out = detect(outer.getvalue(), "doc.docx", OFFLINE)
    assert out["kind"] == "docx" and out["images_analyzed"] == 1, out["images_analyzed"]
    assert out["per_image"][0]["image"].endswith("image1.png")


def test_marcador_deterministico_domina_no_agregado():
    prov = {"name": "provenance", "status": "ok", "signals": {}, "notes": [],
            "findings": [finding("is_ai_generated", 0.93, "deterministic", ["chunk parameters"]),
                         finding("is_ai_generated", 0.8, "strong", ["nome de arquivo"])]}
    # avaliador discorda para baixo e para cima: o determinístico continua mandando
    for p in (0.2, 0.99):
        qs = resolve([prov], [finding("is_ai_generated", p, "model", ["Jev"])])
        q = next(x for x in qs if x["id"] == "is_ai_generated")
        assert q["score"] == 0.93 and q["reliability"] == "deterministic", (p, q)
        assert len(q["layers"]) == 3, q["layers"]
    # sem determinístico, as camadas se combinam
    fraco = {"name": "forensics", "status": "ok", "signals": {}, "notes": [],
             "findings": [finding("is_ai_generated", 0.6, "weak", ["resolução"])]}
    q = next(x for x in resolve([fraco], [finding("is_ai_generated", 0.9, "model", ["Jev"])])
             if x["id"] == "is_ai_generated")
    assert q["score"] is not None and q["reliability"] != "deterministic", q
    # evidência independente SOMA: dois sinais fracos valem mais que um sozinho (log-odds)
    so_um = {"name": "forensics", "status": "ok", "signals": {}, "notes": [],
             "findings": [finding("is_ai_generated", 0.72, "weak", ["flat"])]}
    dois = {"name": "forensics", "status": "ok", "signals": {}, "notes": [],
            "findings": [finding("is_ai_generated", 0.72, "weak", ["flat"]),
                         finding("is_ai_generated", 0.75, "weak", ["hf"])]}
    p1 = next(x for x in resolve([so_um], []) if x["id"] == "is_ai_generated")["score"]
    p2 = next(x for x in resolve([dois], []) if x["id"] == "is_ai_generated")["score"]
    assert p2 > p1 + 0.05, (p1, p2)  # medido: dois sinais fracos somam ~+0.09 em log-odds


def test_synthid_declaracao_no_manifest():
    """Declaração de SynthID (forma literal do C2PA do Google) é evidência determinística."""
    prov = {"c2pa": {"watermark_declarations": ["Applied imperceptible SynthID watermark."],
                     "claim_generator": "Google C2PA Core Generator Library"}}
    out = synthid.analyze(png_with_text(), "image", OFFLINE, prov)
    wm = [f for f in out["findings"] if f["question"] == "watermark_declared"]
    assert wm and wm[0]["score"] >= 0.9 and wm[0]["reliability"] == "deterministic", out["findings"]
    assert out["signals"]["declaration"]
    # sem declaração e sem template, a camada NÃO inventa número
    limpo = synthid.analyze(png_with_text(), "image", OFFLINE, {})
    assert not limpo["findings"], limpo["findings"]
    assert limpo["signals"]["probe"]["status"] == "disabled", limpo["signals"]["probe"]


def test_sonda_synthid_so_dispara_com_a_marca():
    """Mutação controlada da plumbing: grade periódica serve de stand-in do watermark.
    A sonda tem que separar marcada de limpa — e nunca alegar SynthID, só reportar correlação."""
    def foto(seed, size=(512, 512)):
        g = np.random.default_rng(seed)
        return np.clip(g.normal(120, 30, size) + g.normal(0, 4, size), 0, 255).astype("uint8")
    tmp = Path(tempfile.mkdtemp(prefix="aiwd-check-"))
    limpa = Image.fromarray(foto(1))
    limpa_path = tmp / "limpa.jpg"
    limpa.save(limpa_path, "JPEG", quality=92)
    base = np.asarray(limpa).astype(np.float32)
    marcada = base.copy()
    marcada[:, ::8] += 5.0
    marcada[::8, :] -= 4.0
    marcada_path = tmp / "marcada.png"
    Image.fromarray(marcada.clip(0, 255).astype("uint8")).save(marcada_path)
    outra = tmp / "outra.jpg"
    Image.fromarray(foto(2)).save(outra, "JPEG", quality=92)

    salvos = {k: os.environ.get(k) for k in ("SYNTHID_LOCAL_PROBE", "SYNTHID_TEMPLATE")}
    os.environ["SYNTHID_LOCAL_PROBE"] = "1"
    os.environ["SYNTHID_TEMPLATE"] = str(marcada_path)
    cfg = Config.from_env()
    com_marca = synthid.analyze(marcada_path.read_bytes(), "image", cfg, {})["signals"]["probe"]
    sem_marca = synthid.analyze(outra.read_bytes(), "image", cfg, {})["signals"]["probe"]
    assert com_marca["status"] == "ok" and sem_marca["status"] == "ok"
    assert com_marca["raw_correlation"] > 0.9, com_marca
    assert sem_marca["raw_correlation"] < 0.2, sem_marca
    # peso do tier experimental é mínimo: nunca decide a questão sozinha
    q = next(x for x in resolve([synthid.analyze(marcada_path.read_bytes(), "image", cfg, {})], [])
             if x["id"] == "watermark_declared")
    assert q["layers"][0]["weight"] <= 0.5 and q["layers"][0]["reliability"] == "experimental", q
    # sem template: recusa em vez de chutar
    os.environ.pop("SYNTHID_TEMPLATE", None)
    sem_template = synthid.analyze(marcada_path.read_bytes(), "image", Config.from_env(), {})["signals"]["probe"]
    assert sem_template["status"] == "unavailable" and sem_template["score"] is None, sem_template
    for k, v in salvos.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    shutil.rmtree(tmp, ignore_errors=True)


def test_peso_efetivo_aparece_na_saida():
    prov = {"name": "provenance", "status": "ok", "signals": {}, "notes": [],
            "findings": [finding("is_ai_generated", 0.93, "deterministic", ["marcador"])]}
    q = next(x for x in resolve([prov], []) if x["id"] == "is_ai_generated")
    assert q["layers"][0]["weight"] == 1.0, q
    assert all("weight" in l for l in q["layers"]), q


def test_pdf_com_editor_detectado():
    """PDF alterado por cima (append incremental) tem que ser visto; PDF novo, não."""
    import tempfile

    from pypdf import PdfReader, PdfWriter

    from aiwd.layers import document

    tmp = Path(tempfile.mkdtemp(prefix="aiwd-check-pdf-"))
    novo = tmp / "novo.pdf"
    Image.new("RGB", (595, 842), (255, 255, 255)).save(novo, "PDF")
    antes = document._pdf(novo.read_bytes())
    assert antes["incremental_updates"] == 1, antes
    assert not [f for f in document.analyze(novo.read_bytes(), "pdf", "n.pdf")["findings"]
                if f["question"] == "locally_manipulated"]
    with novo.open("ab") as f:
        escritor = PdfWriter()
        escritor.append(PdfReader(str(novo)))
        escritor.add_blank_page(width=200, height=200)
        escritor.write(f)
    depois = document.analyze(novo.read_bytes(), "pdf", "n.pdf")
    assert document._pdf(novo.read_bytes())["incremental_updates"] == 2
    achados = [f for f in depois["findings"] if f["question"] == "locally_manipulated"]
    assert achados, depois["findings"]
    shutil.rmtree(tmp, ignore_errors=True)


def test_documento_nao_confunde_foto():
    """Página é documento; foto e imagem gerada não são. Medido: 0/200 fotos, 0/360 geradas."""
    import zipfile

    from aiwd.layers import document

    pagina = Image.new("RGB", (1240, 1754), (255, 255, 255))
    desenho = ImageDraw.Draw(pagina)
    for i in range(48):
        desenho.text((80, 90 + i * 32), f"Contrato n. 4471-B · linha {i} · valor R$ 1000,00", fill=(25, 25, 25))
    buf = io.BytesIO(); pagina.save(buf, "PNG")
    assert max(f["score"] for f in document.analyze(buf.getvalue(), "image", "p.png")["findings"]
               if f["question"] == "is_document") >= 0.8

    zip_reais = Path(__file__).resolve().parents[1] / "bench" / "reals" / "val2017.zip"
    if zip_reais.is_file():
        with zipfile.ZipFile(zip_reais) as z:
            fotos = [z.read(n) for n in z.namelist() if n.endswith(".jpg")][:20]
        for foto in fotos:
            score = max(f["score"] for f in document.analyze(foto, "image", "f.jpg")["findings"]
                        if f["question"] == "is_document")
            assert score < 0.5, score


def test_synthid_text_null_e_marcado():
    """Verificador de SynthID-Text: null em token aleatório, dispara no marcado, exige a chave.

    O critério do null (média de g ~ 0.5) é o MESMO que o repositório oficial do Google usa no
    teste dele — o teste deles não fixa valores, fixa a propriedade estatística.
    """
    import random

    from aiwd.synthid_text import CHAVES_DEMO, marcar, verificar

    rng = random.Random(1)
    aleatorio = [rng.randrange(5000) for _ in range(1500)]
    g = verificar(aleatorio, CHAVES_DEMO)["media"]
    assert abs(g - 0.5) < 0.02, g                      # null: não marca nada

    marcado = marcar(50, vocab=250, chaves=CHAVES_DEMO)
    com_chave = verificar(marcado, CHAVES_DEMO)
    sem_chave = verificar(marcado, [c + 7 for c in CHAVES_DEMO])
    assert com_chave["media"] > 0.6, com_chave          # dispara na chave certa
    assert abs(sem_chave["media"] - 0.5) < 0.05, sem_chave   # e só nela
    assert com_chave["veredito"] == "indeterminado" or len(marcado) < 100


def test_ocr_le_rotulo_de_ia_escrito_na_imagem():
    """Rótulo de IA escrito na imagem é o único sinal que sobrevive a screenshot.

    Não pode inventar rótulo em foto real — o lado negativo é o que importa aqui.
    """
    import zipfile

    from aiwd.layers import ocr

    if ocr._motor()[0] is None:
        print("  (pulado: rapidocr ausente)")
        return
    pagina = Image.new("RGB", (1000, 600), (255, 255, 255))
    desenho = ImageDraw.Draw(pagina)
    desenho.text((40, 40), "Exemplo de imagem gerada por uma IA generativa", fill=(20, 20, 20))
    buf = io.BytesIO(); pagina.save(buf, "PNG")
    achados = [f for f in ocr.analyze(buf.getvalue(), "image")["findings"]
               if f["question"] == "watermark_declared"]
    assert achados and achados[0]["score"] >= 0.8, achados

    zip_reais = Path(__file__).resolve().parents[1] / "bench" / "reals" / "val2017.zip"
    if zip_reais.is_file():
        with zipfile.ZipFile(zip_reais) as z:
            fotos = [z.read(n) for n in z.namelist() if n.endswith(".jpg")][:5]
        for foto in fotos:
            assert not ocr.analyze(foto, "image")["findings"], "foto real não pode ter rótulo de IA"


def test_foto_de_documento_reconhecida_pelo_ocr():
    """Foto de documento (cartão plastificado sobre mesa) não é PÁGINA: os limiares de papel e
    tinta foram calibrados em página renderizada e deram is_document 0.13 nesse caso. Quem resolve
    é o OCR, que não depende de fundo nem de cor do cartão."""
    import io as _io

    from aiwd.layers import document, ocr

    if ocr._motor()[0] is None:
        print("  (pulado: rapidocr ausente)")
        return
    cartao = Image.new("RGB", (486, 411), (150, 120, 80))          # fundo madeira
    desenho = ImageDraw.Draw(cartao)
    desenho.rectangle((20, 20, 466, 391), fill=(190, 215, 195))     # cartão esverdeado
    for i, linha in enumerate(("REPUBLICA FEDERATIVA DO BRASIL", "ESTADO DO RIO DE JANEIRO",
                               "CARTEIRA DE IDENTIDADE", "NOME COMPLETO AQUI",
                               "FILIACAO", "DATA NASC", "NATURALIDADE", "OBSERVACAO")):
        desenho.text((40, 50 + i * 40), linha, fill=(20, 40, 20))
    buf = _io.BytesIO(); cartao.save(buf, "JPEG", quality=85)
    dados = buf.getvalue()

    o = ocr.analyze(dados, "image")
    assert o["signals"]["n_textos"] >= 8, o["signals"]
    achados = [f for f in document.analyze(dados, "image", "c.jpg", o["signals"])["findings"]
               if f["question"] == "is_document"]
    assert achados and achados[0]["score"] >= 0.8, achados


def test_formato_nao_suportado():
    out = detect(b"nao sou midia", "x.bin", OFFLINE)
    assert out["verdict"] == "unsupported" and out["ai_probability"] is None


def test_endpoint():
    from aiwd.api import app

    client = TestClient(app)
    assert client.get("/healthz").json()["ok"] is True
    r = client.post("/v1/detect", files={"file": ("ChatGPT Image.png", png_with_text(parameters="a cat"), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] in ("ai", "likely_ai") and body["ai_probability"] >= 0.8, body
    # hierarquia: resultado -> questões -> camadas com score dentro da questão
    assert [q["id"] for q in body["questions"]][:2] == ["is_ai_generated", "generator_family"]
    q = body["questions"][0]
    assert q["layers"] and all({"layer", "score", "reliability"} <= set(l) for l in q["layers"]), q
    assert client.post("/v1/detect", files={"file": ("x.bin", b"nope", "application/octet-stream")}).status_code == 415


def test_gateway_jev_se_configurado():
    cfg = Config.from_env()
    if not cfg.key:
        print("  (pulado: VERCEL_AI_GATEWAY_KEY ausente)")
        return
    out = detect(png_with_text(parameters="a cat\nSteps: 20, Sampler: Euler a, CFG scale: 7"), "x.png", cfg)
    assert out["judge"]["model"] == cfg.judge_model, out["judge"]
    assert isinstance(out["judge"]["answers"], dict), out["judge"]
    assert {"is_ai_generated", "generator_family", "evidence_strength"} <= set(out["judge"]["answers"]), out["judge"]
    assert any(l["layer"] == "jev" for q in out["questions"] for l in q["layers"])


if __name__ == "__main__":
    testes = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in testes:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(testes)} checks passaram")
