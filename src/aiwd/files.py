"""Detecção de tipo e extração das imagens de dentro de um arquivo container."""

import io
import zipfile

from PIL import Image

MAGIC = (
    (b"\xff\xd8\xff", "image"), (b"\x89PNG\r\n\x1a\n", "image"), (b"GIF8", "image"),
    (b"BM", "image"), (b"II*\x00", "image"), (b"MM\x00*", "image"),
    (b"RIFF", "image"), (b"\x00\x00\x00\x0cjP", "image"), (b"%PDF", "pdf"),
    (b"PK\x03\x04", "zip"), (b"\x1aE\xdf\xa3", "unsupported"),
)
ZIP_KINDS = {"word/": "docx", "ppt/": "pptx", "xl/": "xlsx"}
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".avif", ".heic")
MAX_IMAGES = 8


def detect_kind(data: bytes, filename: str) -> str:
    for magic, kind in MAGIC:
        if data.startswith(magic):
            if kind == "zip":
                try:
                    with zipfile.ZipFile(io.BytesIO(data)) as z:
                        names = z.namelist()
                    return next((k for p, k in ZIP_KINDS.items() if any(n.startswith(p) for n in names)), "zip")
                except zipfile.BadZipFile:
                    return "unsupported"
            if kind == "image" and data.startswith(b"RIFF") and data[8:12] not in (b"WEBP", b"AVI "):
                return "unsupported"
            return kind
    if data.startswith(b"\x00\x00\x00") and data[4:8] == b"ftyp":
        return "image"
    return "unsupported"


def extract_images(data: bytes, kind: str, filename: str) -> list[tuple[str, bytes]]:
    """Imagens embutidas no arquivo, na ordem em que aparecem (limitado a MAX_IMAGES)."""
    if kind == "image":
        return [(filename or "upload", data)]
    if kind == "pdf":
        from pypdf import PdfReader

        out = []
        for i, page in enumerate(PdfReader(io.BytesIO(data)).pages):
            for j, img in enumerate(getattr(page, "images", [])):
                out.append((f"page{i + 1}:{getattr(img, 'name', j)}", img.data))
                if len(out) >= MAX_IMAGES:
                    return out
        return out
    if kind in ("docx", "pptx", "xlsx", "zip"):
        out = []
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                if name.startswith("__MACOSX") or not name.lower().endswith(IMAGE_EXT):
                    continue
                try:
                    blob = z.read(name)
                    Image.open(io.BytesIO(blob)).verify()
                except Exception:
                    continue
                out.append((name, blob))
                if len(out) >= MAX_IMAGES:
                    break
        return out
    return []
