"""Servidor de referência do descritor (implementa o contrato de AIGC_VLM_URL).

Recebe POST com os bytes da imagem e o prompt no header `x-prompt`, devolve
{"description": "..."}. Serve para (a) plugar qualquer VLM local atrás deste contrato e
(b) medir o ganho da descrição sem depender de modelo escolhido — com o baseline
`--mock` devolvendo string vazia, para o A/B de "com descrição" vs "sem descrição".

    python tools/run_describer.py --modelo <repo-hf> [--porta 8100]

O modelo é carregado por transformers (precisa de torch; é o único ponto do projeto que
usa, e roda fora do processo da API).
"""

import argparse
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from PIL import Image

MODELO = None
PROCESSADOR = None


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        tamanho = int(self.headers.get("content-length", 0))
        data = self.rfile.read(tamanho)
        prompt = self.headers.get("x-prompt", "Descreva a imagem.")
        try:
            imagem = Image.open(io.BytesIO(data)).convert("RGB")
            mensagens = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            texto = PROCESSADOR.apply_chat_template(mensagens, add_generation_prompt=True)
            entradas = PROCESSADOR(text=texto, images=[imagem], return_tensors="pt")
            with torch.no_grad():
                saida = MODELO.generate(**entradas, max_new_tokens=512, do_sample=False)
            resposta = PROCESSADOR.batch_decode(saida[:, entradas["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            corpo = json.dumps({"description": resposta}).encode()
            self.send_response(200)
        except Exception as e:
            corpo = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
            self.send_response(500)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def log_message(self, *a) -> None:
        pass


def main() -> int:
    global MODELO, PROCESSADOR
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--porta", type=int, default=8100)
    a = ap.parse_args()
    from transformers import AutoModelForVision2Seq, AutoProcessor

    PROCESSADOR = AutoProcessor.from_pretrained(a.modelo)
    MODELO = AutoModelForVision2Seq.from_pretrained(a.modelo, torch_dtype=torch.float32).eval()
    print(f"descritor servindo em http://127.0.0.1:{a.porta} (modelo {a.modelo})")
    HTTPServer(("127.0.0.1", a.porta), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
