"""Optional local FLUX.2 Klein 4B editor, NOT an NVIDIA NIM container.

Only local safetensors weights and loopback JSON are accepted at runtime.
The operator runs a real GPU edit of synthetic geometry before readiness.
Personal photo payloads stay in memory and never reach a hosted provider.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

MODEL_REPO = "black-forest-labs/FLUX.2-klein-4B"
MODEL_ALIAS = "retake-local/flux2-klein-4b"
MAX_BODY = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 4_000_000
MAX_IMAGES = 3


def decode_image(value):
    from PIL import Image
    if not isinstance(value, str):
        raise ValueError("image must be a local data URI")
    prefix, separator, data = value.partition(",")
    if not separator or prefix not in ("data:image/png;base64", "data:image/jpeg;base64"):
        raise ValueError("only PNG/JPEG base64 is accepted")
    if len(data) > (MAX_IMAGE_BYTES * 4 // 3 + 4):
        raise ValueError("image exceeds size limit")
    raw = base64.b64decode(data, validate=True)
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds size limit")
    with Image.open(io.BytesIO(raw)) as image:
        expected = "PNG" if prefix.startswith("data:image/png") else "JPEG"
        if image.format != expected or image.width * image.height > MAX_PIXELS:
            raise ValueError("unsupported or oversized image")
        return image.convert("RGB")


def parse_edit(payload):
    if not isinstance(payload, dict) or set(payload) - {"model", "prompt", "image", "n", "response_format"}:
        raise ValueError("invalid edit payload")
    if payload.get("model") != MODEL_ALIAS or payload.get("n") != 1 or payload.get("response_format") != "b64_json":
        raise ValueError("unsupported model or response format")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000:
        raise ValueError("invalid prompt")
    images = payload.get("image")
    if isinstance(images, str):
        images = [images]
    if not isinstance(images, list) or not 1 <= len(images) <= MAX_IMAGES:
        raise ValueError("one target and at most two references")
    return prompt, [decode_image(value) for value in images]


def model_dimensions(width, height, max_edge):
    if min(width, height) < 16 or max(width, height) / min(width, height) > 3:
        raise ValueError("unsupported crop aspect ratio")
    scale = min(1, max_edge / max(width, height))
    return tuple(max(64, round(value * scale / 16) * 16) for value in (width, height))


def edit_image(pipe, torch, prompt, images, max_edge=768, seed=42):
    from PIL import Image
    target = images[0]
    w, h = model_dimensions(target.width, target.height, max_edge)
    prepared = [target.resize((w, h), Image.Resampling.LANCZOS)]
    for ref in images[1:]:
        # Keep reference orientation; small references reduce attention memory.
        rw, rh = model_dimensions(ref.width, ref.height, min(max_edge, 512))
        prepared.append(ref.resize((rw, rh), Image.Resampling.LANCZOS))
    with torch.inference_mode():
        result = pipe(
            prompt=prompt, image=prepared if len(prepared)>1 else prepared[0],
            width=w, height=h, guidance_scale=1.0, num_inference_steps=4,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]
    if not isinstance(result, Image.Image) or result.width < 16 or result.height < 16:
        raise RuntimeError("model returned invalid image")
    return result.convert("RGB").resize(target.size, Image.Resampling.LANCZOS)


def smoke_edit(pipe, torch, max_edge):
    """Exercise a real multi-reference image edit, without any personal photo."""
    from PIL import Image, ImageDraw, ImageChops
    original=Image.new("RGB",(max_edge,max_edge),(245,245,245))
    inset=max_edge//5
    ImageDraw.Draw(original).ellipse((inset,inset,max_edge-inset,max_edge-inset),fill=(190,30,30))
    reference=Image.new("RGB",(512,512),(245,245,245))
    ImageDraw.Draw(reference).rectangle((100,100,412,412),fill=(30,70,200))
    second=Image.new("RGB",(512,512),(245,245,245))
    ImageDraw.Draw(second).ellipse((100,100,412,412),fill=(30,70,200))
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    output=edit_image(pipe,torch,"Change the red circle in the first image into a blue circle, preserving the white background. Use the color of the second image.",[original,reference,second],max_edge,seed=123)
    torch.cuda.synchronize()
    # Numeric non-noop evidence, not a claim of prompt fidelity or face quality.
    difference=ImageChops.difference(original,output)
    count=sum(1 for rgb in difference.getdata() if max(rgb)>=8)
    ratio=count/(original.width*original.height)
    if ratio<0.03:
        raise RuntimeError("synthetic edit did not visibly change enough pixels")
    return {"synthetic_edit_passed":True,"changed_ratio":round(ratio,5),
            "elapsed_seconds":round(time.monotonic()-start,3),
            "peak_allocated_mib":round(torch.cuda.max_memory_allocated()/2**20),
            "peak_reserved_mib":round(torch.cuda.max_memory_reserved()/2**20),
            "test":"synthetic geometry, not identity or photo-quality certification"}


def load_local_pipeline(model_dir, offload):
    import torch
    from diffusers import Flux2KleinPipeline
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA and BF16 support are required")
    free,total=torch.cuda.mem_get_info()
    if total<12*2**30 or free<8*2**30:
        raise RuntimeError("insufficient available GPU memory for experimental local editor")
    path=Path(model_dir).resolve()
    if not (path/"model_index.json").is_file():
        raise RuntimeError("download the official model into the specified local directory first")
    pipe=Flux2KleinPipeline.from_pretrained(str(path),torch_dtype=torch.bfloat16,
                                           local_files_only=True,use_safetensors=True)
    if offload=="sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()
    return pipe,torch,{"gpu":torch.cuda.get_device_name(0),
                      "vram_total_mib":round(total/2**20),"vram_free_before_mib":round(free/2**20),
                      "offload":offload,"torch":torch.__version__}


def serve(pipe, torch, proof, port, max_edge):
    state={"ready":True}
    class Handler(BaseHTTPRequestHandler):
        protocol_version="HTTP/1.0"
        def setup(self):
            self.request.settimeout(30)
            super().setup()
        def log_message(self,*_):
            # Never log prompts or photo payloads.
            pass
        def reply(self,status,value):
            raw=json.dumps(value,separators=(",",":")).encode()
            self.send_response(status)
            self.send_header("Content-Type","application/json")
            self.send_header("Cache-Control","no-store")
            self.send_header("Content-Length",str(len(raw)))
            self.end_headers()
            try:self.wfile.write(raw)
            except (BrokenPipeError,ConnectionResetError):pass
        def local(self):
            # No cross-origin browser requests, DNS rebinding or proxy URLs.
            return self.client_address[0]=="127.0.0.1" and self.headers.get("Host")==f"127.0.0.1:{port}" and not self.headers.get("Origin")
        def do_GET(self):
            if not self.local():return self.reply(403,{"error":"LOCAL_ONLY"})
            if self.path!="/v1/health/ready":return self.reply(404,{"error":"NOT_FOUND"})
            self.reply(200 if state["ready"] else 503,{"ready":state["ready"],"model":MODEL_ALIAS,"implementation":"diffusers-local-not-nim",**proof})
        def do_POST(self):
            if not self.local():return self.reply(403,{"error":"LOCAL_ONLY"})
            if self.path!="/v1/images/edits":return self.reply(404,{"error":"NOT_FOUND"})
            if not state["ready"]:return self.reply(503,{"error":"RESTART_AND_REPROBE_REQUIRED"})
            try:
                if self.headers.get("Content-Type","").split(";")[0].strip()!="application/json":
                    raise ValueError("application/json required")
                size=int(self.headers.get("Content-Length","0"))
                if not 0<size<=MAX_BODY:raise ValueError("body limit")
                prompt,images=parse_edit(json.loads(self.rfile.read(size)))
            except (ValueError,TypeError,OSError):
                return self.reply(400,{"error":"INVALID_LOCAL_IMAGE_EDIT"})
            try:
                output=edit_image(pipe,torch,prompt,images,max_edge)
                buffer=io.BytesIO();output.save(buffer,format="PNG")
                self.reply(200,{"data":[{"b64_json":base64.b64encode(buffer.getvalue()).decode("ascii")}]})
            except Exception as error:
                # Do not include prompt, paths or image data in error messages.
                state["ready"]=False
                print("Local editor inference failed:",type(error).__name__,flush=True)
                self.reply(503,{"error":"LOCAL_GPU_EDIT_FAILED"})
    with HTTPServer(("127.0.0.1",port),Handler) as server:
        print(json.dumps({"ready":True,"port":port,"model":MODEL_ALIAS,**proof}),flush=True)
        server.serve_forever()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir",required=True)
    parser.add_argument("--port",type=int,default=8004)
    parser.add_argument("--max-edge",type=int,choices=(512,768,1024),default=768)
    parser.add_argument("--offload",choices=("model","sequential"),default="model")
    parser.add_argument("--probe-only",action="store_true")
    args=parser.parse_args()
    if not 1024<=args.port<=65535 or args.port==8765:
        parser.error("choose a separate local port")
    os.environ["HF_HUB_OFFLINE"]="1"
    os.environ["TRANSFORMERS_OFFLINE"]="1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
    pipe,torch,info=load_local_pipeline(args.model_dir,args.offload)
    proof={**info,**smoke_edit(pipe,torch,args.max_edge)}
    if args.probe_only:
        print(json.dumps({"model":MODEL_ALIAS,"ready":True,**proof}),flush=True)
    else:
        serve(pipe,torch,proof,args.port,args.max_edge)

if __name__=="__main__":
    main()
