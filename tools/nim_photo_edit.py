"""Optional on-device image editors; no remote API, key or image URL.
Official JSON /v1/images/edits contract:
https://docs.nvidia.com/nim/visual-genai/latest/getting-started.html
Configured local ports are trusted operator inputs, never browser parameters.
"""
from __future__ import annotations

import base64
import http.client
import json
import os
from pathlib import Path

try:
    from .managed_flux2 import MODEL as MANAGED_MODEL, managed_config, managed_request
except ImportError:
    from managed_flux2 import MODEL as MANAGED_MODEL, managed_config, managed_request

MODELS = {
    "qwen/qwen-image-edit-2511": 3,
    "qwen/qwen-image-edit-2509": 3,
    "qwen/qwen-image-edit": 1,
    "black-forest-labs/flux-3-image-klein": 8,
    "black-forest-labs/flux-3-image-klein-base": 8,
    "black-forest-labs/flux.2-klein-4b": 8,
    "black-forest-labs/flux.1-kontext-dev": 1,
    # Separate open-weight Diffusers implementation, not an official NIM.
    "retake-local/flux2-klein-4b": 3,
}
MAX_RESPONSE = 32 * 1024 * 1024
PROMPT = (
    "Refine the facial skin texture and fine details of the FIRST photographic input, "
    "using the additional portraits if supplied solely as references for its existing face. "
    "Retain the first image's head position, expression, eyes, lighting, colors and framing. "
    "Keep its hair, glasses, ears, neck, clothes and background unchanged. "
    "No illustration, beauty filter, extra person, watermark, text or collage."
)

def configured_editors(value: str | None = None) -> dict[str, int]:
    raw = os.environ.get("RETAKE_NIM_EDITORS", "") if value is None else value
    configured = {}
    ports = set()
    for row in raw.split(","):
        if not row.strip():
            continue
        name, sep, port = row.strip().partition("=")
        if not sep or name not in MODELS or not port.isascii() or not port.isdigit() or not 1024 <= int(port) <= 65535 or int(port) == 8765 or int(port) in ports or name in configured:
            raise ValueError("RETAKE_NIM_EDITORS: yalnız listelenen model=yerel_port çiftleri; portlar ayrı olmalı.")
        configured[name] = int(port)
        ports.add(int(port))
    if value is None and MANAGED_MODEL not in configured and managed_config() is not None:
        configured[MANAGED_MODEL] = 0  # internal one-shot worker, not a port
    return configured

def local_json(port: int, path: str, payload=None, timeout=180):
    # No DNS, redirects, HTTP proxy environment, remote endpoints or tokens.
    if not 1024 <= port <= 65535 or port == 8765 or path not in ("/v1/health/ready", "/v1/images/edits"):
        raise ValueError("Yerel NIM adresi geçersiz.")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET" if payload is None else "POST", path,
                     None if payload is None else json.dumps(payload).encode("utf-8"),
                     {"Accept":"application/json","Content-Type":"application/json"})
        res=conn.getresponse()
        if res.status != 200:
            raise RuntimeError(f"NIM HTTP {res.status}")
        raw=res.read(MAX_RESPONSE+1)
        if len(raw)>MAX_RESPONSE:
            raise ValueError("NIM yanıtı çok büyük.")
        if payload is None:
            return True
        return json.loads(raw)
    finally:
        conn.close()

def image_url(path: Path):
    raw=path.read_bytes()
    mime='image/png' if raw.startswith(b'\x89PNG\r\n\x1a\n') else 'image/jpeg' if raw.startswith(b'\xff\xd8') else None
    if not mime or len(raw)>12*1024*1024:
        raise ValueError('NIM için yerel görsel geçersiz.')
    return 'data:'+mime+';base64,'+base64.b64encode(raw).decode('ascii')

def edit_payload(model: str, image: Path, refs: list[Path]):
    count=MODELS[model]
    images=[image_url(image),*(image_url(p) for p in refs[:count-1])]
    return {"model":model,"prompt":PROMPT,"image":images if count>1 else images[0],"n":1,"response_format":"b64_json"}

def candidate_from_editor(model: str, port: int, base: Path, refs: list[Path], bounds, output: Path, request=local_json, extra_strengths=(), texture_strengths=()):
    if model == MANAGED_MODEL and port == 0:
        request = managed_request
    import cv2
    import numpy as np
    # Strict mask: NIM never gets to enlarge the existing authorized matte.
    original=cv2.imread(str(base),cv2.IMREAD_UNCHANGED)
    if original is None or original.ndim!=3 or original.shape[2]!=4:
        raise ValueError("Doğrulanmış RGBA aday yok.")
    if not request(port,"/v1/health/ready",timeout=3):
        raise RuntimeError("NIM hazır değil.")
    # Encode only the actual preblended RGB, not an alpha-transparent image.
    ok, buf=cv2.imencode(".png",original[:,:,:3])
    if not ok: raise ValueError("NIM girdi kodlaması başarısız.")
    input_path=output.with_suffix(".input.png")
    input_path.write_bytes(buf.tobytes())
    result=request(port,"/v1/images/edits",edit_payload(model,input_path,refs),timeout=360 if model=="retake-local/flux2-klein-4b" else 180)
    rows=result.get("data") if isinstance(result,dict) else None
    encoded=rows[0].get("b64_json") if isinstance(rows,list) and len(rows)==1 and isinstance(rows[0],dict) else None
    if not isinstance(encoded,str) or len(encoded)>MAX_RESPONSE:
        raise ValueError("NIM görsel yanıtı geçersiz.")
    raw=base64.b64decode(encoded,validate=True)
    if not (raw.startswith(b"\x89PNG\r\n\x1a\n") or raw.startswith(b"\xff\xd8")):
        raise ValueError("NIM görsel biçimi desteklenmiyor.")
    generated=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if generated is None: raise ValueError("NIM görseli okunamadı.")
    h,w=original.shape[:2]
    gh,gw=generated.shape[:2]
    if abs((gw/gh)/(w/h)-1)>.015:
        raise ValueError("NIM çıktı oranı hedefle eşleşmiyor.")
    if (gw,gh)!=(w,h):
        generated=cv2.resize(generated,(w,h),interpolation=cv2.INTER_AREA if gw>=w else cv2.INTER_CUBIC)
    x,y,fw,fh=(float(bounds[k]) for k in ("x","y","width","height"))
    yy,xx=np.ogrid[:h,:w]
    radius=((xx+.5-x-fw*.5)/(fw*.43))**2+((yy+.5-y-fh*.5)/(fh*.46))**2
    feather=np.clip((1-radius)*2.5,0,1)
    # One inference, several bounded composites: compare the actual pixels
    # rather than repeating diffusion or blindly accepting a stronger edit.
    for strength in (0.35, *extra_strengths):
        if not isinstance(strength, (int, float)) or not 0 < strength <= .50:
            raise ValueError("Yerel düzenleme karışım gücü geçersiz.")
        weight=(feather*np.minimum(original[:,:,3].astype(np.float32)/255,1)*strength).astype(np.float32)
        merged=original.copy()
        merged[:,:,:3]=np.clip(np.rint(original[:,:,:3]*(1-weight[:,:,None])+generated*weight[:,:,None]),0,255).astype(np.uint8)
        path=output if strength==.35 else output.with_name(f"{output.stem}-{round(strength*100)}{output.suffix}")
        if not cv2.imwrite(str(path),merged): raise OSError("NIM adayı yazılamadı.")
    for strength in texture_strengths:
        merged=texture_only_candidate(original,generated,bounds,strength)
        path=output.with_name(f"{output.stem}-texture-{round(strength*100)}{output.suffix}")
        if not cv2.imwrite(str(path),merged): raise OSError("NIM doku adayı yazılamadı.")
    return output


def texture_only_candidate(original, generated, bounds, strength):
    """Limited luminance-band transfer from the SAME diffusion inference.

    No low-frequency color, geometry, facial outline or feature transplant.
    Restrict the residual to interior forehead/cheeks; reject poorly aligned
    structures and strongly different local exposure. Every output is still
    independently scored by the engine.
    """
    import cv2
    import numpy as np
    if not isinstance(strength,(int,float)) or not 0<strength<=.5:
        raise ValueError("Invalid texture blend strength")
    if original.ndim!=3 or original.shape[2]!=4 or generated.shape!=original[:,:,:3].shape:
        raise ValueError("Invalid texture candidate dimensions")
    h,w=original.shape[:2]
    x,y,fw,fh=(float(bounds[k]) for k in ("x","y","width","height"))
    if not np.isfinite([x,y,fw,fh]).all() or min(fw,fh)<48 or x<0 or y<0 or x+fw>w+1 or y+fh>h+1:
        return original.copy()
    x0,y0=max(0,int(x)),max(0,int(y))
    x1,y1=min(w,int(np.ceil(x+fw))),min(h,int(np.ceil(y+fh)))
    base=original[y0:y1,x0:x1]
    gen=generated[y0:y1,x0:x1]
    a=cv2.cvtColor(base[:,:,:3],cv2.COLOR_BGR2GRAY).astype(np.float32)
    b=cv2.cvtColor(gen,cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Only fine residuals, not the diffusion model's face shape or tone.
    low_a=cv2.GaussianBlur(a,(0,0),1.25)
    low_b=cv2.GaussianBlur(b,(0,0),1.25)
    hi_a,hi_b=a-low_a,b-low_b
    # Suppress mismatched structures and exposure. Compare locally after
    # smoothing, rather than interpreting a generated texture as real pores.
    disagreement=cv2.GaussianBlur(np.abs(low_a-low_b),(0,0),2.0)
    confidence=np.clip((18-disagreement)/12,0,1)
    gx=cv2.Sobel(low_a,cv2.CV_32F,1,0,ksize=3)/8.
    gy=cv2.Sobel(low_a,cv2.CV_32F,0,1,ksize=3)/8.
    edge=np.clip((12-np.hypot(gx,gy))/8,0,1)
    yy,xx=np.mgrid[y0:y1,x0:x1]
    nx,ny=(xx-x)/fw,(yy-y)/fh
    mask=np.zeros(a.shape,np.float32)
    for cx,cy,rx,ry in ((.50,.17,.23,.095),(.24,.54,.085,.105),(.76,.54,.085,.105)):
        radius=((nx-cx)/rx)**2+((ny-cy)/ry)**2
        zone=np.clip((1-radius)*2,0,1).astype(np.float32)
        mask=np.maximum(mask,zone*zone*(3-2*zone))
    opacity=base[:,:,3].astype(np.float32)/255.
    residual=np.clip(hi_b-hi_a,-8,8)*mask*confidence*edge*opacity*strength
    output=original.copy()
    corrected=np.clip(np.rint(base[:,:,:3].astype(np.float32)+residual[:,:,None]),0,255).astype(np.uint8)
    output[y0:y1,x0:x1,:3]=corrected
    return output
