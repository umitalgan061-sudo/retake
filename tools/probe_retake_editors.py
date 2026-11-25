"""Report real local GPU/device and optional editor readiness without personal photos."""
from __future__ import annotations
import argparse
import base64
import csv
import io
import json
import subprocess
import time
try:
    from .nim_photo_edit import configured_editors, local_json, MODELS
except ImportError:
    from nim_photo_edit import configured_editors, local_json, MODELS

def gpu_inventory():
    try:
        result=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,memory.free,driver_version","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=10)
        return [{"name":row[0].strip(),"total_mib":int(row[1]),"free_mib":int(row[2]),"driver":row[3].strip()} for row in csv.reader(result.stdout.splitlines()) if len(row)==4]
    except (OSError,subprocess.SubprocessError,ValueError):
        return []

def synthetic_payload(model):
    from PIL import Image,ImageDraw
    images=[]
    for color in ((180,40,40),(40,70,180),(60,120,180))[:min(MODELS[model],3)]:
        img=Image.new("RGB",(512,512),(240,240,240))
        ImageDraw.Draw(img).ellipse((100,100,412,412),fill=color)
        buf=io.BytesIO();img.save(buf,format="PNG")
        images.append("data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode("ascii"))
    return {"model":model,"prompt":"Change the colored circle in the first image to blue, preserve the background and framing.","image":images if MODELS[model]>1 else images[0],"n":1,"response_format":"b64_json"}

def probe(model,port,run_edit=False,request=local_json):
    row={"model":model,"port":port,"configured":True,"ready":False,"synthetic_edit_passed":False}
    try:
        row["ready"]=request(port,"/v1/health/ready",timeout=5) is True
        if not row["ready"] or not run_edit:return row
        from PIL import Image,ImageChops
        payload=synthetic_payload(model)
        encoded=payload["image"][0] if isinstance(payload["image"],list) else payload["image"]
        original=Image.open(io.BytesIO(base64.b64decode(encoded.split(",",1)[1]))).convert("RGB")
        start=time.monotonic()
        result=request(port,"/v1/images/edits",payload,timeout=360)
        data=result.get("data") if isinstance(result,dict) else None
        if not isinstance(data,list) or len(data)!=1:raise ValueError("invalid edit response")
        raw=base64.b64decode(data[0]["b64_json"],validate=True)
        with Image.open(io.BytesIO(raw)) as im:
            if im.format not in ("PNG","JPEG") or im.width*im.height>4_000_000 or abs(im.width/im.height-1)>.015:raise ValueError("invalid edit dimensions")
            final=im.convert("RGB").resize(original.size)
        changed=sum(1 for px in ImageChops.difference(original,final).getdata() if max(px)>=8)/(512*512)
        row.update(changed_ratio=round(changed,5),elapsed_seconds=round(time.monotonic()-start,3),synthetic_edit_passed=changed>=.03)
    except Exception as e:
        row["error"]=type(e).__name__
    return row

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edit",action="store_true",help="Run actual synthetic edits on every configured local editor")
    args=p.parse_args()
    editors=configured_editors()
    result={"gpu_inventory":gpu_inventory(),"editors":[probe(m,p,args.edit) for m,p in editors.items()],"note":"Readiness is not a real edit; synthetic edit is not a face-identity quality test. No photos or credentials are sent."}
    print(json.dumps(result,indent=2))
    if not result["gpu_inventory"] or not editors or any(not row["ready"] or args.edit and not row["synthetic_edit_passed"] for row in result["editors"]):return 1
    return 0

if __name__=="__main__":
    raise SystemExit(main())
