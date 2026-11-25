"""Quiet Windows Task Scheduler entry point; no administrator rights or cloud."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

def write_status(path, state, **details):
    try:
        tmp=path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"state":state,"pid":os.getpid(),"updated_at":time.time(),**details}),encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass

def port_open(port=8765):
    try:
        with socket.create_connection(("127.0.0.1",port),timeout=1):
            return True
    except OSError:
        return False

def build_args(cfg):
    args=["--facefusion-dir",cfg["facefusion_dir"],"--host","127.0.0.1","--port","8765",
          "--face-swapper-pixel-boost",cfg.get("pixel_boost","1024x1024")]
    if cfg.get("pairing"):args.append("--enable-pairing")
    if cfg.get("enhancer"):args.append("--enable-face-enhancer")
    if cfg.get("shading") is not None:args.extend(["--shading-strength",str(cfg["shading"])])
    realism=Path(cfg["realism_dir"])
    marker=realism/"python.txt"
    if not cfg.get("disable_realism") and marker.is_file():
        args.extend(["--realism-python",marker.read_text(encoding="utf-8-sig").strip(),"--realism-models",str(realism/"models")])
    return args

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args()
    status=args.config.with_name("background-status.json")
    try:
        cfg=json.loads(args.config.read_text(encoding="utf-8-sig"))
        if not (Path(cfg["facefusion_dir"])/"facefusion.py").is_file():
            raise FileNotFoundError("FaceFusion")
        # Capture the activated Conda PATH at installation, not the base env
        # from Task Scheduler. CUDA runtime DLLs live in Library/bin.
        python=Path(cfg["python"]).resolve()
        if not python.is_file() or python.parent!=Path(sys.executable).resolve().parent:
            raise RuntimeError("Conda environment changed; reinstall startup task")
        os.environ["PATH"]=cfg["activated_path"]
        os.environ["CONDA_PREFIX"]=str(python.parent)
        os.environ["RETAKE_BACKGROUND"]="1"
        os.environ["RETAKE_NIM_EDITORS"]=cfg.get("editors","")
        sys.executable=str(python)
        dll_handles=[]
        if hasattr(os,"add_dll_directory"):
            for folder in (python.parent,python.parent/"Library"/"bin",python.parent/"DLLs"):
                if folder.is_dir():dll_handles.append(os.add_dll_directory(str(folder)))
        # Do not terminate another engine or unknown service on 8765.
        while port_open():
            write_status(status,"waiting_for_port")
            time.sleep(5)
        from facefusion_local_server import main as serve
        sys.argv=[str(Path(__file__).with_name("facefusion_local_server.py")),*build_args(cfg)]
        write_status(status,"starting")
        # Raw logs may include a pairing token, temporary photo paths and
        # diagnostics. Keep them out of persistent logs; status contains only
        # operational metadata. Health provides live details.
        with open(os.devnull,"w",encoding="utf-8") as sink:
            sys.stdout=sys.stderr=sink
            write_status(status,"running")
            code=serve()
        write_status(status,"stopped",exit_code=code)
        return code if code else 1
    except Exception as error:
        write_status(status,"failed",error_type=type(error).__name__)
        return 1

if __name__=="__main__":
    raise SystemExit(main())
