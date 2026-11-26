"""Quiet Windows Task Scheduler entry point; no administrator rights or cloud."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

try:
    from .retake_event_log import EVENT_LOG_NAME, EventLog, EventLogSink
except ImportError:  # Task Scheduler runs this file directly from tools/.
    from retake_event_log import EVENT_LOG_NAME, EventLog, EventLogSink

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

def open_event_log(config_path):
    """Scrubbed operational log next to the config; RETAKE_EVENT_LOG=off disables."""
    if os.environ.get("RETAKE_EVENT_LOG","").strip().lower() in {"0","off","false","no"}:
        return None
    return EventLog(Path(config_path).with_name(EVENT_LOG_NAME))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args()
    status=args.config.with_name("background-status.json")
    events=open_event_log(args.config)
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
        import facefusion_local_server as engine
        serve=engine.main
        sys.argv=[str(Path(__file__).with_name("facefusion_local_server.py")),*build_args(cfg)]
        write_status(status,"starting")
        if events:
            events.record("start","Gizli motor başlatıldı",
                          revision=str(getattr(engine,"QUALITY_PIPELINE_REVISION","")),
                          boost=str(cfg.get("pixel_boost","1024x1024")))
        # Raw logs may include a pairing token, temporary photo paths and
        # diagnostics. Never persist them: only scrubbed operational lines
        # (memory fallbacks, skipped steps, jury and job summaries) reach
        # engine-events.jsonl; everything else is discarded as before.
        original_stdout,original_stderr=sys.stdout,sys.stderr
        sink=EventLogSink(events) if events else open(os.devnull,"w",encoding="utf-8")
        with sink:
            sys.stdout=sys.stderr=sink
            try:
                write_status(status,"running")
                code=serve()
            finally:
                sys.stdout,sys.stderr=original_stdout,original_stderr
        write_status(status,"stopped",exit_code=code)
        if events:
            events.record("stop","Gizli motor durdu",code=int(code or 0))
        return code if code else 1
    except Exception as error:
        write_status(status,"failed",error_type=type(error).__name__)
        if events:
            # Exception type only: messages can contain local paths.
            events.record("error",f"Gizli motor çalışamadı: {type(error).__name__}")
        return 1

if __name__=="__main__":
    raise SystemExit(main())
