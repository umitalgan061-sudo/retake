import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from tools import managed_flux2 as managed
from tools.nim_photo_edit import configured_editors

class ManagedFluxTest(unittest.TestCase):
    def test_requires_explicit_probe_record_and_local_files(self):
        with tempfile.TemporaryDirectory() as d, patch.object(Path,"home",return_value=Path(d)):
            root=Path(d)
            self.assertIsNone(managed.managed_config())
            python=root/"retake-flux2-venv"/"Scripts"/"python.exe"
            model=root/"retake-models"/"FLUX.2-klein-4B"/"model_index.json"
            for file in (python,model):
                file.parent.mkdir(parents=True,exist_ok=True); file.touch()
            proof=managed.proof_path()
            proof.parent.mkdir()
            record={"model":managed.MODEL,"ready":True,"synthetic_edit_passed":True,"offload":"sequential","max_edge":512}
            proof.write_text(json.dumps(record))
            self.assertIsNotNone(managed.managed_config())
            with patch.dict("os.environ",{"RETAKE_NIM_EDITORS":""}):
                self.assertEqual(configured_editors(),{managed.MODEL:0})
            self.assertEqual(configured_editors(""),{})
            proof.write_text(json.dumps(dict(record,synthetic_edit_passed=False)))
            self.assertIsNone(managed.managed_config())

    def test_worker_is_disposable_and_does_not_listen_on_port(self):
        with patch.object(managed,"managed_config",return_value=(Path("python.exe"),Path("model"))):
            self.assertTrue(managed.managed_request(0,"/v1/health/ready"))
            with self.assertRaises(ValueError):
                managed.managed_request(8004,"/v1/health/ready")
            output_path=[]
            def execute(args,**kwargs):
                self.assertIn("--one-shot-output",args)
                self.assertNotIn("--port",args)
                self.assertEqual(kwargs["timeout"],360)
                self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"],"1")
                self.assertEqual(json.loads(kwargs["input"])["model"],managed.MODEL)
                path=Path(args[-1]); output_path.append(path)
                path.write_text(json.dumps({"data":[{"b64_json":"test"}]}))
                return SimpleNamespace(returncode=0)
            with patch.object(managed.subprocess,"run",side_effect=execute):
                result=managed.managed_request(0,"/v1/images/edits",{"model":managed.MODEL})
            self.assertEqual(result["data"][0]["b64_json"],"test")
            self.assertFalse(output_path[0].exists())

    def test_worker_failure_does_not_return_fake_image(self):
        with patch.object(managed,"managed_config",return_value=(Path("python.exe"),Path("model"))), patch.object(managed.subprocess,"run",return_value=SimpleNamespace(returncode=1)):
            with self.assertRaises(RuntimeError):
                managed.managed_request(0,"/v1/images/edits",{"model":managed.MODEL})
