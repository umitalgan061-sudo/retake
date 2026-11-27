import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools import facefusion_local_server as srv
from tools.retake_cache import PassCache
from tools.local_flux2_editor import cleanup_editor
from tools.retake_background import build_args, write_status

class EngineLifecycleTest(unittest.TestCase):
    def test_clear_pass_cache_preserves_disk_and_returns_count(self):
        cache=PassCache()
        cache.put("a",b"first","768x768")
        cache.put("b",b"second","1024x1024")
        self.assertEqual(cache.clear(),2)
        self.assertEqual(len(cache),0)
        self.assertIsNone(cache.get("a"))
    def test_job_cleanup_always_runs_and_releases_lock(self):
        for succeeds in (True,False):
            with self.subTest(succeeds=succeeds):
                cache=PassCache()
                cache.put("pass",b"temporary","1024x1024")
                server=SimpleNamespace(facefusion_dir=Path("."),face_swapper_pixel_boost="1024x1024",
                                       face_enhancer_enabled=False,pass_cache=cache)
                job=srv.create_job()
                self.assertTrue(srv.INFERENCE_LOCK.acquire(blocking=False))
                with patch.object(srv,"run_ultra_search",return_value=(b"result",{}),
                                  side_effect=None if succeeds else RuntimeError("test error")):
                    srv.run_job(job,server,())
                self.assertFalse(srv.INFERENCE_LOCK.locked())
                self.assertEqual(job["status"],"done" if succeeds else "error")
                self.assertEqual(server.last_job_cleanup["pass_entries_cleared"],1)
                self.assertTrue(server.last_job_cleanup["ok"])
                self.assertEqual(len(cache),0)
                if succeeds:self.assertEqual(job["result"],b"result")
        with srv.JOBS_LOCK:srv.JOBS.clear()
    def test_result_expiry_never_evicts_running_work(self):
        now=10000
        with srv.JOBS_LOCK:
            srv.JOBS.clear()
            srv.JOBS["running"]={"id":"running","status":"running","created":0}
            srv.JOBS["expired"]={"id":"expired","status":"done","finished":now-srv.JOB_RESULT_TTL_SECONDS,"result":b"x"}
            for i in range(4):
                srv.JOBS[str(i)]={"id":str(i),"status":"done","finished":now-i,"result":b"result"}
        self.assertEqual(srv.prune_jobs(now),2)
        self.assertIn("running",srv.JOBS)
        self.assertIn("0",srv.JOBS)
        self.assertNotIn("expired",srv.JOBS)
        with srv.JOBS_LOCK:srv.JOBS.clear()
    def test_flux_cleanup_calls_offload_and_empty_cache_even_on_error(self):
        pipe,torch=Mock(),Mock()
        self.assertTrue(cleanup_editor(pipe,torch))
        pipe.maybe_free_model_hooks.assert_called_once()
        torch.cuda.empty_cache.assert_called_once()
        pipe.maybe_free_model_hooks.side_effect=RuntimeError("offload error")
        self.assertFalse(cleanup_editor(pipe,torch))
        self.assertEqual(torch.cuda.empty_cache.call_count,2)
    def test_background_builds_loopback_only_args_and_no_secrets_in_status(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg={"facefusion_dir":folder,"realism_dir":folder,"pixel_boost":"768x768","pairing":True,"enhancer":False,"shading":0.35}
            args=build_args(cfg)
            self.assertEqual(args[args.index("--host")+1],"127.0.0.1")
            self.assertIn("--enable-pairing",args)
            self.assertNotIn("--enable-face-enhancer",args)
            self.assertEqual(args[args.index("--shading-strength")+1],"0.35")
            status=Path(folder)/"status.json"
            write_status(status,"waiting_for_port")
            import json
            self.assertEqual(json.loads(status.read_text())["state"],"waiting_for_port")
            self.assertFalse(status.with_suffix(".tmp").exists())


class UpdateScriptContractTest(unittest.TestCase):
    def test_update_checks_manual_job_before_pull_and_again_before_stop(self):
        script=(Path(__file__).resolve().parents[1]/"tools"/"update_retake_ultra.ps1").read_text(encoding="utf-8-sig")
        start=script.index("$manualHealth = Invoke-RestMethod")
        self.assertLess(start,script.index("git pull --ff-only origin main"))
        stop=script.index("Stop-Process -Id $bridge.ProcessId")
        self.assertLess(script.index("Manual engine state changed"),stop)
        self.assertLess(script.index("$same.CreationDate"),stop)
        self.assertIn("Another or unidentified Retake worktree",script)

    def test_deploy_verifies_live_bytes_and_managed_revision_first(self):
        script=(Path(__file__).resolve().parents[1]/"tools"/"update_retake_ultra.ps1").read_text(encoding="utf-8-sig")
        body=script.split("function Publish-VerifiedRetakeWeb {",1)[1].split("Push-Location $Root",1)[0]
        self.assertIn('foreach ($file in @("app.js", "src/diagnostics.js", "sw.js"))',body)
        self.assertIn("Get-FileHash -Algorithm SHA256",body)
        self.assertIn("https://retake06-ua.web.app/",body)
        self.assertNotIn("X-Retake-Token",body)
        self.assertNotIn("pairing-token",body)
        self.assertIn("if (-not $verified) { throw",body)
        managed=script.split("Push-Location $Root",1)[1]
        self.assertLess(managed.index("Managed Retake engine verified"),managed.index("Publish-VerifiedRetakeWeb"))
