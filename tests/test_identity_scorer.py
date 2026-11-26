"""Persistent ArcFace scorer (v25): one model load per job, safe fallbacks."""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import facefusion_local_server as srv
from tools.identity_scorer import MAX_FAILURES, PersistentIdentityScorer

FAKE_WORKER = r'''
import json, os, sys, time
MODE = {mode!r}
print("FaceFusion log satırı", flush=True)
if MODE == "slow-start":
    time.sleep(5)
if MODE == "broken-start":
    print("RETAKE_IDENTITY_READY=" + json.dumps({{"ok": False, "error": "model yok"}}), flush=True)
    sys.exit(2)
print("RETAKE_IDENTITY_READY=" + json.dumps({{"ok": True}}), flush=True)
calls = 0
for line in sys.stdin:
    request = json.loads(line)
    if request.get("op") == "quit":
        break
    calls += 1
    if MODE == "crash-second" and calls == 2:
        sys.exit(3)
    if MODE == "stale-first":
        print("RETAKE_IDENTITY_REPORT=" + json.dumps({{"id": -1, "ok": True, "similarities": [0.0]}}), flush=True)
    if MODE == "error":
        print("RETAKE_IDENTITY_REPORT=" + json.dumps({{"id": request["id"], "ok": False, "error": "yüz yok"}}), flush=True)
        continue
    if MODE == "hang":
        time.sleep(30)
    values = [round(0.5 + 0.01 * index, 3) for index, _ in enumerate(request["probes"])]
    print("RETAKE_IDENTITY_REPORT=" + json.dumps({{"id": request["id"], "ok": True, "similarities": values, "pid": os.getpid()}}), flush=True)
'''

CENTROID = [0.1] * 512


class ScorerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.scorers = []

    def tearDown(self):
        for scorer in self.scorers:
            scorer.close()
        self.directory.cleanup()

    def scorer(self, mode="ok", **kwargs):
        worker = self.root / f"worker-{mode}.py"
        worker.write_text(FAKE_WORKER.format(mode=mode), encoding="utf-8")
        scorer = PersistentIdentityScorer(self.root, python=sys.executable, worker=worker, **kwargs)
        self.scorers.append(scorer)
        return scorer

    def probes(self, count=2):
        return [self.root / f"probe-{index}.png" for index in range(count)]

    def test_models_are_loaded_once_for_many_measurements(self):
        scorer = self.scorer()
        first = scorer.score(CENTROID, self.probes(3))
        second = scorer.score(CENTROID, self.probes(1))
        self.assertEqual(first, [0.5, 0.51, 0.52])
        self.assertEqual(second, [0.5])
        info = scorer.describe()
        self.assertEqual((info["starts"], info["calls"], info["failures"]), (1, 2, 0))
        self.assertTrue(info["running"])
        scorer.close()
        self.assertFalse(scorer.describe()["running"])
        # Closed between jobs: the next measurement starts it again.
        self.assertEqual(scorer.score(CENTROID, self.probes(1)), [0.5])
        self.assertEqual(scorer.describe()["starts"], 2)

    def test_stale_answers_and_log_lines_are_ignored(self):
        scorer = self.scorer("stale-first")
        self.assertEqual(scorer.score(CENTROID, self.probes(2)), [0.5, 0.51])

    def test_a_crash_falls_back_once_then_restarts(self):
        scorer = self.scorer("crash-second")
        self.assertIsNotNone(scorer.score(CENTROID, self.probes()))
        self.assertIsNone(scorer.score(CENTROID, self.probes()), "caller uses the one-shot path")
        self.assertFalse(scorer.disabled)
        self.assertIsNotNone(scorer.score(CENTROID, self.probes()), "a fresh worker answers again")
        self.assertEqual(scorer.describe()["starts"], 2)

    def test_timeouts_and_errors_disable_after_repeated_failures(self):
        scorer = self.scorer("hang", call_timeout=0.5)
        for _ in range(MAX_FAILURES):
            self.assertIsNone(scorer.score(CENTROID, self.probes()))
        self.assertTrue(scorer.disabled)
        self.assertIn("TimeoutError", scorer.describe()["last_error"])
        self.assertIsNone(scorer.score(CENTROID, self.probes()), "stays off until the engine restarts")

    def test_worker_errors_and_failed_start_return_none(self):
        self.assertIsNone(self.scorer("error").score(CENTROID, self.probes()))
        broken = self.scorer("broken-start")
        self.assertIsNone(broken.score(CENTROID, self.probes()))
        self.assertIn("model yok", broken.describe()["last_error"])
        slow = self.scorer("slow-start", ready_timeout=0.5)
        self.assertIsNone(slow.score(CENTROID, self.probes()))
        self.assertIsNone(self.scorer().score([], self.probes()), "no centroid: nothing to ask")


class WorkerServeTest(unittest.TestCase):
    def setUp(self):
        tools = str(Path(__file__).resolve().parents[1] / "tools")
        if tools not in sys.path:
            sys.path.insert(0, tools)
        import facefusion_identity_worker as worker
        self.worker = worker

    def test_serve_answers_each_line_with_loaded_models(self):
        loaded = []

        def embed(models, path):
            loaded.append(models)
            if path.name == "noface.png":
                raise RuntimeError("yüz yok")
            return [1.0] + [0.0] * 511

        stdin = io.StringIO("\n".join([
            json.dumps({"id": 1, "centroid": [1.0] + [0.0] * 511, "probes": ["a.png", "noface.png"]}),
            "not json",
            json.dumps({"id": 3, "centroid": [1.0], "probes": ["a.png"]}),
            json.dumps({"op": "quit"}),
            json.dumps({"id": 4, "centroid": [1.0] * 512, "probes": ["never.png"]}),
        ]) + "\n")
        stdout = io.StringIO()
        with patch.object(self.worker, "embed_largest_face", embed):
            code = self.worker.serve(Path("."), models="MODELS", stdin=stdin, stdout=stdout)
        self.assertEqual(code, 0)
        lines = stdout.getvalue().splitlines()
        self.assertEqual(json.loads(lines[0].split("=", 1)[1]), {"ok": True})
        first = json.loads(lines[1].split("=", 1)[1])
        self.assertEqual(first["id"], 1)
        self.assertEqual(first["similarities"][0], 1.0)
        self.assertIsNone(first["similarities"][1])
        self.assertFalse(json.loads(lines[2].split("=", 1)[1])["ok"], "bad JSON answers, does not die")
        third = json.loads(lines[3].split("=", 1)[1])
        self.assertEqual((third["id"], third["ok"]), (3, False), "a tiny centroid is refused")
        self.assertEqual(len(lines), 4, "nothing after quit")
        self.assertTrue(all(models == "MODELS" for models in loaded))

    def test_model_load_failure_is_reported_on_the_ready_line(self):
        stdout = io.StringIO()
        with patch.object(self.worker, "load_models", side_effect=RuntimeError("ArcFace yok")):
            self.assertEqual(self.worker.serve(Path("."), stdin=io.StringIO(""), stdout=stdout), 2)
        ready = json.loads(stdout.getvalue().split("=", 1)[1])
        self.assertFalse(ready["ok"])
        self.assertIn("ArcFace yok", ready["error"])


class EngineIntegrationTest(unittest.TestCase):
    def tearDown(self):
        srv.IDENTITY_SCORER = None

    def test_score_identity_prefers_the_persistent_scorer(self):
        class Fake:
            facefusion_dir = Path("ff")

            def __init__(self, answer):
                self.answer, self.calls = answer, 0

            def score(self, centroid, paths):
                self.calls += 1
                return self.answer

        srv.IDENTITY_SCORER = Fake([0.8])
        with patch.object(srv.subprocess, "run") as one_shot:
            self.assertEqual(srv.score_identity(Path("ff"), [0.1] * 512, [Path("p.png")]), [0.8])
            one_shot.assert_not_called()

        class Completed:
            stdout = 'RETAKE_IDENTITY_REPORT={"ok": true, "similarities": [0.7]}'
            stderr = ""
            returncode = 0

        with tempfile.TemporaryDirectory() as directory:
            probe = Path(directory) / "p.png"
            srv.IDENTITY_SCORER = Fake(None)  # persistent path failed
            with patch.object(srv.subprocess, "run", return_value=Completed()) as one_shot:
                self.assertEqual(srv.score_identity(Path("ff"), [0.1] * 512, [probe]), [0.7])
                one_shot.assert_called_once()
            srv.IDENTITY_SCORER = Fake([0.9])  # another FaceFusion folder: not this scorer
            with patch.object(srv.subprocess, "run", return_value=Completed()):
                self.assertEqual(srv.score_identity(Path("other"), [0.1] * 512, [probe]), [0.7])

    def test_memory_error_frees_the_scorer_before_lowering_quality(self):
        events = []

        class Running:
            def describe(self):
                return {"running": True}

            def close(self):
                events.append("closed")

        srv.IDENTITY_SCORER = Running()
        answers = iter([(False, "CUDA error: out of memory"), (True, "")])

        def fake_run(*args, **kwargs):
            events.append(args[5])
            return next(answers)

        with patch.object(srv, "run_facefusion", fake_run):
            used = srv.run_with_memory_fallback(Path("ff"), Path("t"), [Path("s")], Path("o"), "m", "1024x1024")
        self.assertEqual(used, "1024x1024", "the same quality setting after freeing memory")
        self.assertEqual(events, ["1024x1024", "closed", "1024x1024"])

    def test_job_cleanup_closes_the_scorer(self):
        closed = []

        class Scorer:
            def close(self):
                closed.append(True)

        srv.IDENTITY_SCORER = Scorer()

        class Server:
            pass

        stats = srv.cleanup_after_job(Server())
        self.assertTrue(stats["ok"])
        self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
