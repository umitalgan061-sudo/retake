import base64
import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
import zlib
from collections import namedtuple
from pathlib import Path

from tools import retake_doctor as doctor
from tools.retake_event_log import EVENT_LOG_NAME, EventLog

PAIRING = "pairQ8Zr2vX0b9yTq8Zr2vX0b9yTq8Zr2vX0b9yT"
PROJECT = "projQ8Zr2vX0b9yTq8Zr2vX0b9yTq8Zr2vX0b9yTxyz"
Usage = namedtuple("Usage", "total used free")
RELEASE = "7fea8d25e452e7ea87b6fa66ad3ab33b8ddbd16f"


class FakeNet:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, url, *, headers=None, data=None, timeout=10.0, method=None):
        self.calls.append({"url": url, "headers": dict(headers or {}), "data": data, "method": method})
        for prefix, reply in self.routes.items():
            if url.startswith(prefix):
                if isinstance(reply, Exception):
                    raise reply
                if callable(reply):
                    return reply(url, headers or {}, data)
                status, body = reply
                return status, {}, body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        raise ConnectionRefusedError(url)


class FakeRun:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def __call__(self, args, timeout):
        self.calls.append(args)
        key = " ".join(args[3:] if args[0] == "git" else args)
        for prefix, reply in self.outputs.items():
            if key.startswith(prefix):
                return reply
        return 127, ""


def healthy_routes(revision="portrait-quality-test-r9"):
    health = {
        "ok": True, "quality_revision": revision, "background_mode": True, "busy": False, "cuda": True,
        "pixel_boost": "1024x1024", "realism": True, "personal_model": False, "face_enhancer_enabled": False,
        "nvidia_judge": "configured", "identity_scorer": {"enabled": True, "running": True, "failures": 0},
        "pass_cache": {"last_job": {"hits": 12, "misses": 30}}, "providers": ["CUDAExecutionProvider"],
    }
    return {
        doctor.ENGINE_URL + "/health": (200, health),
        doctor.ENGINE_URL + "/api/calibration": (200, {"enabled": True, "banks": {"a": {"jobs": 4, "self_tests": 2}}}),
        doctor.HAFIZE_URL + "/api/health": (200, {"projectAiGateway": {"enabled": True, "visionJudge": {
            "models": ["nvidia/a", "meta/b", "google/c", "mistralai/d"], "reserveModels": ["x/e"],
            "layouts": ["face", "review-sheet"], "projects": ["retake"]}}}),
        doctor.SITE_URL + "/retake-release.json": (200, {"commit": RELEASE, "site": "retake06-ua"}),
        doctor.SITE_URL + "/sw.js": (200, b'const SHELL_CACHE = "retake-shell-v38";'),
    }


class DoctorTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "tools").mkdir()
        (self.root / "tools" / "facefusion_local_server.py").write_text(
            'X = 1\nQUALITY_PIPELINE_REVISION = "portrait-quality-test-r9"\n', encoding="utf-8")
        (self.root / "sw.js").write_text('const SHELL_CACHE = "retake-shell-v38";\n', encoding="utf-8")
        self.home = Path(tempfile.mkdtemp())
        self.retake = self.home / ".retake"
        self.retake.mkdir()
        facefusion = self.home / "FaceFusion"
        facefusion.mkdir()
        (facefusion / "facefusion.py").write_text("", encoding="utf-8")
        realism = self.home / "Realism"
        realism.mkdir()
        (realism / "python.txt").write_text("x", encoding="utf-8")
        self.python = self.home / "env" / "python.exe"
        self.python.parent.mkdir()
        self.python.write_text("", encoding="utf-8")
        (self.retake / "background-config.json").write_text(json.dumps({
            "python": str(self.python), "facefusion_dir": str(facefusion), "pixel_boost": "1024x1024",
            "pairing": True, "enhancer": False, "disable_realism": False, "realism_dir": str(realism), "editors": "",
        }), encoding="utf-8")
        (self.retake / "background-status.json").write_text(json.dumps({"state": "running", "updated_at": 1000.0}), encoding="utf-8")
        (self.retake / "pairing-token").write_text(PAIRING, encoding="utf-8")
        (self.retake / "hafize-project-token").write_text(PROJECT, encoding="utf-8")
        self.runner = FakeRun({
            "rev-parse --short HEAD": (0, "abc1234\n"),
            "rev-parse --abbrev-ref HEAD": (0, "main\n"),
            "rev-parse HEAD": (0, RELEASE + "\n"),
            "status --porcelain": (0, ""),
            "fetch --quiet origin main": (0, ""),
            "rev-list --left-right --count HEAD...origin/main": (0, "0\t0\n"),
            "nvidia-smi --query-gpu": (0, "NVIDIA RTX A4000, 16376, 900, 572.16, 41, 3\n"),
            "nvidia-smi --query-compute-apps": (0, ""),
        })
        self.net = FakeNet(healthy_routes())
        self.packages = {"onnxruntime-gpu": "1.20.1", "numpy": "1.26.4", "opencv-python": "4.12.0.88"}

    def context(self, **overrides):
        values = dict(
            root=self.root, home=self.home, opener=self.net, run=self.runner,
            windows={"task": {"exists": True, "state": "Running", "last_result": 0, "working_dir_matches": True},
                     "port": {"listening": True, "owner_is_retake": True}},
            now=2000.0, executable=str(self.python),
            env={"USERNAME": "umit", "COMPUTERNAME": "ABB-PC-123", "USERPROFILE": str(self.home)},
            disk_usage=lambda folder: Usage(10 ** 12, 10 ** 11, 400 * 1024 ** 3),
            package_version=lambda name: self.packages.get(name),
        )
        values.update(overrides)
        return doctor.Context(**values)

    def statuses(self, checks):
        return {status: [check.title for check in checks if check.status == status] for status in ("ok", "warn", "fail", "info")}

    def test_healthy_setup_has_no_warnings_and_says_so(self):
        log = EventLog(self.retake / EVENT_LOG_NAME, clock=lambda: 1990.0)
        log.record_line("[Ultra] Sonuç: 19 doğrulanmış tur, kazanan 7. tur; benzerlik=0.712; süre 1510 sn.")
        checks = doctor.collect(self.context())
        result = self.statuses(checks)
        self.assertEqual(result["fail"], [])
        self.assertEqual(result["warn"], [])
        report = doctor.render(checks, now=2000.0)
        self.assertIn("Önerilen adım yok", report)
        self.assertIn("portrait-quality-test-r9", report)
        self.assertIn("opencv-python v4.12.0.88", report)
        self.assertIn("benzerlik=0.712", report)
        self.assertTrue(report.startswith("===== RETAKE TANI RAPORU"))
        self.assertTrue(report.rstrip().endswith("===== RAPOR SONU ====="))

    def test_report_never_contains_secrets_user_or_paths(self):
        routes = healthy_routes()
        routes[doctor.ENGINE_URL + "/health"][1]["provider_error"] = r"C:\Users\umit\miniconda3\envs\x\onnx.dll failed"
        routes[doctor.ENGINE_URL + "/health"][1]["cuda"] = False
        routes[doctor.ENGINE_URL + "/health"][1]["identity_scorer"] = {"enabled": True, "failures": 2,
                                                                        "last_error": f"C:\\Users\\umit\\x {PAIRING}"}
        self.net.routes = routes
        self.runner.outputs["nvidia-smi --query-compute-apps"] = (0, r"4242, C:\Users\umit\AppData\Pro\ArcGISPro.exe, 3100" + "\n")
        self.runner.outputs["status --porcelain"] = (0, "?? umit-photo.jpg\n?? ABB-PC-123.png\n")
        log = EventLog(self.retake / EVENT_LOG_NAME, clock=lambda: 1990.0)
        log.record_line(r"[Gerçekçilik] Başarısız (kod 1): C:\Users\umit\Temp\target.png")
        ctx = self.context(probe_jury=True)
        self.net.routes[doctor.HAFIZE_URL + "/api/project-ai"] = (200, {"decision": "B", "votes": 4, "models": ["nvidia/a"]})
        checks = doctor.collect(ctx)
        report = doctor.render(checks)
        payload = json.dumps([vars(check) for check in checks], ensure_ascii=False)
        for text in (report, payload):
            for secret in (PAIRING, PROJECT, "umit", "ABB-PC-123", str(self.home), "target.png", "photo.jpg"):
                self.assertNotIn(secret, text)
        self.assertIn("ArcGISPro.exe", report)
        # The engine request carried the pairing code; the report did not.
        self.assertEqual(self.net.calls[0]["headers"].get("X-Retake-Token"), PAIRING)

    def test_outdated_running_engine_is_a_problem_with_the_update_command_first(self):
        self.net.routes[doctor.ENGINE_URL + "/health"][1]["quality_revision"] = "portrait-quality-old-r7"
        checks = doctor.collect(self.context())
        self.assertTrue(any("eski kalite sürümünde" in title for title in self.statuses(checks)["fail"]))
        report = doctor.render(checks)
        steps = report.split("Önerilen adımlar")[1]
        self.assertIn("1. .\\tools\\update_retake_ultra.ps1 -SkipPull -SkipDeploy", steps)

    def test_unreachable_engine_and_wrong_pairing(self):
        self.net.routes[doctor.ENGINE_URL + "/health"] = ConnectionRefusedError("refused")
        result = self.statuses(doctor.collect(self.context()))
        self.assertIn("Motor yanıt vermiyor (127.0.0.1:8765).", result["fail"])
        self.net.routes[doctor.ENGINE_URL + "/health"] = (401, {"ok": False, "pairing_required": True})
        result = self.statuses(doctor.collect(self.context()))
        self.assertTrue(any("eşleştirme kodu" in title for title in result["fail"]))

    def test_behind_repo_dirty_tree_and_other_branch(self):
        self.runner.outputs["rev-list --left-right --count HEAD...origin/main"] = (0, "0\t3\n")
        self.runner.outputs["status --porcelain"] = (0, " M tools/x.py\n")
        self.runner.outputs["rev-parse --abbrev-ref HEAD"] = (0, "feature\n")
        result = self.statuses(doctor.check_repo(self.context()))
        self.assertTrue(any("3 commit geride" in title for title in result["warn"]))
        self.assertTrue(any("1 dosyada yerel değişiklik" in title for title in result["warn"]))

    def test_offline_mode_contacts_only_the_local_engine(self):
        doctor.collect(self.context(offline=True))
        self.assertTrue(self.net.calls)
        self.assertTrue(all(call["url"].startswith(doctor.ENGINE_URL) for call in self.net.calls))
        self.assertFalse(any("fetch" in args for args in self.runner.calls))

    def test_gpu_contention_and_low_free_memory(self):
        self.runner.outputs["nvidia-smi --query-compute-apps"] = (
            0, "11, ArcGISPro.exe, 2600\n12, python.exe, 9000\n13, chrome.exe, 300\n")
        checks = doctor.check_gpu(self.context(), {"busy": False})
        warning = [check for check in checks if check.status == "warn"][0]
        self.assertIn("2900 MB", warning.title)
        self.assertIn("ArcGISPro.exe 2600 MB", warning.title)
        self.assertNotIn("python", warning.title)
        self.runner.outputs["nvidia-smi --query-compute-apps"] = (0, "1, ArcGISPro.exe, [N/A]\n")
        self.runner.outputs["nvidia-smi --query-gpu"] = (0, "NVIDIA RTX A4000, 16376, 12000, 572.16, 41, 3\n")
        checks = doctor.check_gpu(self.context(), {"busy": False})
        self.assertTrue(any("boş" in check.title and check.status == "warn" for check in checks))
        self.runner.outputs["nvidia-smi --query-gpu"] = (127, "")
        self.assertEqual(doctor.check_gpu(self.context(), None)[0].status, "warn")

    def test_conflicting_onnxruntime_packages(self):
        self.packages["onnxruntime"] = "1.20.1"
        result = self.statuses(doctor.check_packages(self.context()))
        self.assertTrue(any("Hem 'onnxruntime' hem 'onnxruntime-gpu'" in title for title in result["warn"]))

    def test_startup_task_for_another_worktree_and_foreign_port_owner(self):
        windows = {"task": {"exists": True, "state": "Running", "working_dir_matches": False},
                   "port": {"listening": True, "owner_is_retake": False}}
        result = self.statuses(doctor.check_background(self.context(windows=windows)))
        self.assertEqual(len(result["fail"]), 2)

    def test_low_boost_disabled_realism_and_failed_launcher(self):
        config = json.loads((self.retake / "background-config.json").read_text(encoding="utf-8"))
        config.update(pixel_boost="768x768", disable_realism=True)
        (self.retake / "background-config.json").write_text(json.dumps(config), encoding="utf-8")
        (self.retake / "background-status.json").write_text(json.dumps({"state": "failed", "error_type": "RuntimeError", "updated_at": 1}), encoding="utf-8")
        checks = doctor.check_background(self.context())
        result = self.statuses(checks)
        self.assertTrue(any("768x768" in title and "gerçekçilik kapalı (ayar)" in title for title in result["warn"]))
        self.assertTrue(any("RuntimeError" in title for title in result["fail"]))

    def test_event_problems_become_warnings_with_recent_lines(self):
        log = EventLog(self.retake / EVENT_LOG_NAME, clock=lambda: 1990.0)
        log.record_line("hyperswap_1a_256: 1024x1024 bellek hatası; 768x768 ayarına düşülüyor.")
        log.record_line("[NVIDIA jüri] Atlandı, yerel seçim korundu: HTTP 502")
        checks = doctor.check_events(self.context())
        result = self.statuses(checks)
        self.assertTrue(any("GPU bellek hatası" in title for title in result["warn"]))
        self.assertTrue(any("jüri atlanması" in title for title in result["warn"]))
        self.assertTrue(any("[memory]" in title for title in result["info"]))
        later = doctor.check_events(self.context(now=10 ** 7))
        self.assertFalse(any(check.status == "warn" for check in later))

    def test_jury_probe_sends_two_generated_cards_and_reads_votes(self):
        seen = {}

        def judge(url, headers, data):
            seen["headers"], seen["body"] = headers, json.loads(data)
            return 200, {}, json.dumps({"decision": "B", "votes": 6, "models": ["nvidia/a", "meta/b", "google/c"]}).encode()

        self.net.routes[doctor.HAFIZE_URL + "/api/project-ai/retake/vision-judge"] = judge
        checks = doctor.check_jury(self.context(probe_jury=True), {"nvidia_judge": "configured"})
        probe = checks[-1]
        self.assertEqual(probe.status, "ok")
        self.assertIn("6 oy, 3 model", probe.title)
        self.assertEqual(seen["headers"]["Authorization"], "Bearer " + PROJECT)
        self.assertEqual(set(seen["body"]), {"task", "candidates"})
        self.assertEqual(len(seen["body"]["candidates"]), 2)
        for uri in seen["body"]["candidates"]:
            raw = base64.b64decode(uri.split(",", 1)[1])
            self.assertEqual(raw[:8], b"\x89PNG\r\n\x1a\n")
        self.net.routes[doctor.HAFIZE_URL + "/api/project-ai/retake/vision-judge"] = (401, {"error": "PROJECT_AI_UNAUTHORIZED"})
        failed = doctor.check_jury(self.context(probe_jury=True), {"nvidia_judge": "configured"})[-1]
        self.assertEqual(failed.status, "fail")
        self.assertIn("PROJECT_AI_UNAUTHORIZED", failed.title)
        self.assertIn("HAFIZE_PROJECT_AI_TOKEN_HASHES", failed.fix)

    def test_jury_states_without_token_old_hafize_and_benched_models(self):
        (self.retake / "hafize-project-token").unlink()
        self.assertEqual(doctor.check_jury(self.context(), None)[0].status, "info")
        (self.retake / "hafize-project-token").write_text("short", encoding="utf-8")
        self.assertEqual(doctor.check_jury(self.context(), None)[0].status, "fail")
        (self.retake / "hafize-project-token").write_text(PROJECT, encoding="utf-8")
        judge = self.net.routes[doctor.HAFIZE_URL + "/api/health"][1]["projectAiGateway"]["visionJudge"]
        judge["layouts"] = ["face"]
        judge["health"] = {"benched": ["meta/b"], "models": {"nvidia/a": {"calls": 8, "valid": 8}, "meta/b": {"calls": 4, "valid": 0}}}
        result = self.statuses(doctor.check_jury(self.context(), {"nvidia_judge": "not-configured"}))
        self.assertTrue(any("inceleme paftasını tanımıyor" in title for title in result["warn"]))
        self.assertTrue(any("a 8/8" in title and "b 0/4" in title for title in result["warn"]))
        self.assertTrue(any("motor jüriyi görmüyor" in title for title in result["warn"]))

    def test_direct_nvidia_jury_is_recognised_without_reading_out_the_key(self):
        (self.retake / "hafize-project-token").unlink()
        secret = "nvapi-" + "Q" * 60
        (self.root / ".env").write_text(f"# local\nOTHER=1\nNVIDIA_API_KEY=\"{secret}\"\n", encoding="utf-8")
        health = {"nvidia_judge": "configured", "nvidia_judge_transport": "direct"}
        self.runner.outputs[str(self.python)] = (0, "NVIDIA jürisi yolu: doğrudan NVIDIA (.env).\n"
                                                    "NVIDIA jürisi çalışıyor (review-sheet): karar=winner kazanan=1 oy=3\n")
        checks = doctor.check_jury(self.context(probe_jury=True), health)
        self.assertEqual([check.status for check in checks], ["ok", "ok"])
        self.assertIn("(.env;", checks[0].title)
        self.assertIn("karar=winner", checks[1].detail)
        report = doctor.render(checks)
        self.assertNotIn(secret, report)
        self.assertNotIn("Q" * 32, report)
        # No Hafize traffic for the direct route.
        self.assertFalse(any(call["url"].startswith(doctor.HAFIZE_URL) for call in self.net.calls))
        self.runner.outputs[str(self.python)] = (1, "NVIDIA jüri testi başarısız: HTTP 401\n")
        self.assertEqual(doctor.check_jury(self.context(probe_jury=True), health)[-1].status, "fail")
        self.runner.outputs[str(self.python)] = (1, "ModuleNotFoundError: No module named 'numpy'\n")
        self.assertEqual(doctor.check_jury(self.context(probe_jury=True), health)[-1].status, "warn")
        stale = doctor.check_jury(self.context(), {"nvidia_judge": "not-configured", "nvidia_judge_transport": "none"})
        self.assertTrue(any(check.status == "warn" and "görmüyor" in check.title for check in stale))

    def test_direct_key_sources(self):
        ctx = self.context()
        self.assertIsNone(doctor.direct_key_source(ctx))
        (self.root / ".dev.vars").write_bytes("NVIDIA_API_KEY=abc\r\n".encode("utf-16"))
        self.assertEqual(doctor.direct_key_source(ctx), ".dev.vars")
        (self.root / ".env").write_text("NVIDIA_API_KEY=\n", encoding="utf-8")
        self.assertEqual(doctor.direct_key_source(ctx), ".dev.vars")
        self.assertEqual(doctor.direct_key_source(self.context(env={"NVIDIA_API_KEY": "x"})), "ortam değişkeni")
        self.assertIsNone(doctor.direct_key_source(self.context(env={"NVIDIA_API_KEY": "x", "RETAKE_NVIDIA_JUDGE": "off"})))

    def test_identity_model_files(self):
        (self.root / "tools" / "facefusion_local_server.py").write_text(
            'QUALITY_PIPELINE_REVISION = "portrait-quality-test-r9"\n'
            'IDENTITY_MODEL = "hyperswap_1b_256"\nIDENTITY_FALLBACK_MODEL = "alphaface_256"\n', encoding="utf-8")
        models = self.home / "FaceFusion" / ".assets" / "models"
        self.assertEqual(doctor.check_models(self.context(), None)[0].status, "info")
        models.mkdir(parents=True)
        self.assertIn("henüz yok", doctor.check_models(self.context(), None)[0].title)
        (models / "alphaface_256.onnx").write_bytes(b"x" * 10)
        warning = doctor.check_models(self.context(), None)[0]
        self.assertEqual(warning.status, "warn")
        self.assertIn("hyperswap_1b_256", warning.title)
        self.assertIn("yedek alphaface_256 hazır", warning.title)
        (models / "hyperswap_1b_256.onnx").write_bytes(b"x" * 10)
        ready = doctor.check_models(self.context(), {"identity_model": "hyperswap_1b_256"})[0]
        self.assertEqual(ready.status, "ok")
        self.assertIn("2 model", ready.title)
        self.assertEqual(doctor.check_models(self.context(), {"identity_model": "hyperswap_1c_256"})[0].status, "warn")

    def test_identity_fallback_and_detail_events_warn(self):
        log = EventLog(self.retake / EVENT_LOG_NAME, clock=lambda: 1990.0)
        log.record_line("hyperswap_1b_256 kullanılamadı (RuntimeError: x); alphaface_256 ile devam ediliyor.")
        log.record_line("Uyarı: son fotoğraf bölgesel ayrıntı kontrolünü geçemedi (eyes).")
        titles = [check.title for check in doctor.check_events(self.context()) if check.status == "warn"]
        self.assertTrue(any("yedeğe düşmesi" in title for title in titles))
        self.assertTrue(any("bölgesel ayrıntı" in title for title in titles))

    def test_site_release_marker_is_compared_by_web_files(self):
        self.assertEqual(doctor.check_site(self.context())[0].title, "Yayındaki site bu depoyla aynı commit'te (7fea8d2).")
        self.runner.outputs["rev-parse HEAD"] = (0, "1" * 40 + "\n")
        self.runner.outputs[f"cat-file -e {RELEASE}^{{commit}}"] = (0, "")
        self.runner.outputs[f"diff --quiet {RELEASE} HEAD --"] = (0, "")
        same = doctor.check_site(self.context())[0]
        self.assertEqual(same.status, "ok")
        self.assertIn("aynı web dosyalarını", same.title)
        diff_args = [args for args in self.runner.calls if "diff" in args][-1]
        self.assertEqual(diff_args[-len(doctor.WEB_PATHS):], list(doctor.WEB_PATHS))
        self.runner.outputs[f"diff --quiet {RELEASE} HEAD --"] = (1, "")
        self.runner.outputs[f"merge-base --is-ancestor {RELEASE} HEAD"] = (0, "")
        older = doctor.check_site(self.context())[0]
        self.assertEqual(older.status, "warn")
        self.assertIn("eski web dosyaları", older.title)
        self.runner.outputs[f"merge-base --is-ancestor {RELEASE} HEAD"] = (1, "")
        self.assertIn("daha yeni", doctor.check_site(self.context())[0].title)
        self.runner.outputs[f"cat-file -e {RELEASE}^{{commit}}"] = (1, "")
        self.assertEqual(doctor.check_site(self.context())[0].status, "info")

    def test_site_falls_back_to_the_cache_name_without_a_marker(self):
        self.net.routes[doctor.SITE_URL + "/retake-release.json"] = (404, b"Not Found")
        self.assertEqual(doctor.check_site(self.context())[0].status, "info")
        self.net.routes[doctor.SITE_URL + "/sw.js"] = (200, b'const SHELL_CACHE = "retake-shell-v37";')
        self.assertEqual(doctor.check_site(self.context())[0].status, "warn")
        self.net.routes[doctor.SITE_URL + "/retake-release.json"] = (200, {"commit": "not-a-sha"})
        self.assertEqual(doctor.check_site(self.context())[0].status, "warn")

    def test_disk_thresholds(self):
        ctx = self.context(disk_usage=lambda folder: Usage(10 ** 12, 0, 3 * 1024 ** 3))
        self.assertEqual(doctor.check_disk(ctx)[0].status, "fail")
        ctx = self.context(disk_usage=lambda folder: Usage(10 ** 12, 0, 10 * 1024 ** 3))
        self.assertEqual(doctor.check_disk(ctx)[0].status, "warn")

    def test_render_orders_problems_first_and_deduplicates_steps(self):
        checks = [
            doctor.Check("A", "warn", "w", fix="adım-1"),
            doctor.Check("B", "fail", "f", fix="adım-2"),
            doctor.Check("C", "warn", "w2", fix="adım-1"),
            doctor.Check("D", "info", "i", fix="bilgi adımı"),
        ]
        steps = doctor.render(checks).split("Önerilen adımlar")[1]
        self.assertLess(steps.index("adım-2"), steps.index("adım-1"))
        self.assertEqual(steps.count("adım-1"), 1)
        self.assertNotIn("bilgi adımı", steps)

    def test_model_ids_stay_readable_but_tokens_do_not(self):
        self.assertEqual(doctor.redact("Panel: nemotron-3-nano-omni-30b-a3b-reasoning, llama-3.2-90b-vision-instruct", {}),
                         "Panel: nemotron-3-nano-omni-30b-a3b-reasoning, llama-3.2-90b-vision-instruct")
        for secret in (PAIRING, PROJECT, "f" * 40, "q8Zr2vX0b9yTq8Zr2vX0b9yTq8Zr2vX0b9yT", "abcdefghijklmnopqrstuvwxyz0123456789"):
            self.assertEqual(doctor.redact(f"x {secret} y", {}), "x <gizli> y", secret)

    def test_task_results_are_named(self):
        self.assertEqual(doctor.task_result(267009), "görev şu an çalışıyor")
        self.assertEqual(doctor.task_result(0), "başarılı")
        self.assertEqual(doctor.task_result(-2147020576), "kullanıcı oturumu yok")
        self.assertEqual(doctor.task_result(5), "kod 5 (0x00000005)")
        windows = {"task": {"exists": True, "state": "Running", "last_result": 267009, "working_dir_matches": True}}
        titles = [check.detail for check in doctor.check_background(self.context(windows=windows))]
        self.assertIn("Son çalışma sonucu: görev şu an çalışıyor.", titles)

    def test_redact_uses_word_boundaries(self):
        env = {"USERNAME": "ali", "COMPUTERNAME": "PC-01"}
        self.assertEqual(doctor.redact("kalite ali PC-01", env), "kalite <kullanıcı> <bilgisayar>")

    def test_generated_png_is_valid(self):
        data = doctor.solid_png((10, 20, 30), size=8)
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", data[16:24])
        self.assertEqual((width, height), (8, 8))
        length = struct.unpack(">I", data[33:37])[0]
        raw = zlib.decompress(data[41:41 + length])
        self.assertEqual(len(raw), 8 * (1 + 8 * 3))
        try:
            from PIL import Image
        except ImportError:
            return
        image = Image.open(io.BytesIO(data))
        self.assertEqual(image.size, (8, 8))
        self.assertEqual(image.getpixel((0, 0)), (10, 20, 30))

    def test_main_writes_report_file_without_a_git_repo(self):
        output = Path(tempfile.mkdtemp()) / "report.txt"
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = doctor.main(["--root", str(self.root), "--home", str(Path(tempfile.mkdtemp())), "--offline", "--output", str(output)])
        self.assertIn("RAPOR SONU", printed.getvalue())
        self.assertEqual(code, 0)
        text = output.read_text(encoding="utf-8")
        self.assertIn("RETAKE TANI RAPORU", text)
        self.assertIn("[Depo]", text)


if __name__ == "__main__":
    unittest.main()
