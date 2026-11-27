import http.client
import json
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from tools import facefusion_local_server as srv

ORIGIN = "https://retake06-ua.web.app"
TOKEN = "pairing-code-for-tests-0123"
PNG = srv.PNG_SIGNATURE + b"fake"


def multipart_body():
    boundary = "retakeboundary"
    parts = []

    def add(name, payload, filename=None, content_type=None):
        head = f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
        if filename:
            head += f'; filename="{filename}"'
        head += "\r\n"
        if content_type:
            head += f"Content-Type: {content_type}\r\n"
        parts.append(head.encode() + b"\r\n" + payload + b"\r\n")

    add("target", PNG, "t.png", "image/png")
    add("appearance_profile", PNG, "a.png", "image/png")
    for index in range(srv.MAX_REFERENCES):
        add("reference", PNG, f"r{index}.png", "image/png")
    add("metadata", json.dumps({
        "referenceWeights": [1] * srv.MAX_REFERENCES,
        "faceBounds": {"x": 10, "y": 10, "width": 100, "height": 100}
    }).encode())
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


class EngineServerTest(unittest.TestCase):
    pairing_token = ""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        facefusion_dir = Path(self.temp.name)
        (facefusion_dir / "facefusion.py").write_text("")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), srv.RetakeFaceFusionHandler)
        self.server.allowed_origins = {ORIGIN}
        self.server.facefusion_dir = facefusion_dir
        self.server.face_swapper_pixel_boost = "1024x1024"
        self.server.face_enhancer_enabled = False
        self.server.pairing_token = self.pairing_token
        self.port = self.server.server_port
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        srv.JOBS.clear()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.temp.cleanup()
        deadline = time.time() + 5
        while srv.INFERENCE_LOCK.locked() and time.time() < deadline:
            time.sleep(0.02)

    def request(self, method, path, body=None, headers=None, host=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        all_headers = {"Origin": ORIGIN, **(headers or {})}
        if host:
            all_headers["Host"] = host
        connection.request(method, path, body=body, headers=all_headers)
        response = connection.getresponse()
        payload = response.read()
        connection.close()
        return response, payload

    def assert_engine_freed(self):
        # The handler answers first and releases the lock in its finally block.
        deadline = time.time() + 2
        while srv.INFERENCE_LOCK.locked() and time.time() < deadline:
            time.sleep(0.01)
        self.assertFalse(srv.INFERENCE_LOCK.locked())

    def wait_for_job(self, job_id, headers=None):
        for _ in range(200):
            response, payload = self.request("GET", f"/api/jobs/{job_id}", headers=headers)
            job = json.loads(payload)
            if job["status"] in {"done", "error"}:
                return job
            time.sleep(0.02)
        self.fail("job did not finish")


class LocalEngineTest(EngineServerTest):
    def test_health_reports_async_jobs_without_pairing(self):
        response, payload = self.request("GET", "/health")
        health = json.loads(payload)
        self.assertEqual(response.status, 200)
        self.assertTrue(health["async_jobs"])
        self.assertFalse(health["paired"])
        self.assertTrue(health["ultra_mode"])
        self.assertTrue(health["lossless_preblended_transport"])
        self.assertEqual(health["ultra_rounds"], 24)
        self.assertEqual(health["ultra_min_seconds"], 420)
        self.assertEqual(health["ultra_search_budget_seconds"], 420)
        # v23: depths are advertised; old clients still get the seven-minute default.
        self.assertTrue(health["quality_depths"])
        self.assertEqual(health["ultra_default_depth"], "standard")
        self.assertEqual(set(health["ultra_depths"]), {"standard", "deep", "max"})
        self.assertGreater(health["ultra_depths"]["deep"]["search_seconds"], 420)

    def test_calibration_summary_is_numbers_only_and_health_advertises_learning(self):
        from tools.ultra_calibration import CalibrationStore
        self.server.calibration = CalibrationStore(Path(self.temp.name) / "cal.json", signature="s")
        self.server.calibration.record("bank", [(3, {"X-Retake-Self-Fidelity": "ssim:0.9000,deltaE:2.000"})], self_test=True)
        response, payload = self.request("GET", "/api/calibration")
        summary = json.loads(payload)
        self.assertEqual(response.status, 200)
        self.assertEqual(summary["banks"]["bank"]["variants"]["3"]["fidelity"], {"n": 1, "mean": 0.86})
        _response, payload = self.request("GET", "/health")
        health = json.loads(payload)
        self.assertTrue(health["ultra_learning"])
        self.assertTrue(health["self_test_benchmark"])
        response, _ = self.request("GET", "/api/calibration", host="evil.example")
        self.assertEqual(response.status, 403)

    def test_requests_addressed_to_another_host_are_refused(self):
        response, _ = self.request("GET", "/health", host="evil.example")
        self.assertEqual(response.status, 403)

    def test_swap_runs_as_a_job_and_reports_stages(self):
        seen = []

        def fake_swap(facefusion_dir, pixel_boost, enhancer, parsed, progress, caches=None):
            for step in range(len(srv.JOB_STAGES)):
                progress(step)
                seen.append(step)
            return PNG, {"X-Retake-Variation": "0", "X-Retake-Similarity": "0.80"}

        body, content_type = multipart_body()
        with patch.object(srv, "process_swap", fake_swap), patch.object(srv, "ULTRA_MIN_SECONDS", 0):
            response, payload = self.request("POST", "/api/face-swap", body, {"Content-Type": content_type})
            self.assertEqual(response.status, 202)
            accepted = json.loads(payload)
            job = self.wait_for_job(accepted["job_id"])
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["step"], job["total"])
        self.assertEqual(seen, list(range(len(srv.JOB_STAGES))) * len(srv.ULTRA_VARIATIONS))
        self.assertTrue(job["ultra"])
        self.assertEqual(job["total"], len(srv.JOB_STAGES) * len(srv.ULTRA_VARIATIONS))
        response, payload = self.request("GET", f"/api/jobs/{accepted['job_id']}/result")
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "image/png")
        self.assertEqual(response.getheader("X-Retake-Variation"), "0")
        self.assertEqual(response.getheader("X-Retake-Ultra"), "on")
        self.assertEqual(response.getheader("X-Retake-Ultra-Rounds"), "24")
        self.assertEqual(payload, PNG)

    def test_failed_swap_is_reported_through_the_job_and_frees_the_engine(self):
        def failing_swap(*args):
            raise srv.NoOpSwapError("yüz değişmedi")

        body, content_type = multipart_body()
        with patch.object(srv, "process_swap", failing_swap):
            _, payload = self.request("POST", "/api/face-swap", body, {"Content-Type": content_type})
            job = self.wait_for_job(json.loads(payload)["job_id"])
        self.assertEqual(job["status"], "error")
        self.assertEqual(job["http_status"], 422)
        self.assertIn("yüz değişmedi", job["error"])
        response, _ = self.request("GET", f"/api/jobs/{json.loads(payload)['job_id']}/result")
        self.assertEqual(response.status, 409)
        self.assert_engine_freed()

    def test_invalid_upload_is_rejected_immediately_and_frees_the_engine(self):
        response, payload = self.request(
            "POST", "/api/face-swap", b"--x--\r\n", {"Content-Type": "multipart/form-data; boundary=x"}
        )
        self.assertEqual(response.status, 400)
        self.assert_engine_freed()

    def test_unknown_job_is_not_found(self):
        response, _ = self.request("GET", "/api/jobs/does-not-exist")
        self.assertEqual(response.status, 404)

    def test_only_recent_jobs_are_kept(self):
        # Only completed jobs are eviction candidates. The actual inference
        # lock prevents multiple pending jobs in production.
        jobs = []
        for index in range(srv.MAX_KEPT_JOBS + 2):
            job = srv.create_job()
            job.update(status="done", finished=job["created"])
            jobs.append(job)
        srv.prune_jobs()
        self.assertEqual(len(srv.JOBS), srv.MAX_KEPT_JOBS)
        self.assertNotIn(jobs[0]["id"], srv.JOBS)
        self.assertIn(jobs[-1]["id"], srv.JOBS)


class PairedEngineTest(EngineServerTest):
    pairing_token = TOKEN

    def test_every_request_needs_the_pairing_code(self):
        for headers in ({}, {srv.PAIRING_HEADER: "wrong-code-wrong-code"}):
            response, payload = self.request("GET", "/health", headers=headers)
            self.assertEqual(response.status, 401)
            self.assertTrue(json.loads(payload)["pairing_required"])
        body, content_type = multipart_body()
        response, _ = self.request("POST", "/api/face-swap", body, {"Content-Type": content_type})
        self.assertEqual(response.status, 401)
        self.assert_engine_freed()
        response, _ = self.request("GET", "/api/jobs/anything")
        self.assertEqual(response.status, 401)

    def test_paired_client_can_reach_the_engine_through_a_tunnel_host(self):
        response, payload = self.request(
            "GET", "/health", headers={srv.PAIRING_HEADER: TOKEN}, host="calm-river.trycloudflare.com"
        )
        self.assertEqual(response.status, 200)
        self.assertTrue(json.loads(payload)["paired"])

    def test_other_websites_are_refused_even_with_the_code(self):
        body, content_type = multipart_body()
        response, _ = self.request("POST", "/api/face-swap", body, {
            "Content-Type": content_type, srv.PAIRING_HEADER: TOKEN, "Origin": "https://evil.example"
        })
        self.assertEqual(response.status, 403)

    def test_preflight_allows_the_pairing_header(self):
        response, _ = self.request("OPTIONS", "/api/face-swap", headers={
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-retake-token"
        })
        self.assertEqual(response.status, 204)
        self.assertIn(srv.PAIRING_HEADER, response.getheader("Access-Control-Allow-Headers"))
        self.assertEqual(response.getheader("Access-Control-Allow-Origin"), ORIGIN)


class PoseMetadataValidationTests(unittest.TestCase):
    def test_pose_distances_are_typed_and_optional(self):
        metadata = {
            "referenceWeights": [1] * 25,
            "faceBounds": {"x": 12, "y": 10, "width": 140, "height": 160},
            "poseDistances": [0.04] * 20 + [None] * 5
        }
        checked = srv._validate_metadata(metadata, 25)
        self.assertEqual(checked["poseDistances"][0], 0.04)
        self.assertIsNone(checked["poseDistances"][24])
        self.assertIsNone(srv._validate_metadata({
            "referenceWeights": [1] * 25,
            "faceBounds": metadata["faceBounds"]
        }, 25)["poseDistances"])
        for bad in ([0.1] * 24, [float("nan")] * 25, [True] * 25, [-1] * 25):
            with self.assertRaisesRegex(ValueError, "poz mesafeleri"):
                srv._validate_metadata({**metadata, "poseDistances": bad}, 25)


class AllowedOriginsTest(unittest.TestCase):
    def test_new_and_old_sites_are_allowed_by_default(self):
        source = Path(srv.__file__).read_text(encoding="utf-8")
        self.assertIn('"https://retake06-ua.web.app"', source)
        self.assertIn('"https://retake06.web.app"', source)


class PairingTokenFileTest(unittest.TestCase):
    def test_token_is_created_once_and_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "pairing-token"
            first = srv.load_pairing_token(path)
            self.assertGreaterEqual(len(first), 20)
            self.assertEqual(srv.load_pairing_token(path), first)
            path.write_text("short")
            self.assertNotEqual(srv.load_pairing_token(path), "short")


if __name__ == "__main__":
    unittest.main()


try:
    import cv2
    import numpy as np
except ImportError:  # OpenCV/NumPy live in the FaceFusion environment.
    cv2 = None


@unittest.skipIf(cv2 is None, "OpenCV/NumPy yüklü değil")
class CachedPipelineTest(unittest.TestCase):
    """Runs the real process_swap with FaceFusion and ArcFace replaced by fakes."""

    def setUp(self):
        from tools.retake_cache import IdentityReportCache, PassCache
        # These fakes paint flat shapes; the naturalness guard is tested on its
        # own (CompositeChoiceTest), here the identity logic is.
        guard = patch.object(srv, "NATURALNESS_TOLERANCE", 1e9)
        guard.start()
        self.addCleanup(guard.stop)
        from tools import retake_compositing
        grain = patch.object(retake_compositing, "match_skin_grain", lambda composite, *args, **kwargs: composite)
        grain.start()
        self.addCleanup(grain.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.caches = {"identity": IdentityReportCache(Path(self.temp.name) / "identity"), "passes": PassCache()}
        self.arcface_calls = 0
        self.facefusion_calls = []
        self.transplant_modes = []
        rng = np.random.default_rng(5)
        target = np.full((512, 391, 3), (120, 150, 205), np.uint8)
        target = np.clip(target + rng.normal(0, 3, target.shape), 0, 255).astype(np.uint8)
        self.target_png = cv2.imencode(".png", target)[1].tobytes()
        self.reference = cv2.imencode(".png", target[:100, :100])[1].tobytes()

    def tearDown(self):
        self.temp.cleanup()

    def fake_arcface(self, facefusion_dir, reference_paths):
        self.arcface_calls += 1
        report = {"ok": True, "median_similarity": 0.8, "outlier_indices": [], "outlier_files": [], "centroid": [1.0, 0.0]}
        if getattr(self, "enable_quality_candidates", False):
            report["reference_median_similarities"] = [0.55 + index * 0.01 for index in range(srv.MAX_REFERENCES)]
        return report

    render_scores: list = []

    reinforce_scores = (0.70, 0.60)
    fused_score = 0.77

    blend_scores = None
    candidate_scores = None

    def fake_score(self, facefusion_dir, centroid, probes):
        names = [path.name for path in probes]
        if names == ["retake-verified-realism.png"]:
            # The merged realism candidate is measured independently.
            # In this fixture its visible centre is the render's shade.
            image=cv2.imread(str(probes[0]),cv2.IMREAD_UNCHANGED)
            self.assertEqual(image.shape[2],3,"verifier receives opaque pixels")
            shade=int(image[250,195,0])
            if self.candidate_scores and names[0] in self.candidate_scores:
                return [self.candidate_scores[names[0]]]
            return [self.render_scores[{40:0,60:1}[shade]]]
        if names == ["retake-rim-clean.png"]:
            # The optional post-render cleanup uses a second score probe.
            # Keep the preceding full-rescue probe available to existing tests.
            return [self.candidate_scores.get(names[0], 0.74) if self.candidate_scores else 0.74]
        if names == ["retake-identity-reinforced-detail.png"]:
            return [self.fused_score]
        if "retake-identity-reinforced.png" in names:
            self.reinforce_probes = names
            return list(self.reinforce_scores)
        if self.blend_scores is not None and "retake-hybrid-1.png" in names:
            self.blend_probes = names
            return [self.blend_scores[int(name.split("-")[-1][:-4]) if name[-5].isdigit() else 0] for name in names]
        self.scored = (centroid, names)
        values = []
        for name in names:
            if self.candidate_scores and name in self.candidate_scores:
                values.append(self.candidate_scores[name])
            elif name == "target.png":
                values.append(0.21)
            elif name.startswith("realism-"):
                values.append(self.render_scores[int(name.removeprefix("realism-").removesuffix(".png")) - 1])
            elif name == "retake-pre-geometry.png" or name == "retake-identity-only.png":
                values.append(0.73)
            else:
                values.append(0.74)
        return values

    def fake_realism(self, realism, final_output, target_path, bounds, centroid, temp_dir, landmarks=None):
        """Two 'renders': the result with a visibly re-rendered face."""
        self.realism_calls = getattr(self, "realism_calls", 0) + 1
        image = cv2.imread(str(final_output), cv2.IMREAD_COLOR)
        target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape, target.shape, "realism receives native-size target, not padded inference copy")
        paths = []
        for index, shade in enumerate((40, 60), start=1):
            render = image.copy()
            cv2.circle(render, (195, 250), 30, (shade, shade, shade), -1)
            path = temp_dir / f"realism-{index}.png"
            cv2.imwrite(str(path), render)
            paths.append(path)
        return paths

    def fake_facefusion(self, facefusion_dir, target_path, sources, output, model, boost, weight):
        self.facefusion_calls.append((model, weight))
        self.source_sizes = getattr(self, "source_sizes", [])
        self.source_sizes.append((model, len(sources)))
        image = cv2.imread(str(target_path))
        cv2.ellipse(image, (195, 260), (85, 110), 0, 0, 360, (90, 110, 170), -1)
        cv2.imwrite(str(output), image)
        return boost

    def fake_transplant(self, result, target, real_face, face_width):
        self.transplant_modes.append(real_face.get("mode"))
        painted = result.copy()
        cv2.circle(painted, (195, 230), 20, (10, 200, 10), -1)
        return painted, True

    def run_swap(self, variation, stages, jaw_warp=None, realism=None, real_face=False, detail_gate=None, extra_metadata=None):
        # These orchestration tests deliberately paint flat synthetic ellipses
        # instead of real faces. Stub photographic detail independently so the
        # original identity/geometry invariants remain under test. The detail
        # gate itself has pixel-based tests in test_ultra_quality.py.
        gate = detail_gate or (lambda *_a: (True, "passed", .9, {"eyes": .9, "mouth": .9}))
        metadata = {
            "jawWarp": jaw_warp,
            "referenceWeights": [1] * srv.MAX_REFERENCES,
            "faceBounds": {"x": 96.0, "y": 116.0, "width": 200.0, "height": 270.0},
            "variation": variation,
            "targetSize": {"width": 391, "height": 512},
            "headFit": {"scaleX": 0.97, "scaleY": 1.04},
            **(extra_metadata or {})
        }
        if real_face:
            metadata["realFace"] = {"mode": real_face if real_face in ("lower", "full", "both") else "full", "targetLandmarks": [[0.0, 0.0]] * 478, "references": [{"landmarks": [[0.0, 0.0]] * 478}]}
            metadata["realReferenceBytes"] = [b"photo"]
        parsed = (self.target_png, [self.reference] * srv.MAX_REFERENCES, ".png", [".png"] * srv.MAX_REFERENCES,
                  self.reference, ".png", metadata)
        with patch.object(srv, "verify_reference_identity", self.fake_arcface), \
                patch.object(srv, "run_with_memory_fallback", self.fake_facefusion), \
                patch.object(srv, "score_identity", self.fake_score), \
                patch.object(srv, "run_realism", self.fake_realism), \
                patch.object(srv, "apply_real_face", self.fake_transplant), \
                patch.object(srv, "check_detail_gate", gate):
            result, headers = srv.process_swap(
                Path(self.temp.name), "1024x1024", False, parsed,
                lambda step, cached=False: stages.append((step, cached)), {**self.caches, "realism": realism}
            )
        self.last_result = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(headers["X-Retake-Alpha-Transport"], "binary")
        self.assertTrue(set(np.unique(self.last_result[:, :, 3])) <= {0, 255})
        original = cv2.imdecode(np.frombuffer(self.target_png, np.uint8), cv2.IMREAD_COLOR)
        outside = self.last_result[:, :, 3] == 0
        np.testing.assert_array_equal(self.last_result[outside, :3], original[outside])
        self.assertEqual(cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_UNCHANGED).shape[:2], (512, 391))
        return headers

    def test_blurry_high_identity_candidate_is_ineligible(self):
        self.candidate_scores = {"retake-pre-geometry.png": .99}
        def gate(original, candidate, bounds):
            if candidate.name == "retake-pre-geometry.png":
                return False, "eyes-blur", .5, {"eyes": .4}
            return True, "passed", .9, {"eyes": .9}
        headers = self.run_swap(0, [], detail_gate=gate)
        self.assertNotEqual(headers["X-Retake-Identity-Selection"], "no-head-warp")
        self.assertLess(float(headers["X-Retake-Similarity"]), .99)
        self.assertEqual(headers["X-Retake-Detail-Gate"], "passed")

    def test_when_no_candidate_passes_the_detail_gate_the_best_identity_is_labelled_not_lost(self):
        headers = self.run_swap(0, [], detail_gate=lambda *_a: (False, "eyes-blur", .5, {"eyes": .4}))
        self.assertIsNotNone(srv.score_from_headers(headers))
        self.assertEqual(headers["X-Retake-Detail-Gate"], "failed:eyes-blur")
        self.assertEqual(headers["X-Retake-Photo-Balance"], "off")

    def test_regenerating_reuses_arcface_and_both_gpu_passes(self):
        first, second = [], []
        self.run_swap(0, first)
        self.assertEqual(self.arcface_calls, 1)
        self.assertEqual([model for model, _ in self.facefusion_calls], ["hyperswap_1b_256", "inswapper_128", "uniface_256"])
        # Stage 5 (photographic realism) is not installed here and is skipped.
        self.assertEqual(first, [(0, False), (1, False), (2, False), (3, False), (4, False), (6, False)])

        self.run_swap(2, second)  # same identity model and swapper weight, different head fit
        self.assertEqual(self.arcface_calls, 1, "ArcFace report reused")
        self.assertEqual(len(self.facefusion_calls), 3, "all FaceFusion passes reused")
        self.assertEqual(second, [(0, True), (1, True), (2, True), (3, True), (4, False), (6, False)])

    def test_identity_reinforcement_is_kept_only_when_it_is_closer_to_the_person(self):
        self.reinforce_scores = (0.70, 0.78)
        headers = self.run_swap(0, [])
        self.assertEqual(headers["X-Retake-Identity-Reinforced"], "on")
        self.assertEqual(self.reinforce_probes, ["retake-identity-alpha.png", "retake-identity-reinforced.png"])
        self.reinforce_scores = (0.70, 0.66)
        self.assertEqual(self.run_swap(2, [])["X-Retake-Identity-Reinforced"], "off")

    def test_a_blurring_reinforcement_keeps_its_identity_through_detail_fusion(self):
        self.reinforce_scores = (0.70, 0.78)

        def guard(single, candidate, bounds):
            if candidate.name == "retake-identity-reinforced.png":
                return False, "eyes-blur", {"eyes": 0.80}
            return True, "passed", {"eyes": 0.95}

        with patch.object(srv, "check_reinforcement_detail", guard):
            headers = self.run_swap(0, [])
            self.assertEqual(headers["X-Retake-Identity-Reinforced"], "on")
            self.assertEqual(headers["X-Retake-Reinforce-Guard"], "detail-fused")
            self.fused_score = 0.703          # no real identity gain: not used
            headers = self.run_swap(2, [])
            self.assertEqual(headers["X-Retake-Identity-Reinforced"], "off")

    def test_softer_variant_is_not_reinforced(self):
        headers = self.run_swap(3, [])
        self.assertEqual(headers["X-Retake-Identity-Reinforced"], "off")
        self.assertNotIn("inswapper_128", [model for model, _ in self.facefusion_calls])

    def test_jaw_warp_runs_in_the_pipeline_and_respects_the_variant(self):
        jaw = {"controls": [{"from": [195.0, 370.0], "to": [195.0, 388.0]}], "anchors": [[195.0, 200.0]]}
        moved = self.run_swap(0, [], jaw)
        kept = self.run_swap(1, [], jaw)  # "photo's own proportions": no reshape
        self.assertEqual(moved["X-Retake-Variation"], "0")
        self.assertEqual(moved["X-Retake-Similarity"], "0.7400")
        self.assertEqual(moved["X-Retake-Similarity-Before"], "0.2100")
        self.assertEqual(self.scored[0], [1.0, 0.0], "scored against the cached centroid")
        self.assertEqual(kept["X-Retake-Variation"], "1")

    def test_realism_render_is_used_when_it_keeps_the_identity(self):
        stages = []
        self.render_scores = [0.70, 0.79]
        headers = self.run_swap(0, stages, realism={"python": Path("py"), "models": Path("m")})
        self.assertEqual([step for step, _ in stages][-2:], [srv.REALISM_STAGE, srv.SCORING_STAGE])
        self.assertEqual(headers["X-Retake-Realism"], "on")
        self.assertEqual(headers["X-Retake-Similarity"], "0.7900", "the closer render is reported")
        self.assertEqual(self.scored[1][3:5], ["realism-1.png", "realism-2.png"])
        # Its re-rendered face is in the result, and covered by the matte.
        self.assertTrue(np.all(self.last_result[250, 195, :3] == 60))
        self.assertEqual(int(self.last_result[250, 195, 3]), 255)

    def test_refined_settings_reuse_cached_passes_and_are_reported(self):
        first = self.run_swap(0, [])
        self.assertEqual(first["X-Retake-Variation-Refined"], "off")
        passes = len(self.facefusion_calls)
        headers = self.run_swap(0, [], extra_metadata={"variationParams": (1.0, 0.26, 1.0)})
        self.assertEqual(len(self.facefusion_calls), passes, "same swapper weight: no new FaceFusion pass")
        self.assertEqual(headers["X-Retake-Variation-Params"], "1.000,0.260,1.000")
        self.assertEqual(headers["X-Retake-Variation-Refined"], "on")

    def fine_refiner(self, report):
        def refine(original, composite, output, bounds, landmarks):
            self.fine_calls = getattr(self, "fine_calls", 0) + 1
            image = cv2.imread(str(composite), cv2.IMREAD_UNCHANGED)
            image[250, 195, :3] = (1, 2, 3)  # a visible, verifiable change
            cv2.imwrite(str(output), image)
            return report
        return refine

    def test_fine_feature_refinement_is_kept_only_when_arcface_verifies_it(self):
        applied = {"status": "applied", "quality_before": 0.70, "quality_after": 0.82,
                   "catchlights": {"eye_r": "restored", "eye_l": "present"}, "catchlights_restored": 1,
                   "features": {"eye_r": {"before": 0.61, "after": 0.97, "action": "restore", "shift": [0, 0]}}}
        mesh = {"targetLandmarks": [[195.0, 250.0]] * 478}
        self.candidate_scores = {"retake-fine-features.png": 0.80}
        with patch.object(srv, "refine_fine_features_file", self.fine_refiner(applied)):
            headers = self.run_swap(0, [], extra_metadata=mesh)
        self.assertEqual(headers["X-Retake-Fine-Features"], "applied")
        self.assertEqual(headers["X-Retake-Fine-Detail"], "eye_r:0.610>0.970")
        self.assertEqual(headers["X-Retake-Catchlights"], "1/2")
        self.assertEqual(headers["X-Retake-Similarity"], "0.8000")
        self.assertEqual(tuple(int(v) for v in self.last_result[250, 195, :3]), (1, 2, 3))

        self.candidate_scores = {"retake-fine-features.png": 0.70}
        with patch.object(srv, "refine_fine_features_file", self.fine_refiner(applied)):
            headers = self.run_swap(0, [], extra_metadata=mesh)
        self.assertEqual(headers["X-Retake-Fine-Features"], "identity-loss")
        self.assertEqual(headers["X-Retake-Similarity"], "0.7400")
        self.assertNotEqual(tuple(int(v) for v in self.last_result[250, 195, :3]), (1, 2, 3))

    def test_fine_features_need_the_target_mesh(self):
        with patch.object(srv, "refine_fine_features_file", self.fine_refiner({"status": "applied"})):
            headers = self.run_swap(0, [])
        self.assertEqual(headers["X-Retake-Fine-Features"], "no-mesh")
        self.assertEqual(getattr(self, "fine_calls", 0), 0)

    def test_realism_merge_that_loses_identity_is_not_used(self):
        self.render_scores=[.70,.79]
        self.candidate_scores={"retake-verified-realism.png":.69}
        headers=self.run_swap(0,[],realism={"python":Path("py"),"models":Path("m")})
        self.assertEqual(headers["X-Retake-Realism"],"off")
        self.assertEqual(headers["X-Retake-Similarity"],"0.7400")
        self.assertFalse(np.all(self.last_result[250,195,:3]==60))

    def test_realism_and_fine_refinement_share_one_identity_budget(self):
        self.render_scores=[.70,.79]
        self.candidate_scores={"retake-verified-realism.png":.7385,"retake-fine-features.png":.7359}
        applied={"status":"applied","quality_before":.70,"quality_after":.82,
                 "catchlights_restored":1,"features":{"eye_r":{"before":.61,"after":.97,"action":"restore","shift":[0,0]}}}
        mesh={"targetLandmarks":[[195.0,250.0]]*478}
        with patch.object(srv,"refine_fine_features_file",self.fine_refiner(applied)), \
                patch.object(srv,"repair_dark_rim_file",return_value={"corrected_pixels":0}):
            headers=self.run_swap(0,[],realism={"python":Path("py"),"models":Path("m")},extra_metadata=mesh)
        self.assertEqual(headers["X-Retake-Realism"],"on")
        self.assertEqual(headers["X-Retake-Fine-Features"],"identity-loss")
        self.assertEqual(headers["X-Retake-Similarity"],"0.7385")
        self.assertEqual(headers["X-Retake-Finishing-Floor"],"0.7360")

    def test_realism_render_that_loses_the_person_is_not_used(self):
        self.render_scores = [0.61, 0.70]
        headers = self.run_swap(0, [], realism={"python": Path("py"), "models": Path("m")})
        self.assertEqual(headers["X-Retake-Realism"], "off")
        self.assertEqual(headers["X-Retake-Similarity"], "0.7400")
        self.assertFalse(np.all(self.last_result[250, 195, :3] == 60))

    def test_chooses_unwarped_face_when_head_fit_damages_identity(self):
        self.candidate_scores = {"retake-pre-geometry.png": 0.88}
        headers = self.run_swap(0, [])
        self.assertEqual(headers["X-Retake-Identity-Selection"], "no-head-warp")
        self.assertEqual(headers["X-Retake-Similarity"], "0.8800")
        self.assertEqual(headers["X-Retake-Compositing"], "preblended")

    def test_falls_back_to_identity_only_when_beautifying_loses_resemblance(self):
        self.candidate_scores = {"retake-identity-only.png": 0.89}
        headers = self.run_swap(0, [])
        self.assertEqual(headers["X-Retake-Identity-Selection"], "identity-only")
        self.assertEqual(headers["X-Retake-Similarity"], "0.8900")

    def test_refuses_to_claim_success_when_arcface_cannot_verify_any_result(self):
        with patch.object(self, "fake_score") as mocked:
            def fake_score(facefusion_dir, centroid, probes):
                if any(path.name == "retake-identity-reinforced.png" for path in probes):
                    return [None] * len(probes)
                if any(path.name.startswith("retake-hybrid") for path in probes):
                    return [None] * len(probes)
                return [None] * len(probes)
            mocked.side_effect = fake_score
            with self.assertRaisesRegex(srv.IdentityVerificationError, "doğrulanamadı"):
                self.run_swap(0, [])

    def test_refuses_output_if_identity_regresses_below_original(self):
        self.candidate_scores = {
            "retake-hybrid-profile.png": 0.35,
            "retake-pre-geometry.png": 0.36,
            "retake-identity-only.png": 0.38,
            # Cosmetic finishing must also simulate a weak measured identity;
            # the mock default (.74) would otherwise invent a successful rescue.
            "retake-upper-seam.png": 0.38,
            "target.png": 0.50,
        }
        with self.assertRaisesRegex(srv.IdentityConflictError, "düşük"):
            self.run_swap(0, [])

    def test_consensus_identity_uses_reference_agreement_when_it_improves_arcface(self):
        self.enable_quality_candidates = True
        self.candidate_scores = {
            "retake-identity-alpha.png": 0.73,
            "retake-identity-consensus.png": 0.85
        }
        headers = self.run_swap(0, [])
        self.assertEqual(headers["X-Retake-Identity-Source"], "consensus")
        self.assertIn(("hyperswap_1b_256", 25), self.source_sizes)
        self.assertIn(("hyperswap_1b_256", 12), self.source_sizes)
        self.assertEqual(headers["X-Retake-Compositing"], "preblended")

    def test_real_reference_texture_can_win_over_a_blurred_mean(self):
        self.enable_quality_candidates = True
        self.candidate_scores = {"retake-texture-0.png": 0.91, "retake-texture-1.png": 0.89}
        with patch.object(srv, "BORDER_STEP_FLOOR", 1e9):
            headers = self.run_swap(0, [])
        self.assertEqual(headers["X-Retake-Texture-Source"], "reference-25")
        self.assertIn(("uniface_256", 1), self.source_sizes)
        self.assertEqual(headers["X-Retake-Similarity"], "0.7400",
                         "post-processing score is evaluated separately from blend candidate score")

    def test_without_the_realism_environment_nothing_changes(self):
        headers = self.run_swap(0, [])
        self.assertEqual(headers["X-Retake-Realism"], "unavailable")
        self.assertEqual(getattr(self, "realism_calls", 0), 0)

    def test_photo_transplant_is_a_candidate_not_forced(self):
        # Order: tone 0.9 plain, + transplant, tone 0.6 plain, + transplant, ...
        self.blend_scores = [0.70, 0.60, 0.72, 0.61, 0.69, 0.62]
        # Model the same weak identity in the independently measured full-face
        # rescue candidates; native-size detail scoring now really runs here.
        # Otherwise the fake ArcFace assigns these candidates the default .74
        # (a tie), incorrectly simulating a successful identity-preserving rescue.
        self.candidate_scores = {"retake-identity-full-20.png": 0.60,
                                 "retake-identity-full-40.png": 0.61}
        guard = patch.object(srv, "BORDER_STEP_FLOOR", 1e9)   # the selection is what is tested here
        guard.start()
        self.addCleanup(guard.stop)
        headers = self.run_swap(0, [], real_face=True)
        self.assertEqual(len(self.blend_probes), 6)
        self.assertEqual(headers["X-Retake-Real-Face"], "off", "the transplant lost identity: not used")
        self.assertFalse(np.all(self.last_result[230, 195, :3] == (10, 200, 10)))
        self.blend_scores = [0.60, 0.80, 0.61, 0.70, 0.62, 0.71]
        headers = self.run_swap(1, [], real_face=True)
        self.assertEqual(headers["X-Retake-Real-Face"], "on")

    def test_lower_face_candidate_can_win_when_arcface_prefers_reference_beard(self):
        self.blend_scores = [0.70, 0.88, 0.69, 0.75, 0.68, 0.72]
        with patch.object(srv, "BORDER_STEP_FLOOR", 1e9):
            headers = self.run_swap(2, [], real_face="lower")
        self.assertEqual(headers["X-Retake-Lower-Face-Transfer"], "on")
        self.assertEqual(headers["X-Retake-Real-Face"], "off",
                         "lower-only mode must not pretend to have pasted the whole face")
        self.assertEqual(headers["X-Retake-Variation"], "2")

    def test_complete_and_lower_reference_face_compete_for_identity(self):
        # Three blends, each evaluating [model, lower reference, full photo].
        self.blend_scores = [.70, .81, .83, .68, .80, .81, .69, .79, .80]
        with patch.object(srv, "BORDER_STEP_FLOOR", 1e9):
            headers = self.run_swap(0, [], real_face="both")
        self.assertEqual(len(self.blend_probes), 9)
        self.assertEqual(headers["X-Retake-Real-Face"], "on")
        self.assertEqual(headers["X-Retake-Complete-Identity"], "on")
        self.assertEqual(headers["X-Retake-Lower-Face-Transfer"], "off")

    def test_lower_face_wins_when_full_reference_loses_identity(self):
        self.blend_scores = [.70, .85, .79, .68, .83, .78, .69, .82, .77]
        with patch.object(srv, "BORDER_STEP_FLOOR", 1e9):
            headers = self.run_swap(0, [], real_face="both")
        self.assertEqual(headers["X-Retake-Lower-Face-Transfer"], "on")
        self.assertEqual(headers["X-Retake-Complete-Identity"], "off")

    def test_real_reference_warp_is_cached_across_three_tone_variants(self):
        self.run_swap(0, [], real_face="both")
        self.assertEqual(self.transplant_modes.count("lower"), 5,
                         "once across tone variants plus four strength-adjusted identity rescues")
        self.assertEqual(self.transplant_modes.count("full"), 3,
                         "once across tone variants plus two low-strength identity rescues")

    def test_weak_complete_face_reference_can_win_when_identity_is_preserved(self):
        self.candidate_scores = {
            "retake-hybrid-profile.png": 0.806,
            "retake-pre-geometry.png": 0.806,
            "retake-identity-only.png": 0.833,
            "retake-identity-full-20.png": 0.832,
            "retake-identity-full-40.png": 0.829,
            "retake-identity-beard-35.png": 0.829,
            "retake-identity-beard-65.png": 0.828,
        }
        with patch.object(srv, "BORDER_STEP_FLOOR", 1e9):
            headers = self.run_swap(0, [], real_face="both")
        self.assertEqual(headers["X-Retake-Identity-Selection"], "identity-full")
        self.assertEqual(headers["X-Retake-Complete-Identity"], "on")
        self.assertEqual(headers["X-Retake-Real-Face"], "on")
        self.assertEqual(headers["X-Retake-Similarity"], "0.8320")

    def test_bad_full_photo_cannot_override_strong_identity_fallback(self):
        self.candidate_scores = {
            "retake-hybrid-profile.png": 0.806,
            "retake-pre-geometry.png": 0.806,
            "retake-identity-only.png": 0.833,
            "retake-identity-full-20.png": 0.80,
            "retake-identity-full-40.png": 0.78,
            "retake-identity-beard-35.png": 0.82,
            "retake-identity-beard-65.png": 0.81,
        }
        with patch.object(srv, "BORDER_STEP_FLOOR", 1e9):
            headers = self.run_swap(0, [], real_face="both")
        self.assertEqual(headers["X-Retake-Identity-Selection"], "identity-only")
        self.assertEqual(headers["X-Retake-Complete-Identity"], "off")
        self.assertEqual(headers["X-Retake-Similarity"], "0.8330")

    def test_identity_beard_can_survive_if_cosine_is_near_tied_with_identity_only(self):
        # The observed v19 failure: 0.833 pure-identity fallback always
        # removed the actual reference beard. A feathered reference beard
        # within .003 cosine is now allowed, with an additional seam gate.
        self.candidate_scores = {
            "retake-hybrid-profile.png": 0.806,
            "retake-pre-geometry.png": 0.806,
            "retake-identity-only.png": 0.833,
            "retake-identity-beard-35.png": 0.832,
            "retake-identity-beard-65.png": 0.831,
        }
        with patch.object(srv, "BORDER_STEP_FLOOR", 1e9):
            headers = self.run_swap(0, [], real_face="both")
        self.assertEqual(headers["X-Retake-Identity-Selection"], "identity-beard")
        self.assertEqual(headers["X-Retake-Lower-Face-Transfer"], "on")
        self.assertEqual(headers["X-Retake-Complete-Identity"], "off")
        self.assertEqual(headers["X-Retake-Similarity"], "0.8320")

    def test_identity_beard_rejected_when_cosine_loss_is_not_a_tie(self):
        self.candidate_scores = {
            "retake-hybrid-profile.png": 0.806,
            "retake-pre-geometry.png": 0.806,
            "retake-identity-only.png": 0.833,
            "retake-identity-beard-35.png": 0.826,
            "retake-identity-beard-65.png": 0.824,
        }
        with patch.object(srv, "BORDER_STEP_FLOOR", 1e9):
            headers = self.run_swap(0, [], real_face="both")
        self.assertEqual(headers["X-Retake-Identity-Selection"], "identity-only")
        self.assertEqual(headers["X-Retake-Lower-Face-Transfer"], "off")
        self.assertEqual(headers["X-Retake-Similarity"], "0.8330")

    def test_identity_only_jaw_candidate_is_scored_and_may_win_a_strict_tie(self):
        jaw = {"controls": [{"from": [195, 370], "to": [195, 381]}], "anchors": [[195, 200]]}
        self.candidate_scores = {
            "retake-hybrid-profile.png": 0.804,
            "retake-pre-geometry.png": 0.804,
            "retake-identity-only.png": 0.833,
            "retake-identity-jaw.png": 0.832,
        }
        headers = self.run_swap(2, [], jaw_warp=jaw)
        self.assertEqual(headers["X-Retake-Identity-Selection"], "identity-jaw")
        self.assertEqual(headers["X-Retake-Jaw-Focused"], "on")
        self.assertEqual(headers["X-Retake-Lower-Face-Transfer"], "off")

    def test_measured_jaw_variant_has_dedicated_header(self):
        jaw = {"controls": [{"from": [195, 370], "to": [195, 385]}], "anchors": [[195, 200]]}
        focused = self.run_swap(2, [], jaw_warp=jaw)
        ordinary = self.run_swap(0, [], jaw_warp=jaw)
        self.assertEqual(focused["X-Retake-Jaw-Focused"], "on")
        self.assertEqual(ordinary["X-Retake-Jaw-Focused"], "off")

    def test_exact_3d_jaw_plan_is_never_exaggerated(self):
        jaw = {"controls": [{"from": [195, 370], "to": [195, 385]}], "anchors": [[195, 200]]}
        scales = []
        real = srv.reshape_jawline

        def record(path, warp, width, scale):
            scales.append(round(scale, 4))
            return real(path, warp, width, scale)

        with patch.object(srv, "reshape_jawline", record):
            self.run_swap(2, [], jaw_warp=jaw)
            focused = list(scales)
            scales.clear()
            self.run_swap(2, [], jaw_warp={**jaw, "exact": True})
        self.assertTrue(any(scale > 1.0 for scale in focused), focused)
        self.assertTrue(scales and all(scale <= srv.JAW_EXACT_MAX_SCALE for scale in scales), scales)

    def test_jaw_exact_flag_is_validated(self):
        size, bounds = {"width": 391, "height": 512}, {"width": 200.0}
        jaw = {"controls": [{"from": [195, 370], "to": [195, 385]}], "anchors": [[195, 200]]}
        self.assertFalse(srv._validate_jaw_warp(jaw, size, bounds)["exact"])
        self.assertTrue(srv._validate_jaw_warp({**jaw, "exact": True}, size, bounds)["exact"])
        with self.assertRaises(ValueError):
            srv._validate_jaw_warp({**jaw, "exact": "yes"}, size, bounds)

    def test_softer_variant_reruns_the_gpu_passes_but_not_arcface(self):
        self.run_swap(0, [])
        self.run_swap(3, [])
        self.assertEqual(self.arcface_calls, 1)
        models = [model for model, _ in self.facefusion_calls]
        # Variant 0 uses HyperSwap 1b, the odd variant 3 AlphaFace (srv.identity_model_for).
        self.assertEqual((models.count("hyperswap_1b_256"), models.count("alphaface_256")), (1, 1))
        self.assertEqual(self.facefusion_calls[-1][1], 0.95)

    def test_ultra_variants_alternate_identity_models(self):
        self.assertEqual([srv.identity_model_for(v) for v in range(4)],
                         ["hyperswap_1b_256", "alphaface_256", "hyperswap_1b_256", "alphaface_256"])
        headers = self.run_swap(1, [])
        self.assertEqual(headers["X-Retake-Identity-Model"], "alphaface_256")

    def test_a_failed_hyperswap_pass_falls_back_to_alphaface(self):
        real = self.fake_facefusion

        def flaky(facefusion_dir, target, sources, output, model, *rest):
            if model == "hyperswap_1b_256":
                raise RuntimeError("model download failed")
            return real(facefusion_dir, target, sources, output, model, *rest)

        with patch.object(self, "fake_facefusion", flaky):
            headers = self.run_swap(0, [])
        self.assertEqual(headers["X-Retake-Identity-Model"], "alphaface_256")
        self.assertIsNotNone(srv.score_from_headers(headers))


class RemainingTimeTest(unittest.TestCase):
    def test_24_round_eta_stops_at_search_deadline_not_all_24_rounds(self):
        from tools.retake_cache import StageTimings
        with tempfile.TemporaryDirectory() as directory:
            timings = StageTimings(Path(directory) / "times.json")
            for name, seconds in zip(srv.JOB_STAGES, (40, 120, 30, 90, 5, 60, 8)):
                timings.record(name, seconds)
            job = srv.create_job(realism=True)
            job.update(
                status="running",
                step=19,
                created=time.time() - 260,
                stage_started=time.time() - 30
            )
            estimate = srv.estimate_remaining(job, timings)
            self.assertIsNotNone(estimate)
            self.assertGreaterEqual(estimate, 150)
            self.assertLess(estimate, 300, "do not project 24 complete model passes (73 minutes)")

    def test_wait_stage_has_no_gpu_eta_and_incomplete_history_does_not_crash(self):
        from tools.retake_cache import StageTimings
        with tempfile.TemporaryDirectory() as directory:
            empty = StageTimings(Path(directory) / "empty.json")
            job = srv.create_job(realism=True)
            job.update(status="running", created=time.time() - 270, stage_started=None)
            remaining = srv.estimate_remaining(job, empty)
            self.assertIsNotNone(remaining)
            self.assertAlmostEqual(remaining, 150, delta=2)
            job["stage_started"] = time.time()
            self.assertIsNotNone(srv.estimate_remaining(job, empty))

    def test_estimate_skips_cached_stages(self):
        from tools.retake_cache import StageTimings
        with tempfile.TemporaryDirectory() as directory:
            timings = StageTimings(Path(directory) / "t.json")
            for stage, seconds in zip(srv.JOB_STAGES, (40, 120, 30, 90, 5, 60, 8)):
                timings.record(stage, seconds)
            job = {"status": "running", "step": 0, "skipped": set(), "stage_started": time.time()}
            self.assertAlmostEqual(srv.estimate_remaining(job, timings), 353, delta=1)
            job.update(step=1, skipped={0}, stage_started=time.time() - 20)
            self.assertAlmostEqual(srv.estimate_remaining(job, timings), 293, delta=1)
            job.update(step=4, skipped={0, 1, 2, 3}, stage_started=time.time())
            self.assertAlmostEqual(srv.estimate_remaining(job, timings), 73, delta=1)
            # Without the realism environment its stage is not waited for.
            job.update(skipped={0, 1, 2, 3, srv.REALISM_STAGE})
            self.assertAlmostEqual(srv.estimate_remaining(job, timings), 13, delta=1)
            fresh = srv.create_job()
            self.assertIn(srv.REALISM_STAGE, fresh["skipped"])
            self.assertNotIn(srv.REALISM_STAGE, srv.create_job(realism=True)["skipped"])
            self.assertIsNone(srv.estimate_remaining({**job, "status": "done"}, timings))
            self.assertIsNone(srv.estimate_remaining(job, None))


class ScoreIdentityTest(unittest.TestCase):
    def probes(self, directory):
        paths = [Path(directory) / "result.png", Path(directory) / "target.png"]
        for path in paths:
            path.write_bytes(b"x")
        return paths

    def test_reads_the_worker_report(self):
        with tempfile.TemporaryDirectory() as directory:
            probes = self.probes(directory)
            done = SimpleNamespace(returncode=0, stderr="", stdout=
                "[ArcFace] ...\nRETAKE_IDENTITY_REPORT=" + json.dumps({"ok": True, "similarities": [0.71, None]}))
            with patch.object(srv.subprocess, "run", return_value=done) as run:
                self.assertEqual(srv.score_identity(Path(directory), [1.0, 0.0], probes), [0.71, None])
            command = run.call_args.args[0]
            self.assertIn("--centroid-file", command)
            self.assertEqual(command[-2:], [str(probes[0]), str(probes[1])])

    def test_failures_never_break_the_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            probes = self.probes(directory)
            self.assertEqual(srv.score_identity(Path(directory), None, probes), [None, None])
            for outcome in (
                SimpleNamespace(returncode=2, stderr="boom", stdout=""),
                SimpleNamespace(returncode=0, stderr="", stdout="RETAKE_IDENTITY_REPORT={not json"),
                SimpleNamespace(returncode=0, stderr="", stdout='RETAKE_IDENTITY_REPORT={"ok": true, "similarities": [0.5]}'),
            ):
                with patch.object(srv.subprocess, "run", return_value=outcome):
                    self.assertEqual(srv.score_identity(Path(directory), [1.0], probes), [None, None])
            with patch.object(srv.subprocess, "run", side_effect=srv.subprocess.TimeoutExpired("x", 1)):
                self.assertEqual(srv.score_identity(Path(directory), [1.0], probes), [None, None])


class RealismChoiceTest(unittest.TestCase):
    def test_choice_needs_a_verified_identity(self):
        renders = [Path("a.png"), Path("b.png")]
        self.assertEqual(srv.choose_realism_render(0.40, renders, [0.38, 0.51]), (Path("b.png"), 0.51))
        self.assertIsNone(srv.choose_realism_render(0.40, renders, [0.395, None]),
                          "v16 rejects even 0.005 cosine loss for optional realism")
        self.assertEqual(srv.choose_realism_render(0.40, renders, [0.398, None]), (Path("a.png"), 0.398))
        self.assertIsNone(srv.choose_realism_render(0.40, renders, [0.30, 0.38]))
        self.assertIsNone(srv.choose_realism_render(None, renders, [0.9, 0.9]), "unverifiable")
        self.assertIsNone(srv.choose_realism_render(0.40, renders, [None, None]))
        self.assertIsNone(srv.choose_realism_render(0.40, [], []))

    def test_realism_is_found_only_when_fully_installed(self):
        from tools.retake_realism_worker import model_paths
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "python.exe"
            python.write_text("")
            models = root / "models"
            models.mkdir()
            self.assertIsNone(srv.find_realism(python, models))
            self.assertIsNone(srv.find_realism(None, None))
            for path in model_paths(models).values():
                if path.suffix:
                    path.write_bytes(b"x")
                else:
                    path.mkdir()
            self.assertEqual(srv.find_realism(python, models), {"python": python, "models": models})
            self.assertIsNone(srv.find_realism(root / "missing.exe", models))

    def test_a_failing_worker_never_breaks_the_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            failed = SimpleNamespace(returncode=1, stdout="", stderr="CUDA out of memory")
            with patch.object(srv.subprocess, "run", return_value=failed):
                self.assertEqual(srv.run_realism(
                    {"python": Path("py"), "models": temp}, temp / "f.png", temp / "t.png",
                    {"x": 1, "y": 1, "width": 10, "height": 10}, [1.0, 0.0], temp
                ), [])
            with patch.object(srv.subprocess, "run", side_effect=OSError("no python")):
                self.assertEqual(srv.run_realism(
                    {"python": Path("py"), "models": temp}, temp / "f.png", temp / "t.png",
                    {"x": 1, "y": 1, "width": 10, "height": 10}, [1.0, 0.0], temp
                ), [])
            render = temp / "realism-1.png"
            render.write_bytes(b"png")
            report = 'RETAKE_REALISM_REPORT={"ok": true, "device": "cuda", "outputs": [{"path": "%s"}]}' % (
                str(render).replace("\\", "\\\\"))
            done = SimpleNamespace(returncode=0, stdout="loading\n" + report + "\n", stderr="")
            with patch.object(srv.subprocess, "run", return_value=done):
                self.assertEqual(srv.run_realism(
                    {"python": Path("py"), "models": temp}, temp / "f.png", temp / "t.png",
                    {"x": 1, "y": 1, "width": 10, "height": 10}, [1.0, 0.0], temp
                ), [render])


class CompositeChoiceTest(unittest.TestCase):
    blends = [(Path("seamless.png"), 0.30), (Path("middle.png"), 0.45), (Path("faithful.png"), 0.90)]

    def test_closest_to_the_person_wins_among_invisible_borders(self):
        calls = []

        def fake_score(facefusion_dir, centroid, probes):
            calls.append([path.name for path in probes])
            return [0.74, 0.78][:len(probes)]

        with patch.object(srv, "score_identity", fake_score):
            chosen = srv.choose_composite(Path("ff"), [1.0, 0.0], self.blends)
        self.assertEqual(chosen, Path("middle.png"))
        self.assertEqual(calls, [["seamless.png", "middle.png"]], "a visible border is never scored")

    def test_real_lower_face_wins_only_within_tiny_arcface_tie(self):
        labels = ["normal", "ton 0.6 + referans alt yüz", "ton 0.35 + doku nakli"]
        with patch.object(srv, "score_identity", return_value=[0.84, 0.8389]):
            best = srv.choose_composite(Path("ff"), [1.0], self.blends, labels)
        self.assertEqual(best, Path("middle.png"),
                         "real beard detail may win only when ArcFace is effectively equal")
        with patch.object(srv, "score_identity", return_value=[0.84, 0.82]):
            best = srv.choose_composite(Path("ff"), [1.0], self.blends, labels)
        self.assertEqual(best, Path("seamless.png"),
                         "do not sacrifice meaningful global identity for any beard texture")
        with patch.object(srv, "score_identity", return_value=[0.84, 0.839]):
            best = srv.choose_composite(
                Path("ff"), [1.0], self.blends,
                ["normal", "ton 0.6 + doku nakli", "normal"]
            )
        self.assertEqual(best, Path("seamless.png"),
                         "full-face transplant is not rewarded just for having a reference image")

    def test_without_a_measurement_the_most_seamless_blend_is_kept(self):
        with patch.object(srv, "score_identity", lambda *args: [None, None]):
            self.assertEqual(srv.choose_composite(Path("ff"), [1.0], self.blends), Path("seamless.png"))
        self.assertEqual(srv.choose_composite(Path("ff"), None, self.blends), Path("seamless.png"))
        self.assertEqual(srv.choose_composite(Path("ff"), [1.0], self.blends[:1]), Path("seamless.png"))

    def test_a_blend_that_looks_pasted_cannot_win_on_identity(self):
        calls = []

        def fake_score(facefusion_dir, centroid, probes):
            calls.append([path.name for path in probes])
            return {"natural.png": 0.74, "pasted.png": 0.80, "close.png": 0.76}.__getitem__ and [
                {"natural.png": 0.74, "pasted.png": 0.80, "close.png": 0.76}[path.name] for path in probes]

        blends = [(Path("natural.png"), 0.3, 90.0), (Path("pasted.png"), 0.3, 70.0), (Path("close.png"), 0.3, 85.0)]
        with patch.object(srv, "score_identity", fake_score):
            chosen = srv.choose_composite(Path("ff"), [1.0, 0.0], blends)
        self.assertEqual(chosen, Path("close.png"), "pasted one is 20 points less natural: never scored")
        self.assertEqual(calls, [["natural.png", "close.png"]])
        # Without naturalness measurements the old identity choice stands.
        with patch.object(srv, "score_identity", fake_score):
            self.assertEqual(srv.choose_composite(Path("ff"), [1.0], [(p, b) for p, b, _ in blends]), Path("pasted.png"))

    def test_engine_and_compositor_agree_on_the_default_shading(self):
        from tools import retake_compositing as rc
        self.assertEqual(srv.DEFAULT_SHADING_STRENGTH, rc.SHADING_STRENGTH)
        self.assertEqual(srv.COMPOSITE_VARIANTS[0][0], rc.HARMONIZE_STRENGTH)


class HeadFittedLandmarksTest(unittest.TestCase):
    def test_mesh_follows_the_head_fit(self):
        bounds = {"x": 100.0, "y": 100.0, "width": 200.0, "height": 200.0}
        points = [[200.0, 200.0], [300.0, 100.0]]
        self.assertEqual(srv.head_fitted_landmarks(points, bounds, 1.0, 1.0), points)
        fitted = srv.head_fitted_landmarks(points, bounds, 0.9, 1.1)
        for got, want in zip(sum(fitted, []), [200.0, 200.0, 290.0, 90.0]):
            self.assertAlmostEqual(got, want)
        self.assertIsNone(srv.head_fitted_landmarks(None, bounds, 0.9, 1.1))
