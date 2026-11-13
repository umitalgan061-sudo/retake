import http.client
import json
import tempfile
import threading
import time
import unittest
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

    def test_requests_addressed_to_another_host_are_refused(self):
        response, _ = self.request("GET", "/health", host="evil.example")
        self.assertEqual(response.status, 403)

    def test_swap_runs_as_a_job_and_reports_stages(self):
        seen = []

        def fake_swap(facefusion_dir, pixel_boost, enhancer, parsed, progress):
            for step in range(len(srv.JOB_STAGES)):
                progress(step)
                seen.append(step)
            return PNG, {"X-Retake-Variation": "0"}

        body, content_type = multipart_body()
        with patch.object(srv, "process_swap", fake_swap):
            response, payload = self.request("POST", "/api/face-swap", body, {"Content-Type": content_type})
            self.assertEqual(response.status, 202)
            accepted = json.loads(payload)
            job = self.wait_for_job(accepted["job_id"])
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["step"], job["total"])
        self.assertEqual(seen, list(range(len(srv.JOB_STAGES))))
        response, payload = self.request("GET", f"/api/jobs/{accepted['job_id']}/result")
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "image/png")
        self.assertEqual(response.getheader("X-Retake-Variation"), "0")
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
        jobs = [srv.create_job() for _ in range(srv.MAX_KEPT_JOBS + 2)]
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
