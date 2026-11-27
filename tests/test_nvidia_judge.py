"""NVIDIA visual jury (via Hafize) — configuration, data minimisation and Ultra safety.

No network: the HTTP layer uses a recording fake opener. Images are synthetic.
"""
from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from tools import facefusion_local_server as srv
from tools import nvidia_judge as nj

TOKEN = "t" * 48
BOUNDS = {"x": 300.0, "y": 350.0, "width": 220.0, "height": 260.0}


def synthetic_png(seed: int, width: int = 800, height: int = 1000) -> bytes:
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[:, :, :3] = rng.integers(60, 200, size=(height, width, 3), dtype=np.uint8)
    image[:, :, 3] = 255
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, status: int = 200):
        super().__init__(payload)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class RecordingOpener:
    def __init__(self, payload=None, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.requests = []

    def open(self, request, timeout=None):
        self.requests.append((request, timeout))
        if self.error is not None:
            raise self.error
        return FakeResponse(json.dumps(self.payload).encode("utf-8"))


class JudgeConfigTest(unittest.TestCase):
    def test_off_without_token_and_reads_token_from_env_or_private_file(self):
        missing = Path(tempfile.gettempdir()) / "retake-no-such-token"
        self.assertIsNone(nj.judge_config({}, missing))
        self.assertEqual(nj.judge_status({}, missing), "not-configured")
        config = nj.judge_config({"RETAKE_HAFIZE_PROJECT_TOKEN": TOKEN}, missing)
        self.assertEqual(config.url, nj.DEFAULT_URL)
        self.assertEqual(config.token, TOKEN)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "hafize-project-token"
            path.write_text(TOKEN + "\n", encoding="utf-8")
            self.assertEqual(nj.judge_config({}, path).token, TOKEN)
            self.assertIsNone(nj.judge_config({"RETAKE_NVIDIA_JUDGE": "off"}, path))

    def test_only_allow_listed_https_hafize_addresses(self):
        env = {"RETAKE_HAFIZE_PROJECT_TOKEN": TOKEN}
        for good in (
            "https://hafize-ua.web.app/api/project-ai/retake/vision-judge",
            "https://hafize-backend-abc123-ew.a.run.app/api/project-ai/retake/vision-judge",
            "https://hafize-free.onrender.com/api/project-ai/retake/vision-judge",
        ):
            self.assertEqual(nj.judge_config({**env, "RETAKE_HAFIZE_JUDGE_URL": good}).url, good)
        for bad in (
            "http://hafize-ua.web.app/api/project-ai/retake/vision-judge",
            "https://evil.example/api/project-ai/retake/vision-judge",
            "https://hafize-ua.web.app.evil.example/api/project-ai/retake/vision-judge",
            "https://.run.app/api/project-ai/retake/vision-judge",
            "https://hafize-ua.web.app/api/project-ai/borsa/vision-judge",
            "https://hafize-ua.web.app/api/project-ai/retake/vision-judge?next=x",
            "https://user:pw@hafize-ua.web.app/api/project-ai/retake/vision-judge",
            "https://hafize-ua.web.app:8443/api/project-ai/retake/vision-judge",
        ):
            with self.assertRaises(ValueError, msg=bad):
                nj.judge_config({**env, "RETAKE_HAFIZE_JUDGE_URL": bad})
            self.assertEqual(nj.judge_status({**env, "RETAKE_HAFIZE_JUDGE_URL": bad}), "invalid")

    def test_malformed_token_is_rejected_and_factory_disables_the_jury(self):
        with self.assertRaises(ValueError):
            nj.judge_config({"RETAKE_HAFIZE_PROJECT_TOKEN": "short"})
        self.assertIsNone(nj.build_judge({"RETAKE_HAFIZE_PROJECT_TOKEN": "has space " * 6}))
        self.assertTrue(callable(nj.build_judge({"RETAKE_HAFIZE_PROJECT_TOKEN": TOKEN})))


class FaceCropTest(unittest.TestCase):
    def test_context_crop_is_small_jpeg_around_the_face(self):
        crop = nj.face_crop_jpeg(synthetic_png(1), BOUNDS)
        self.assertTrue(crop.startswith(b"\xff\xd8\xff"))
        self.assertLessEqual(len(crop), nj.MAX_CROP_BYTES)
        decoded = cv2.imdecode(np.frombuffer(crop, np.uint8), cv2.IMREAD_COLOR)
        self.assertLessEqual(max(decoded.shape[:2]), nj.CROP_EDGE)
        # Margins include hairline and neck context, but not the whole photo.
        self.assertLess(decoded.shape[0] * decoded.shape[1], 800 * 1000)

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            nj.face_crop_jpeg(b"not an image", BOUNDS)
        with self.assertRaises(ValueError):
            nj.face_crop_jpeg(synthetic_png(1), {"x": 0, "y": 0, "width": float("nan"), "height": 10})


class RequestTest(unittest.TestCase):
    def setUp(self):
        self.config = nj.JudgeConfig(nj.DEFAULT_URL, TOKEN)
        self.crops = [nj.face_crop_jpeg(synthetic_png(seed), BOUNDS) for seed in (1, 2, 3)]

    def test_sends_only_task_and_jpeg_crops_with_server_token(self):
        opener = RecordingOpener({"decision": "winner", "winner": 2, "votes": 6, "agreement": 1, "models": ["nvidia/x-vl", "bad model!"], "meanScores": [1, 2, 9]})
        verdict = nj.request_judgement(self.config, self.crops, opener=opener)
        self.assertEqual(verdict, {"decision": "winner", "winner": 2, "votes": 6, "agreement": 1.0, "models": ["nvidia/x-vl"]})
        request, timeout = opener.requests[0]
        self.assertEqual(request.full_url, nj.DEFAULT_URL)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer " + TOKEN)
        self.assertIsNone(request.get_header("Origin"))
        self.assertEqual(timeout, nj.TIMEOUT_SECONDS)
        body = json.loads(request.data)
        self.assertEqual(set(body), {"task", "candidates"})
        self.assertEqual(body["task"], "face-realism-rank")
        self.assertEqual(len(body["candidates"]), 3)
        for url, crop in zip(body["candidates"], self.crops):
            self.assertTrue(url.startswith("data:image/jpeg;base64,"))
            self.assertEqual(base64.b64decode(url.split(",", 1)[1]), crop)

    def test_rejects_bad_counts_and_non_jpeg_payloads_before_network(self):
        opener = RecordingOpener({})
        with self.assertRaises(ValueError):
            nj.request_judgement(self.config, self.crops[:1], opener=opener)
        with self.assertRaises(ValueError):
            nj.request_judgement(self.config, self.crops * 2, opener=opener)
        with self.assertRaises(ValueError):
            nj.request_judgement(self.config, [synthetic_png(1), self.crops[0]], opener=opener)
        self.assertEqual(opener.requests, [])

    def test_http_errors_expose_only_gateway_codes(self):
        error = urllib.error.HTTPError(nj.DEFAULT_URL, 429, "Too Many", {}, io.BytesIO(b'{"error":"PROJECT_AI_UPSTREAM_LIMIT","detail":"secret"}'))
        with self.assertRaises(nj.JudgeError) as raised:
            nj.request_judgement(self.config, self.crops, opener=RecordingOpener(error=error))
        self.assertIn("429", str(raised.exception))
        self.assertIn("PROJECT_AI_UPSTREAM_LIMIT", str(raised.exception))
        self.assertNotIn("secret", str(raised.exception))
        with self.assertRaises(nj.JudgeError):
            nj.request_judgement(self.config, self.crops, opener=RecordingOpener(error=urllib.error.URLError("down")))

    def test_malformed_verdicts_are_errors(self):
        for payload in (
            [], {"decision": "maybe", "winner": None, "votes": 2, "agreement": 1},
            {"decision": "winner", "winner": 3, "votes": 2, "agreement": 1},
            {"decision": "winner", "winner": True, "votes": 2, "agreement": 1},
            {"decision": "winner", "winner": 1, "votes": 2, "agreement": 1.5},
        ):
            with self.assertRaises(nj.JudgeError, msg=payload):
                nj.request_judgement(self.config, self.crops, opener=RecordingOpener(payload))

    def test_redirects_are_never_followed(self):
        self.assertIsNone(nj._NoRedirect().redirect_request(None, None, 302, "Found", {}, "https://evil.example"))

    def test_local_consensus_is_reapplied(self):
        self.assertEqual(nj.judged_winner({"decision": "winner", "winner": 1, "votes": 3, "agreement": 0.667}, 3), 1)
        self.assertIsNone(nj.judged_winner({"decision": "winner", "winner": 1, "votes": 1, "agreement": 1}, 3))
        self.assertIsNone(nj.judged_winner({"decision": "winner", "winner": 1, "votes": 6, "agreement": 0.5}, 3))
        self.assertIsNone(nj.judged_winner({"decision": "abstain", "winner": None, "votes": 6, "agreement": 0}, 3))
        self.assertIsNone(nj.judged_winner({"decision": "winner", "winner": 5, "votes": 6, "agreement": 1}, 3))


def chat(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def jpeg(value: int) -> bytes:
    return cv2.imencode(".jpg", np.full((300, 240, 3), value, np.uint8))[1].tobytes()


class DirectTransportTest(unittest.TestCase):
    """No Hafize token, NVIDIA_API_KEY on this computer: NVIDIA is asked directly."""

    def test_key_comes_from_env_or_a_git_ignored_env_file_and_is_never_named(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            env_file = root / ".env"
            env_file.write_text("# Hafize\nOTHER=1\nexport NVIDIA_API_KEY = \"nvapi-file\"\n", encoding="utf-8")
            self.assertEqual(nj.direct_config({}, [env_file]), nj.DirectConfig("nvapi-file", str(env_file)))
            env_file.write_text("NVIDIA_API_KEY=nvapi-utf16\r\n", encoding="utf-16")
            self.assertEqual(nj.direct_config({}, [env_file]).key, "nvapi-utf16")
            env_file.write_text("NVIDIA_API_KEY=nvapi-plain # yorum\n", encoding="utf-8")
            self.assertEqual(nj.direct_config({}, [env_file]).key, "nvapi-plain")
            self.assertEqual(nj.direct_config({"NVIDIA_API_KEY": "nvapi-env"}, [env_file]).source, "ortam değişkeni")
            self.assertEqual(nj.direct_config({"RETAKE_ENV_FILE": str(env_file)}).key, "nvapi-plain")
            self.assertIsNone(nj.direct_config({"RETAKE_NVIDIA_JUDGE": "off"}, [env_file]))
            self.assertIsNone(nj.direct_config({}), "explicit env reads no files unless given")
            missing = root / "no-token"
            self.assertEqual(nj.judge_transport({}, missing, [env_file]), ("direct", str(env_file)))
            self.assertEqual(nj.judge_status({}, missing, [env_file]), "configured")
            self.assertEqual(nj.judge_transport({"RETAKE_HAFIZE_PROJECT_TOKEN": TOKEN}, missing, [env_file])[0], "hafize")
            self.assertTrue(callable(nj.build_judge({}, missing, [env_file])))
            self.assertNotIn("nvapi", nj.judge_transport({}, missing, [env_file])[1])
        self.assertIn(".env", Path(".gitignore").read_text(encoding="utf-8").splitlines())

    def test_each_model_votes_once_in_a_rotated_order_and_votes_map_back(self):
        crops = [jpeg(80), jpeg(120), jpeg(160)]
        seen = {}

        def poster(payload, key):
            self.assertEqual(key, "nvapi-k")
            self.assertNotIn("nvapi-k", json.dumps(payload))
            images = [part["image_url"]["url"] for part in payload["messages"][1]["content"] if part["type"] == "image_url"]
            seen[payload["model"]] = images
            # Every model prefers the crop that was originally index 1.
            position = images.index("data:image/jpeg;base64," + base64.b64encode(crops[1]).decode("ascii")) + 1
            return chat(f'<think>looking</think> {{"best": {position}, "reason": "pores"}}')

        verdict = nj.request_direct(nj.DirectConfig("nvapi-k", "x"), crops, poster)
        self.assertEqual(verdict, {"decision": "winner", "winner": 1, "votes": 3, "agreement": 1.0,
                                   "models": list(nj.DIRECT_MODELS)})
        orders = [tuple(images) for images in seen.values()]
        self.assertEqual(len(set(orders)), 3, "each model sees a different order")
        self.assertEqual(nj.judged_winner(verdict, 3), 1)

    def test_split_no_winner_and_unreadable_answers(self):
        crops = [jpeg(80), jpeg(160)]
        answers = iter(['{"best": 1}', '{"best": 0}', "no json"])
        verdict = nj.request_direct(nj.DirectConfig("k", "x"), crops, lambda *_a: chat(next(answers)),
                                    models=("a", "b", "c"))
        self.assertEqual((verdict["decision"], verdict["votes"]), ("abstain", 2))
        self.assertIsNone(nj.judged_winner(verdict, 2))
        with self.assertRaises(nj.JudgeError):
            nj.request_direct(nj.DirectConfig("k", "x"), crops, lambda *_a: chat("?"), models=("a",))

    def test_a_one_image_model_gets_a_numbered_contact_sheet(self):
        crops = [jpeg(80), jpeg(160)]
        sent = []

        def poster(payload, key):
            images = [part for part in payload["messages"][1]["content"] if part["type"] == "image_url"]
            sent.append(len(images))
            if len(images) > 1:
                raise nj.JudgeError("NVIDIA HTTP 400")
            self.assertIn("panel", payload["messages"][1]["content"][0]["text"])
            return chat('{"best": 2}')

        verdict = nj.request_direct(nj.DirectConfig("k", "x"), crops, poster, models=("only-one-image",))
        self.assertEqual(sent, [2, 1])
        self.assertEqual(verdict["winner"], 1)
        sheet = cv2.imdecode(np.frombuffer(nj.contact_sheet(crops), np.uint8), 1)
        self.assertEqual(sheet.shape[0], nj.SHEET_PANEL_HEIGHT)

    def test_direct_judge_sends_review_sheets_and_falls_back_to_face_crops(self):
        with tempfile.TemporaryDirectory() as folder:
            env_file = Path(folder) / ".env"
            env_file.write_text("NVIDIA_API_KEY=nvapi-k\n", encoding="utf-8")
            judge = nj.build_judge({}, Path(folder) / "no-token", [env_file])
        self.assertTrue(getattr(judge, "accepts_images", False))
        images = [synthetic_png(1), synthetic_png(2)]
        prompts = []

        def poster(payload, key):
            prompts.append(payload["messages"][1]["content"][0]["text"])
            return chat('{"best": 1}')

        verdict = judge(images, BOUNDS, None, poster)
        self.assertEqual(verdict["layout"], nj.LAYOUT_SHEET)
        self.assertTrue(all("review sheet" in text for text in prompts))
        prompts.clear()

        def sheet_refused(payload, key):
            text = payload["messages"][1]["content"][0]["text"]
            if "review sheet" in text:
                raise nj.JudgeError("NVIDIA HTTP 500")
            prompts.append(text)
            return chat('{"best": 2}')

        verdict = judge(images, BOUNDS, None, sheet_refused)
        self.assertEqual(verdict["layout"], nj.LAYOUT_FACE)
        self.assertEqual(verdict["votes"], len(nj.DIRECT_MODELS))

    def test_http_layer_targets_only_nvidia_and_never_follows_redirects(self):
        opener = RecordingOpener(chat('{"best": 1}'))
        nj._post_direct({"model": "m"}, "nvapi-k", opener)
        request, timeout = opener.requests[0]
        self.assertEqual(request.full_url, "https://integrate.api.nvidia.com/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer nvapi-k")
        error = urllib.error.HTTPError(nj.DIRECT_URL, 401, "Unauthorized", {}, io.BytesIO(b'{"detail":"nvapi-leak"}'))
        with self.assertRaisesRegex(nj.JudgeError, "HTTP 401") as caught:
            nj._post_direct({"model": "m"}, "nvapi-k", RecordingOpener(error=error))
        self.assertNotIn("nvapi", str(caught.exception))


def ultra_request(consent: bool):
    return (
        b"original photo", [b"reference"] * 25, ".png", [".png"] * 25,
        b"original appearance", ".png", {
            "variation": 0, "referenceWeights": [1] * 25, "nvidiaJudge": consent,
            "faceBounds": dict(BOUNDS)
        }
    )


class UltraJuryTest(unittest.TestCase):
    """The jury may only choose among verified identity near-ties."""

    def run_search(self, judge, consent=True, identities=None):
        identities = identities or {0: .800, 1: .7995, 2: .7990, 3: .700, 4: .7992, 5: .7991}
        images = {index: synthetic_png(100 + index) for index in identities}

        def fake_swap(*args):
            variation = args[3][6]["variation"]
            if variation not in identities:
                raise RuntimeError("no more rounds in this test")
            return images[variation], {
                "X-Retake-Similarity": f"{identities[variation]:.4f}",
                "X-Retake-Detail-Gate": "passed",
                "X-Retake-Compositing": "preblended",
            }

        with patch.object(srv, "process_swap", fake_swap):
            photo, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, ultra_request(consent), lambda *_a, **_k: None,
                {"nvidia_judge": judge}, min_duration_seconds=0
            )
        return photo, headers, images

    def test_switches_only_to_a_near_tie_and_sends_at_most_four_crops(self):
        calls = []

        def judge(crops):
            calls.append(crops)
            return {"decision": "winner", "winner": 2, "votes": 6, "agreement": 1.0, "models": ["nvidia/x-vl"]}

        photo, headers, images = self.run_search(judge)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 4)
        self.assertTrue(all(crop.startswith(b"\xff\xd8\xff") for crop in calls[0]))
        self.assertEqual(headers["X-Retake-NVIDIA-Judge"], "switched")
        self.assertEqual(headers["X-Retake-NVIDIA-Judge-Finalists"], "4")
        self.assertEqual(headers["X-Retake-NVIDIA-Judge-Models"], "nvidia/x-vl")
        self.assertNotEqual(photo, images[3], "an identity outlier must never be offered to the jury")
        self.assertNotEqual(photo, images[0])
        self.assertGreaterEqual(float(headers["X-Retake-Similarity"]), .800 - srv.ULTRA_REALISM_TIE_LOSS)

    def test_an_image_judge_gets_the_delivered_pixels_and_the_face_box(self):
        calls = []

        def judge(images, bounds, landmarks=None):
            calls.append((images, bounds, landmarks))
            return {"decision": "winner", "winner": 1, "votes": 6, "agreement": 1.0, "models": [], "layout": "review-sheet"}

        judge.accepts_images = True
        photo, headers, images = self.run_search(judge)
        self.assertEqual(len(calls), 1)
        sent, bounds, _ = calls[0]
        self.assertTrue(all(image in images.values() for image in sent), "full candidate PNGs, the sheet is made inside")
        self.assertIn("width", bounds)
        self.assertEqual(headers["X-Retake-NVIDIA-Judge-Layout"], "review-sheet")

    def test_agreeing_split_and_failing_juries_keep_the_local_winner(self):
        for verdict, outcome in (
            ({"decision": "winner", "winner": 0, "votes": 6, "agreement": 1.0, "models": []}, "kept"),
            ({"decision": "abstain", "winner": None, "votes": 6, "agreement": 0.5, "models": []}, "abstain"),
            ({"decision": "winner", "winner": 1, "votes": 6, "agreement": 0.5, "models": []}, "abstain"),
        ):
            photo, headers, images = self.run_search(lambda _crops, verdict=verdict: verdict)
            self.assertEqual(photo, images[0], outcome)
            self.assertEqual(headers["X-Retake-NVIDIA-Judge"], outcome)

        def broken(_crops):
            raise nj.JudgeError("down")

        photo, headers, images = self.run_search(broken)
        self.assertEqual(photo, images[0])
        self.assertEqual(headers["X-Retake-NVIDIA-Judge"], "error")

    def test_without_consent_or_configuration_no_crop_leaves_the_computer(self):
        calls = []
        photo, headers, images = self.run_search(lambda crops: calls.append(crops), consent=False)
        self.assertEqual(calls, [])
        self.assertEqual(headers["X-Retake-NVIDIA-Judge"], "not-consented")
        self.assertEqual(photo, images[0])
        photo, headers, images = self.run_search(None, consent=True)
        self.assertEqual(headers["X-Retake-NVIDIA-Judge"], "off")

    def test_no_near_tie_means_nothing_to_judge(self):
        calls = []
        photo, headers, images = self.run_search(lambda crops: calls.append(crops), identities={0: .80, 1: .70, 2: .60})
        self.assertEqual(calls, [])
        self.assertEqual(headers["X-Retake-NVIDIA-Judge"], "single-finalist")
        self.assertEqual(photo, images[0])

    def test_a_soft_original_counts_as_verified_detail_but_a_failed_gate_does_not(self):
        def item(round_, gate):
            return (f"img-{round_}".encode(), {"X-Retake-Similarity": "0.8000", "X-Retake-Detail-Gate": gate}, round_)
        best = item(1, "soft-original")
        pool = srv.judge_pool(best[0], best[1], 1, [item(2, "soft-original"), item(3, "failed:eyes-blur"),
                                                     item(4, "passed")], .800)
        self.assertEqual([entry[2] for entry in pool], [1, 2, 4])

    def test_max_search_tries_a_fresh_local_candidate_for_single_finalist(self):
        from tools.ultra_mode import verified_candidate_score

        request = ultra_request(True)
        request[6]["qualityDepth"] = "max"
        original = synthetic_png(123)
        alternate = synthetic_png(124)
        calls = []
        rendered_specs = []
        planned = srv.plan_extra_spec

        def fake_swap(*args):
            metadata = args[3][6]
            spec = (metadata["variation"], metadata.get("variationParams"))
            rendered_specs.append(spec)
            return (alternate if retry_planned else original), {
                "X-Retake-Similarity": "0.8000",
                "X-Retake-Detail-Gate": "passed",
            }

        retry_planned = False

        def plan(*args):
            nonlocal retry_planned
            retry_planned = True
            return planned(*args)

        def judge(crops):
            calls.append(crops)
            return {"decision": "winner", "winner": 0, "votes": 3,
                    "agreement": 1.0, "models": ["test"]}

        with patch.object(srv, "process_swap", fake_swap), patch.object(srv, "plan_extra_spec", plan):
            photo, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, request, lambda *_a, **_k: None,
                {"nvidia_judge": judge}, min_duration_seconds=0
            )
        self.assertEqual(headers["X-Retake-NVIDIA-Judge-Retry-Attempts"], "1")
        self.assertEqual(headers["X-Retake-NVIDIA-Judge-Finalists"], "2")
        self.assertEqual(len(calls), 1)
        self.assertEqual(photo, original)
        self.assertEqual(verified_candidate_score({
            "X-Retake-Similarity": "0.8", "X-Retake-Detail-Gate": "soft-original"}), .8)
        self.assertIsNone(verified_candidate_score({
            "X-Retake-Similarity": "0.9", "X-Retake-Detail-Gate": "failed:eyes-blur"}))

    def test_metadata_consent_must_be_boolean(self):
        base = {"referenceWeights": [1] * 25, "faceBounds": {"x": 10, "y": 10, "width": 100, "height": 100},
                "targetSize": {"width": 784, "height": 1024}}
        self.assertFalse(srv._validate_metadata(base, 25)["nvidiaJudge"])
        self.assertTrue(srv._validate_metadata({**base, "nvidiaJudge": True}, 25)["nvidiaJudge"])
        with self.assertRaises(ValueError):
            srv._validate_metadata({**base, "nvidiaJudge": "yes"}, 25)


if __name__ == "__main__":
    unittest.main()


class SafeJudgeDiagnosticsTest(unittest.TestCase):
    def test_only_safe_error_categories_reach_browser(self):
        self.assertEqual(nj.judge_headers("error",2,error=nj.JudgeError("secret","http-401"))["X-Retake-NVIDIA-Judge-Error"],"http-401")
        self.assertEqual(nj.judge_headers("error",2,error=nj.JudgeError("secret","private token"))["X-Retake-NVIDIA-Judge-Error"],"unavailable")
        self.assertGreater(nj.TIMEOUT_SECONDS,120)


class JudgeReasonTest(unittest.TestCase):
    def test_safe_reason_survives_transport(self):
        verdict = nj._normalize_verdict({"decision":"abstain","winner":None,"votes":1,"agreement":0,"reason":"NOT_ENOUGH_VOTES"},2)
        self.assertEqual(nj.judge_headers("abstain",2,verdict)["X-Retake-NVIDIA-Judge-Reason"],"NOT_ENOUGH_VOTES")
        error = nj.JudgeError("private message","http-502","PROJECT_AI_UPSTREAM_FAILED")
        self.assertEqual(nj.judge_headers("error",2,error=error)["X-Retake-NVIDIA-Judge-Reason"],"PROJECT_AI_UPSTREAM_FAILED")
        error = nj.JudgeError("private message","http-502","secret\nheader")
        self.assertNotIn("X-Retake-NVIDIA-Judge-Reason",nj.judge_headers("error",2,error=error))

class JuryDiversityTest(unittest.TestCase):
    def test_duplicate_images_do_not_consume_finalist_slots(self):
        headers = {"X-Retake-Similarity": ".8", "X-Retake-Detail-Gate": "passed"}
        rows = [(b"a",headers,1),(b"a",headers,2),(b"b",headers,1),(b"c",headers,3)]
        retained = []
        for row in rows:
            retained = srv.retain_judge_finalists(retained,row,.8)
        self.assertEqual(len(retained),3)
        pool = srv.judge_pool(b"a",headers,1,retained,.8)
        self.assertEqual({row[0] for row in pool},{b"a",b"b",b"c"})
    def test_distinct_same_round_upgrade_may_be_judged(self):
        headers = {"X-Retake-Similarity": ".8", "X-Retake-Detail-Gate": "passed"}
        pool = srv.judge_pool(b"original",headers,1,[(b"upgrade",headers,1)],.8)
        self.assertEqual(len(pool),2)


class NeckMetadataTest(unittest.TestCase):
    def test_neck_scope_is_explicit_and_boolean(self):
        base={"referenceWeights":[1]*25,"faceBounds":{"x":10,"y":10,"width":100,"height":100},"targetSize":{"width":784,"height":1024}}
        self.assertFalse(srv._validate_metadata(base,25)["neckHarmonize"])
        self.assertTrue(srv._validate_metadata({**base,"neckHarmonize":True},25)["neckHarmonize"])
        for invalid in ("true",1,None):
            with self.assertRaises(ValueError):
                srv._validate_metadata({**base,"neckHarmonize":invalid},25)


class ReviewSheetTest(unittest.TestCase):
    """v28: one JPEG per finalist with eye/brow and mouth close-ups."""

    def painted_png(self):
        image = np.full((1000, 800, 4), 255, np.uint8)
        image[:, :, :3] = (120, 130, 140)
        # Eyes band red, mouth band blue (BGR), inside the face box.
        image[350 + 60:350 + 120, 310:510, :3] = (0, 0, 255)
        image[350 + 190:350 + 240, 360:460, :3] = (255, 0, 0)
        ok, encoded = cv2.imencode(".png", image)
        return encoded.tobytes()

    def test_sheet_shows_the_face_and_enlarged_close_ups(self):
        sheet = nj.review_sheet_jpeg(self.painted_png(), BOUNDS)
        self.assertTrue(sheet.startswith(b"\xff\xd8\xff"))
        self.assertLessEqual(len(sheet), nj.MAX_CROP_BYTES)
        decoded = cv2.imdecode(np.frombuffer(sheet, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape[0], nj.SHEET_HEIGHT)
        right = decoded[:, decoded.shape[1] - nj.SHEET_CLOSEUP_WIDTH:]
        half = (nj.SHEET_HEIGHT - nj.SHEET_GAP) // 2
        eyes, mouth = right[:half], right[nj.SHEET_HEIGHT - half:]
        red = lambda panel: float(((panel[:, :, 2] > 200) & (panel[:, :, 0] < 60)).mean())
        blue = lambda panel: float(((panel[:, :, 0] > 200) & (panel[:, :, 2] < 60)).mean())
        self.assertGreater(red(eyes), 0.15, "the eye close-up shows the eye band")
        self.assertGreater(blue(mouth), 0.10, "the mouth close-up shows the mouth")
        self.assertLess(blue(eyes), 0.02)

    def test_mesh_boxes_follow_the_landmarks(self):
        from tests.test_fine_features import synthetic_mesh
        mesh = synthetic_mesh()
        boxes = nj.closeup_boxes({"x": 90, "y": 100, "width": 220, "height": 260}, mesh.tolist(), (420, 400))
        x0, y0, x1, y1 = boxes["eyes"]
        self.assertTrue(x0 < 112 and x1 > 288 and y0 < 138 and y1 > 182)
        mx0, my0, mx1, my1 = boxes["mouth"]
        self.assertTrue(mx0 < 155 and mx1 > 245 and my0 < 274 and my1 > 306)
        fallback = nj.closeup_boxes({"x": 90, "y": 100, "width": 220, "height": 260}, None, (420, 400))
        self.assertEqual(set(fallback), {"eyes", "mouth"})

    def test_judge_sends_sheets_and_falls_back_for_an_older_hafize(self):
        judge = nj.build_judge({"RETAKE_HAFIZE_PROJECT_TOKEN": TOKEN})
        self.assertTrue(getattr(judge, "accepts_images", False))
        images = [synthetic_png(seed) for seed in (1, 2)]
        ok = RecordingOpener({"decision": "abstain", "winner": None, "votes": 2, "agreement": 0.5, "models": []})
        verdict = judge(images, BOUNDS, None, opener=ok)
        self.assertEqual(verdict["layout"], "review-sheet")
        body = json.loads(ok.requests[0][0].data)
        self.assertEqual(body["layout"], "review-sheet")
        self.assertEqual(set(body), {"task", "candidates", "layout"})
        self.assertEqual(nj.judge_headers("abstain", 2, verdict)["X-Retake-NVIDIA-Judge-Layout"], "review-sheet")

        class OldHafize(RecordingOpener):
            def open(self, request, timeout=None):
                self.requests.append((request, timeout))
                if "layout" in json.loads(request.data):
                    raise urllib.error.HTTPError(request.full_url, 400, "bad", {}, io.BytesIO(b'{"error":"PROJECT_AI_INVALID_INPUT"}'))
                return FakeResponse(json.dumps(self.payload).encode("utf-8"))

        old = OldHafize({"decision": "winner", "winner": 1, "votes": 4, "agreement": 1, "models": []})
        verdict = judge(images, BOUNDS, None, opener=old)
        self.assertEqual(verdict["layout"], "face")
        self.assertEqual(verdict["winner"], 1)
        self.assertEqual(len(old.requests), 2)
        self.assertNotIn("layout", json.loads(old.requests[1][0].data))

        class Down(RecordingOpener):
            def open(self, request, timeout=None):
                self.requests.append((request, timeout))
                raise urllib.error.HTTPError(request.full_url, 503, "down", {}, io.BytesIO(b'{"error":"PROJECT_AI_UPSTREAM_FAILED"}'))

        down = Down()
        with self.assertRaises(nj.JudgeError):
            judge(images, BOUNDS, None, opener=down)
        self.assertEqual(len(down.requests), 1, "only a layout refusal is retried")

    def test_bad_layout_is_refused_before_network(self):
        with self.assertRaises(ValueError):
            nj.request_judgement(nj.JudgeConfig(nj.DEFAULT_URL, TOKEN), [nj.face_crop_jpeg(synthetic_png(1), BOUNDS)] * 2,
                                 opener=RecordingOpener({}), layout="poster")


class JuryLocalQualityGuardTest(unittest.TestCase):
    def setUp(self):
        self.best={"X-Retake-Similarity":".832","X-Retake-Detail-Gate":"passed",
                   "X-Retake-Fine-Quality":".90","X-Retake-Reference-Appearance":".85",
                   "X-Retake-Detail-Score":".88","X-Retake-Rim-Fraction":".03",
                   "X-Retake-Detail-Regions":"eyes:1.01,mouth:1.03,lower_face:1.0",
                   "X-Retake-Fine-Detail":"eye_r:0.95>1.01,eye_l:1.0>1.0,lips:1.0>1.0"}

    def test_jury_cannot_override_substantial_local_loss_or_missing_evidence(self):
        from tools.ultra_mode import judge_quality_guard
        for key,bad in (("X-Retake-Fine-Quality",".65"),("X-Retake-Reference-Appearance",".6"),
                        ("X-Retake-Detail-Score",".7"),("X-Retake-Rim-Fraction",".15"),
                        ("X-Retake-Fine-Quality","nan"),("X-Retake-Detail-Regions","eyes:1.01,mouth:1.03"),
                        ("X-Retake-Fine-Detail","eye_r:1.0>0.65,eye_l:1.0>1.0,lips:1.0>1.0")):
            candidate={**self.best,"X-Retake-Similarity":".831",key:bad}
            self.assertFalse(judge_quality_guard(candidate,self.best)[0],key)
            self.assertEqual(len(srv.judge_pool(b"best",self.best,1,[(b"bad",candidate,2)],.832)),1)
        safe={**self.best,"X-Retake-Similarity":".831","X-Retake-Rim-Fraction":".02"}
        self.assertEqual(judge_quality_guard(safe,self.best),(True,"passed"))
        self.assertEqual(len(srv.judge_pool(b"best",self.best,1,[(b"safe",safe,2)],.832)),2)

    def test_jury_cannot_switch_to_a_clearly_less_natural_photo_but_accepts_a_soft_original(self):
        from tools.ultra_mode import judge_quality_guard
        best = {**self.best, "X-Retake-Naturalness": "80.0"}
        edited = {**best, "X-Retake-Similarity": ".831", "X-Retake-Naturalness": "72.0"}
        self.assertEqual(judge_quality_guard(edited, best), (False, "naturalness"))
        close = {**best, "X-Retake-Similarity": ".831", "X-Retake-Naturalness": "77.0"}
        self.assertTrue(judge_quality_guard(close, best)[0])
        soft = {**close, "X-Retake-Detail-Gate": "soft-original"}
        self.assertTrue(judge_quality_guard(soft, best)[0])
        failed = {**close, "X-Retake-Detail-Gate": "failed:eyes-blur"}
        self.assertEqual(judge_quality_guard(failed, best), (False, "detail-gate"))

    def test_malformed_or_contradictory_consensus_never_changes_winner(self):
        good={"decision":"winner","winner":1,"votes":2,"agreement":1.0,"reason":"CONSENSUS"}
        self.assertEqual(nj.judged_winner(good,2),1)
        for changes in ({"agreement":float("nan")},{"agreement":1.2},{"agreement":True},
                        {"votes":True},{"votes":33},{"reason":"SPLIT_VOTE"},{"reason":"NARROW_MARGIN"}):
            self.assertIsNone(nj.judged_winner({**good,**changes},2),changes)
        headers=nj.judge_headers("local-guard",2,{**good,"agreement":float("nan"),"models":["good/model","bad\r\nHeader"]})
        self.assertEqual(headers["X-Retake-NVIDIA-Judge"],"local-guard")
        self.assertNotIn("X-Retake-NVIDIA-Judge-Agreement",headers)
        self.assertEqual(headers["X-Retake-NVIDIA-Judge-Models"],"good/model")
