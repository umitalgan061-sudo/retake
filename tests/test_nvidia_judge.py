"""NVIDIA vision judge: parsing, model fallback, order-reversed passes, Ultra use.

No network: the HTTP poster and the judge are fakes.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from tools import facefusion_local_server as srv
from tools import nvidia_judge as nj
from tools.ultra_mode import ULTRA_JUDGE_GAIN, judge_pick, judge_pool


def png(value: int, size=(240, 200), alpha=False) -> bytes:
    image = np.full((size[0], size[1], 4 if alpha else 3), value, np.uint8)
    if alpha:
        image[:, :, 3] = 0
        image[60:180, 50:150, 3] = 255
    return cv2.imencode(".png", image)[1].tobytes()


def reply(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


class ParseTest(unittest.TestCase):
    def test_reads_json_after_reasoning_and_prose(self):
        text = ('<think>image 2 has a seam {not json}</think> Sure! '
                '{"candidates": [{"image": 2, "realism": 6.5, "issues": ["seam", "made_up"]},'
                ' {"image": 3, "realism": 11, "issues": []}], "best": 3} trailing')
        verdict = nj.parse_verdict(text, 2)
        self.assertEqual(verdict[0], {"realism": 6.5, "issues": ["seam"]})
        self.assertEqual(verdict[1]["realism"], 10.0)        # clamped

    def test_incomplete_or_invalid_answers_are_rejected(self):
        self.assertIsNone(nj.parse_verdict('{"candidates": [{"image": 2, "realism": 7}]}', 2))
        self.assertIsNone(nj.parse_verdict('{"candidates": [{"image": 2, "realism": "nan"}]}', 1))
        self.assertIsNone(nj.parse_verdict("no json here", 1))
        self.assertIsNone(nj.parse_verdict(None, 1))


class AskTest(unittest.TestCase):
    def test_falls_back_to_the_next_model_and_never_sends_the_key_in_the_body(self):
        seen = []

        def poster(payload, key):
            seen.append(payload["model"])
            self.assertNotIn("secret-key", str(payload))
            if payload["model"] == nj.MODELS[0]:
                raise nj.JudgeError("busy", 503)
            return reply('{"candidates": [{"image": 2, "realism": 8, "issues": []}], "best": 2}')

        model, verdict = nj.ask([b"a", b"b"], "secret-key", poster)
        self.assertEqual(model, nj.MODELS[1])
        self.assertEqual(seen, list(nj.MODELS[:2]))
        self.assertEqual(verdict[0]["realism"], 8.0)

    def test_request_holds_the_original_first_then_each_candidate_as_an_image(self):
        captured = {}

        def poster(payload, key):
            captured.update(payload)
            return reply('{"candidates": [{"image": 2, "realism": 5, "issues": []},'
                         '{"image": 3, "realism": 6, "issues": []}]}')

        nj.ask([b"orig", b"c1", b"c2"], "k", poster)
        content = captured["messages"][1]["content"]
        self.assertEqual([part["type"] for part in content], ["text", "image_url", "image_url", "image_url"])
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_a_one_image_model_gets_a_numbered_contact_sheet(self):
        sent = []

        def poster(payload, key):
            images = [part for part in payload["messages"][1]["content"] if part["type"] == "image_url"]
            sent.append(len(images))
            if len(images) > 1:
                raise nj.JudgeError("only one image", 400)
            self.assertIn("panel", payload["messages"][1]["content"][0]["text"])
            return reply('{"candidates": [{"image": 2, "realism": 6, "issues": []}]}')

        crop = nj.face_crop(np.full((300, 240, 3), 128, np.uint8), None)
        model, verdict = nj.ask([crop, crop], "k", poster)
        self.assertEqual((model, sent), (nj.MODELS[0], [2, 1]))
        sheet = cv2.imdecode(np.frombuffer(nj.contact_sheet([crop, crop, crop]), np.uint8), 1)
        self.assertEqual(sheet.shape[0], nj.SHEET_PANEL_HEIGHT)
        self.assertGreater(sheet.shape[1], 3 * 200)

    def test_second_pass_reverses_the_order_and_maps_scores_back(self):
        calls = []

        def poster(payload, key):
            calls.append(payload["model"])
            if len(calls) == 1:   # order: best, other
                return reply('{"candidates": [{"image": 2, "realism": 5, "issues": ["seam", "tone"]},'
                             '{"image": 3, "realism": 8, "issues": []}]}')
            # reversed order: other, best
            return reply('{"candidates": [{"image": 2, "realism": 7, "issues": []},'
                         '{"image": 3, "realism": 4, "issues": ["seam"]}]}')

        report = nj.judge(b"o", [b"best", b"other"], "k", poster)
        self.assertEqual(report["passes"], [[5.0, 8.0], [4.0, 7.0]])
        self.assertEqual(report["scores"], [4.5, 7.5])
        self.assertEqual(report["issues"][0], ["seam"])       # only issues both passes saw
        self.assertEqual(calls, [nj.MODELS[0], nj.MODELS[0]])  # second pass uses the same model


class LocalTest(unittest.TestCase):
    def test_key_comes_only_from_the_environment_and_can_be_switched_off(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": " nvapi-x ", "RETAKE_NVIDIA_JUDGE": ""}):
            self.assertEqual(nj.api_key(), "nvapi-x")
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-x", "RETAKE_NVIDIA_JUDGE": "0"}):
            self.assertIsNone(nj.api_key())
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(nj.api_key())

    def test_crop_is_small_and_alpha_results_are_seen_over_the_photo(self):
        flat = nj.flatten(png(200, alpha=True), png(50))
        self.assertEqual(flat.shape, (240, 200, 3))
        self.assertEqual(int(flat[0, 0, 0]), 50)
        self.assertEqual(int(flat[100, 100, 0]), 200)
        big = np.zeros((2000, 1500, 3), np.uint8)
        crop = cv2.imdecode(np.frombuffer(nj.face_crop(big, {"x": 500, "y": 600, "width": 400, "height": 500}), np.uint8), 1)
        self.assertLessEqual(max(crop.shape[:2]), nj.CROP_SIDE)

    def test_describe(self):
        self.assertEqual(nj.describe(8.25, ["eyes", "seam"]), "NVIDIA hakem 8.2/10 · gözler, kenar dikişi")
        self.assertEqual(nj.describe(None, []), "")


def heads(identity, naturalness=None):
    headers = {"X-Retake-Similarity": str(identity)}
    if naturalness is not None:
        headers["X-Retake-Naturalness"] = str(naturalness)
    return headers


class PickTest(unittest.TestCase):
    def test_pool_holds_only_identity_near_ties_that_are_not_less_natural(self):
        best = (b"best", heads(0.850, 80), 1)
        others = [(b"close", heads(0.845, 82), 2), (b"far", heads(0.830, 95), 3),
                  (b"pasted", heads(0.849, 60), 4), (b"best", heads(0.850, 80), 1),
                  (b"close2", heads(0.842, 79), 5), (b"close3", heads(0.841, 79), 6)]
        pool = judge_pool(best, others)
        self.assertEqual([item[0] for item in pool], [b"best", b"close", b"close2"])

    def test_a_switch_needs_both_passes_and_a_clear_lead(self):
        self.assertEqual(judge_pick([[6.0, 8.0], [6.5, 7.8]]), 1)
        self.assertEqual(judge_pick([[6.0, 8.0], [7.0, 6.9]]), 0, "passes disagree")
        self.assertEqual(judge_pick([[6.0, 6.5], [6.0, 6.8]]), 0, "lead under ULTRA_JUDGE_GAIN")
        self.assertEqual(judge_pick([[4.0, 9.0]]), 0, "a single pass never switches")
        self.assertEqual(judge_pick([[5, 7, 8], [5, 7.5, 8.5]]), 2)
        self.assertGreaterEqual(ULTRA_JUDGE_GAIN, 1.0)


class UltraJudgeTest(unittest.TestCase):
    def request(self):
        return (png(90), [b"reference"] * 25, ".png", [".png"] * 25, b"appearance", ".png",
                {"variation": 0, "referenceWeights": [1] * 25,
                 "faceBounds": {"x": 40, "y": 40, "width": 120, "height": 150}})

    def run_ultra(self, judge, identities):
        def fake_swap(*args):
            index = args[3][6]["variation"]
            value = identities[index] if index < len(identities) else 0.5
            return png(100 + index), {"X-Retake-Similarity": str(value), "X-Retake-Naturalness": "80"}
        with patch.object(srv, "process_swap", fake_swap):
            return srv.run_ultra_search(Path("."), "1024x1024", False, self.request(),
                                        lambda *_a, **_k: None, min_duration_seconds=0, judge=judge)

    def test_the_judge_can_pick_a_near_tie_it_finds_clearly_more_real(self):
        shown = []

        def judge(original, candidates):
            shown.append(len(candidates))
            return {"model": "m", "scores": [5.0, 8.0], "passes": [[5, 8], [5, 8]],
                    "issues": [["plastic_skin"], []]}

        photo, headers = self.run_ultra(judge, [0.800, 0.851, 0.848])
        self.assertEqual(shown, [2])               # 0.800 is not a near tie
        self.assertEqual(photo, png(102))                       # round 3 (0.848), not 0.851
        self.assertEqual(headers["X-Retake-Ultra-Winner"], "3")
        self.assertEqual(headers["X-Retake-Judge"], "8.0")
        self.assertEqual(headers["X-Retake-Judge-Switched"], "yes")

    def test_a_failing_judge_keeps_the_verified_best(self):
        def judge(original, candidates):
            raise nj.JudgeError("NVIDIA istek sınırı doldu.", 429)

        photo, headers = self.run_ultra(judge, [0.800, 0.851, 0.848])
        self.assertEqual(photo, png(101))
        self.assertNotIn("X-Retake-Judge", headers)

    def test_identity_is_never_traded_for_looks(self):
        def judge(original, candidates):
            # Only the best is a near tie (others are > 0.010 behind).
            self.assertEqual(len(candidates), 1)
            return {"model": "m", "scores": [6.0], "passes": [[6.0]], "issues": [[]]}

        photo, headers = self.run_ultra(judge, [0.800, 0.851, 0.830])
        self.assertEqual(photo, png(101))
        self.assertEqual(headers["X-Retake-Judge-Switched"], "no")


if __name__ == "__main__":
    unittest.main()
