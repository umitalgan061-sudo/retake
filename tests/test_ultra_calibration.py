"""Learned Ultra variant order (numbers only; no GPU, no photos)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import ultra_calibration as uc


def fid(ssim, delta_e):
    return {"X-Retake-Self-Fidelity": f"ssim:{ssim:.4f},deltaE:{delta_e:.3f}", "X-Retake-Detail-Score": "0.9"}


def proxy(detail, reinforced="off"):
    return {"X-Retake-Detail-Score": f"{detail:.4f}", "X-Retake-Identity-Reinforced": reinforced}


class QualityTest(unittest.TestCase):
    def test_gold_fidelity_wins_over_proxy_and_is_bounded(self):
        kind, value = uc.round_quality(fid(0.97, 1.3))
        self.assertEqual(kind, "fidelity")
        self.assertAlmostEqual(value, 0.944)
        self.assertAlmostEqual(uc.round_quality(fid(0.6929, 7.152))[1], 0.6929 - 0.14304, places=4)
        self.assertEqual(uc.round_quality(fid(0.1, 50))[1], 0.0)
        self.assertEqual(uc.round_quality(proxy(0.9)), ("proxy", 0.9))
        self.assertAlmostEqual(uc.round_quality(proxy(0.9, "on"))[1], 0.85)
        for bad in ({}, {"X-Retake-Detail-Score": "nan"}, {"X-Retake-Detail-Score": "2"},
                    {"X-Retake-Self-Fidelity": "ssim:9,deltaE:1"}):
            self.assertNotEqual(uc.round_quality(bad), ("fidelity", 0.0)) if "Self" in str(bad) else None
        self.assertIsNone(uc.round_quality({}))
        self.assertIsNone(uc.round_quality({"X-Retake-Detail-Score": "nan"}))
        self.assertIsNone(uc.parse_self_fidelity("ssim:9,deltaE:1"))
        self.assertIsNone(uc.parse_self_fidelity("<x>"))

    def test_keys_are_stable_and_change_with_bank_or_meaning(self):
        a = uc.bank_key([b"a", b"b"], [1, 2])
        self.assertEqual(a, uc.bank_key([b"a", b"b"], [1, 2]))
        self.assertNotEqual(a, uc.bank_key([b"a", b"c"], [1, 2]))
        self.assertNotEqual(a, uc.bank_key([b"a", b"b"], [2, 1]))
        self.assertNotEqual(uc.schedule_signature("v22", range(24)), uc.schedule_signature("v23", range(24)))


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / "cal.json"
        self.store = uc.CalibrationStore(self.path, signature="sig")

    def tearDown(self):
        self.folder.cleanup()

    def test_default_order_until_enough_evidence(self):
        self.assertEqual(self.store.schedule("bank", range(6)), (list(range(6)), "default", 0))
        self.store.record("bank", [(0, proxy(.8)), (1, proxy(.9))])
        self.assertEqual(self.store.schedule("bank", range(6))[1], "default")

    def test_self_tests_put_the_most_faithful_variant_first_but_keep_exploring(self):
        # Three self-tests: variant 4 reproduces the person's real photo best.
        for _ in range(3):
            self.store.record("bank", [(0, fid(.70, 7.0)), (4, fid(.92, 2.0)), (1, fid(.80, 4.0))], self_test=True)
        order, mode, samples = self.store.schedule("bank", range(8))
        self.assertEqual(mode, "learned-fidelity")
        self.assertEqual(samples, 9)
        self.assertEqual(order[0], 4)
        self.assertLess(order.index(1), order.index(0))
        # Untested variants are explored before a proven-bad one.
        self.assertLess(order.index(5), order.index(0))
        self.assertEqual(sorted(order), list(range(8)))

    def test_proxy_learning_when_no_self_tests(self):
        for _ in range(3):
            self.store.record("bank", [(0, proxy(.70, "on")), (2, proxy(.95))])
        order, mode, _ = self.store.schedule("bank", range(3))
        self.assertEqual(mode, "learned-proxy")
        self.assertEqual(order[0], 2)

    def test_banks_are_isolated_persisted_and_reset(self):
        for _ in range(3):
            self.store.record("alice", [(0, fid(.6, 8)), (1, fid(.6, 8)), (2, fid(.6, 8)), (3, fid(.95, 1.0))], self_test=True)
        self.assertEqual(self.store.schedule("bob", range(4))[1], "default")
        again = uc.CalibrationStore(self.path, signature="sig")
        self.assertEqual(again.schedule("alice", range(4))[0][0], 3)
        summary = again.summary()
        self.assertEqual(summary["banks"]["alice"]["self_tests"], 3)
        self.assertEqual(summary["banks"]["alice"]["variants"]["3"]["fidelity"], {"n": 3, "mean": 0.93})
        self.assertNotIn("photo", json.dumps(summary).lower())
        again.reset()
        self.assertFalse(self.path.exists())
        self.assertEqual(again.schedule("alice", range(4))[1], "default")

    def test_foreign_corrupt_or_disabled_stores_fall_back_safely(self):
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(self.store.schedule("bank", range(3))[1], "default")
        self.store.record("bank", [(0, proxy(.5))])
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["signature"], "sig")
        other = uc.CalibrationStore(self.path, signature="different-meaning")
        self.assertEqual(other.summary()["banks"], {})
        disabled = uc.CalibrationStore(self.path, signature="sig", enabled=False)
        self.assertEqual(disabled.record("bank", [(0, proxy(.5))]), 0)
        self.assertEqual(disabled.schedule("bank", range(3))[1], "default")
        self.assertEqual(self.store.record("bank", [("x", proxy(.5)), (1, {})]), 0)

    def test_old_banks_are_pruned(self):
        for index in range(uc.MAX_BANKS + 3):
            self.store.record(f"bank{index}", [(0, proxy(.5))])
        self.assertLessEqual(len(self.store.summary()["banks"]), uc.MAX_BANKS)


if __name__ == "__main__":
    unittest.main()


class UltraLoopLearningTest(unittest.TestCase):
    """The search runs learned variants first and records what it measured."""

    def run_search(self, calibration, rounds=3, exact=None, fail_schedule=False):
        from unittest.mock import patch
        from tools import facefusion_local_server as srv
        seen = []

        def fake_swap(*args):
            variation = args[3][6]["variation"]
            seen.append(variation)
            if len(seen) > rounds:
                raise RuntimeError("out of GPU time in this test")
            headers = {"X-Retake-Similarity": "0.8000", "X-Retake-Detail-Gate": "passed",
                       "X-Retake-Detail-Score": "0.9000"}
            if exact is not None:
                headers["X-Retake-Self-Fidelity"] = "ssim:0.9000,deltaE:2.000" if variation == 7 else "ssim:0.7000,deltaE:6.000"
            return f"photo-{variation}".encode(), headers

        metadata = {"variation": 0, "referenceWeights": [1] * 25,
                    "faceBounds": {"x": 10, "y": 10, "width": 100, "height": 100}}
        if exact is not None:
            metadata["exactReferenceIndex"] = exact
        request = (b"t", [bytes([i]) for i in range(25)], ".png", [".png"] * 25, b"a", ".png", metadata)
        caches = {"calibration": calibration}
        if fail_schedule:
            class Broken:
                def schedule(self, *_a):
                    raise RuntimeError("disk error")
                def record(self, *_a, **_k):
                    raise RuntimeError("disk error")
            caches = {"calibration": Broken()}
        with patch.object(srv, "process_swap", fake_swap):
            # Each fake round takes 200 s of a 420 s window: only a few fit.
            clock = iter(range(0, 100000, 200))
            photo, headers = srv.run_ultra_search(Path("."), "1024x1024", False, request, lambda *_a, **_k: None,
                                                  caches, clock=lambda: next(clock), sleeper=lambda _s: None,
                                                  min_duration_seconds=0)
        return seen, photo, headers

    def test_default_then_learned_order_from_self_tests(self):
        with tempfile.TemporaryDirectory() as folder:
            store = uc.CalibrationStore(Path(folder) / "c.json", signature="s")
            seen, _photo, headers = self.run_search(store, exact=3)
            self.assertEqual(seen[0], 0)
            self.assertEqual(headers["X-Retake-Ultra-Schedule"], "default")
            # Teach it directly that variant 7 reproduces the person best.
            from tools import facefusion_local_server as srv
            bank = uc.bank_key([bytes([i]) for i in range(25)], [1] * 25)
            for _ in range(3):
                store.record(bank, [(v, {"X-Retake-Self-Fidelity": "ssim:0.6000,deltaE:8.000"}) for v in range(24) if v != 7]
                             + [(7, {"X-Retake-Self-Fidelity": "ssim:0.9500,deltaE:1.000"})], self_test=True)
            seen, _photo, headers = self.run_search(store, exact=3)
            self.assertEqual(seen[0], 7)
            self.assertEqual(headers["X-Retake-Ultra-Schedule"], "learned-fidelity")
            self.assertEqual(headers["X-Retake-Ultra-Order"].split(",")[0], "7")
            # The winner header still names the variant, not the chronology.
            self.assertEqual(headers["X-Retake-Ultra-Winner"], "8")
            summary = store.summary(bank)["banks"][bank]
            self.assertGreaterEqual(summary["self_tests"], 5)

    def test_learning_failures_never_break_a_job(self):
        seen, photo, headers = self.run_search(None, fail_schedule=True)
        self.assertEqual(seen[0], 0)
        self.assertEqual(headers["X-Retake-Ultra-Schedule"], "default")
        self.assertTrue(photo.startswith(b"photo-"))


class SelfTestModeTest(unittest.TestCase):
    def test_leave_one_out_pools_never_contain_the_target(self):
        from tools.identity_selection import strict_self_test_pools
        pools = [("all-25", (0, 1, 1, 2, 3)), ("independent-24", (0, 1, 1, 3)), ("consensus", (2,)), ("pose", (1, 3, 0))]
        strict = strict_self_test_pools(pools, 2)
        self.assertTrue(all(2 not in pool for _, pool in strict))
        self.assertEqual([label for label, _ in strict], ["all-25", "pose"])

    def test_self_test_flag_requires_an_exact_reference(self):
        from tools import facefusion_local_server as srv
        base = {"referenceWeights": [1] * 25, "faceBounds": {"x": 10, "y": 10, "width": 100, "height": 100},
                "targetSize": {"width": 784, "height": 1024}}
        self.assertFalse(srv._validate_metadata(base, 25)["selfTest"])
        self.assertTrue(srv._validate_metadata({**base, "exactReferenceIndex": 4, "selfTest": True}, 25)["selfTest"])
        for bad in ({"selfTest": True}, {"selfTest": "yes", "exactReferenceIndex": 4}):
            with self.assertRaises(ValueError):
                srv._validate_metadata({**base, **bad}, 25)


class GridObservationIntegrityTest(unittest.TestCase):
    def test_repeated_trials_and_invalid_metrics_do_not_inflate_learning(self):
        with tempfile.TemporaryDirectory() as folder:
            store=uc.CalibrationStore(Path(folder)/"c.json",signature="test")
            good={**proxy(.8),"X-Retake-Similarity":".82","X-Retake-Detail-Gate":"passed"}
            invalid={**good,"X-Retake-Similarity":"nan"}
            bad_gate={**good,"X-Retake-Detail-Gate":"failed"}
            self.assertEqual(store.record("bank",[(0,good),(0,proxy(.99)),(1,invalid),
                (2,bad_gate),(24,good),(3,good)]),2)
            stats=store.summary("bank")["banks"]["bank"]["variants"]
            self.assertEqual(set(stats),{"0","3"})
            self.assertEqual(stats["0"]["proxy"]["n"],1)
            self.assertAlmostEqual(stats["0"]["proxy"]["mean"],.8)
