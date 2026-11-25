"""Quality depth (v23): longer search, refinement and realism finalists.

No GPU needed: the real Ultra orchestration runs with a mocked swap.
"""
from __future__ import annotations

import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import facefusion_local_server as srv
from tools.retake_cache import PassCache
from tools.ultra_mode import (
    REFINE_DETAIL_RANGE, REFINE_HEAD_RANGE, ULTRA_DEFAULT_DEPTH, ULTRA_DEPTHS,
    ULTRA_SEARCH_BUDGET_SECONDS, ULTRA_VARIATIONS, RefinePlanner, depth_summary,
    finalist_specs, normalize_refinement, ultra_depth, with_refinement
)


def request(depth=None):
    metadata = {
        "variation": 0, "referenceWeights": [1] * 25,
        "faceBounds": {"x": 10, "y": 15, "width": 200, "height": 200},
    }
    if depth is not None:
        metadata["qualityDepth"] = depth
    return (b"original photo", [b"reference"] * 25, ".png", [".png"] * 25,
            b"original appearance", ".png", metadata)


def score(value, realism=False):
    return {
        "X-Retake-Similarity": str(value),
        "X-Retake-Realism": "on" if realism else "off",
        "X-Retake-Texture-Source": "average",
        "X-Retake-Detail-Gate": "passed",
    }


class DepthSettingsTest(unittest.TestCase):
    def test_default_depth_is_the_seven_minute_contract(self):
        self.assertEqual(ULTRA_DEFAULT_DEPTH, "standard")
        standard = ultra_depth(None)
        self.assertEqual(standard.search_seconds, ULTRA_SEARCH_BUDGET_SECONDS)
        self.assertEqual((standard.refinements, standard.realism_finalists), (0, 1))
        self.assertEqual(ultra_depth("unknown").name, "standard")

    def test_deeper_levels_only_buy_more_candidates(self):
        standard, deep, best = (ULTRA_DEPTHS[name] for name in ("standard", "deep", "max"))
        for field in ("search_seconds", "extra_finalists", "realism_finalists", "refinements", "client_timeout_seconds"):
            self.assertLessEqual(getattr(standard, field), getattr(deep, field), field)
            self.assertLessEqual(getattr(deep, field), getattr(best, field), field)
        self.assertGreater(deep.search_seconds, standard.search_seconds)
        summary = depth_summary()
        self.assertEqual(set(summary), {"standard", "deep", "max"})
        self.assertEqual(summary["deep"]["refinements"], deep.refinements)
        # The browser must wait longer than the longest search it asked for.
        for settings in ULTRA_DEPTHS.values():
            self.assertGreater(settings.client_timeout_seconds, settings.search_seconds + 600)

    def test_metadata_accepts_only_known_depths(self):
        base = {"referenceWeights": [1] * 25, "faceBounds": {"x": 0, "y": 0, "width": 100, "height": 100}}
        self.assertEqual(srv._validate_metadata(base, 25)["qualityDepth"], "standard")
        self.assertEqual(srv._validate_metadata({**base, "qualityDepth": "max"}, 25)["qualityDepth"], "max")
        for bad in ("ultra", "", 3, True, None):
            with self.assertRaises(ValueError):
                srv._validate_metadata({**base, "qualityDepth": bad}, 25)

    def test_browser_cannot_inject_engine_only_keys(self):
        base = {"referenceWeights": [1] * 25, "faceBounds": {"x": 0, "y": 0, "width": 100, "height": 100},
                "variationParams": [1.0, 0.4, 1.5], "realismSeedBase": 9}
        validated = srv._validate_metadata(base, 25)
        self.assertNotIn("variationParams", validated)
        self.assertNotIn("realismSeedBase", validated)


class RefinementTest(unittest.TestCase):
    def test_normalize_rejects_values_outside_the_box(self):
        self.assertEqual(normalize_refinement((1, 0.2, 1)), (1.0, 0.2, 1.0))
        for bad in (None, (1.0, 0.2), (1.0, 0.5, 1.0), (1.0, 0.2, 2.0), (1.2, 0.2, 1.0),
                    (1.0, float("nan"), 1.0), (True, 0.2, 1.0), "1,0.2,1"):
            self.assertIsNone(normalize_refinement(bad), bad)

    def test_compass_search_keeps_a_winning_direction(self):
        planner = RefinePlanner((1.0, 0.20, 1.00), budget=4)
        first = planner.next()
        self.assertEqual(first, (1.0, 0.26, 1.0))
        planner.feedback(first, True)
        second = planner.next()
        self.assertEqual(second, (1.0, 0.32, 1.0), "momentum: further the same way")
        planner.feedback(second, False)
        third = planner.next()
        # Back towards 0.20 is already measured, so the head-fit axis is next.
        self.assertEqual(third, (1.0, 0.26, 1.3))
        planner.feedback(third, False)
        self.assertEqual(planner.next(), (1.0, 0.26, 0.7))
        self.assertIsNone(planner.next(), "budget is respected")

    def test_measured_grid_points_are_not_repeated_and_step_halves(self):
        start = (0.95, 0.10, 0.00)
        neighbours = [(0.95, 0.16, 0.0), (0.95, 0.04, 0.0), (0.95, 0.10, 0.30)]
        planner = RefinePlanner(start, tried=neighbours, budget=3)
        seen = []
        while (params := planner.next()) is not None:
            seen.append(params)
            planner.feedback(params, False)
        self.assertNotIn((0.95, 0.16, 0.0), seen)
        self.assertNotIn((0.95, 0.10, 0.30), seen)
        # Head fit cannot go below 0, so only half steps remain.
        self.assertIn((0.95, 0.13, 0.0), seen)
        for weight, detail, head in seen:
            self.assertEqual(weight, 0.95, "the swapper weight never changes")
            self.assertTrue(REFINE_DETAIL_RANGE[0] <= detail <= REFINE_DETAIL_RANGE[1])
            self.assertTrue(REFINE_HEAD_RANGE[0] <= head <= REFINE_HEAD_RANGE[1])

    def test_request_carries_settings_without_touching_the_original(self):
        parsed = request()
        refined = with_refinement(parsed, 7, (1.0, 0.22, 1.1), realism_seed=107)
        self.assertEqual(refined[0], b"original photo")
        self.assertEqual(refined[6]["variation"], 7)
        self.assertEqual(refined[6]["variationParams"], (1.0, 0.22, 1.1))
        self.assertEqual(refined[6]["realismSeedBase"], 107)
        self.assertNotIn("variationParams", parsed[6])
        with self.assertRaises(ValueError):
            with_refinement(parsed, 7, (1.0, 0.9, 1.1))
        with self.assertRaises(ValueError):
            with_refinement(parsed, 7, None, realism_seed=-1)

    def test_finalists_start_with_the_winner_and_stay_distinct(self):
        # Only identity near-ties of the peak qualify (as for any selection).
        records = [((0, None), score(.8440)), ((3, None), score(.80)), ((5, None), score(.8435)),
                   ((0, (1.0, .26, 1.0)), score(.845)), ((5, None), score(.8435))]
        specs = finalist_specs(records, (0, (1.0, .26, 1.0)), .845, 3)
        self.assertEqual(specs[0], (0, (1.0, .26, 1.0)))
        self.assertEqual(specs[1:], [(0, None), (5, None)])
        self.assertNotIn((3, None), finalist_specs(records, (0, None), .845, 4), "not an identity near-tie")
        self.assertEqual(finalist_specs(records, (0, None), .845, 1), [(0, None)])
        self.assertEqual(finalist_specs(records, (0, None), .845, 0), [])


class DeepSearchTest(unittest.TestCase):
    def run_search(self, depth, fake_swap, caches=None, clock=None):
        phases = []
        with patch.object(srv, "process_swap", fake_swap):
            photo, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, request(depth), lambda *_a, **_k: None,
                caches=caches, clock=clock, min_duration_seconds=0,
                on_phase=lambda *item: phases.append(item)
            )
        return photo, headers, phases

    def test_deep_search_runs_past_the_seven_minute_window(self):
        now = [0.0]
        visited = []

        def fake_swap(*args):
            now[0] += 125.0
            visited.append(args[3][6]["variation"])
            return b"best", score(.83)

        photo, headers, _ = self.run_search("deep", fake_swap, clock=lambda: now[0])
        rounds = int(headers["X-Retake-Ultra-Rounds"])
        self.assertGreater(rounds, 4, "standard stops after 4 such rounds")
        self.assertLessEqual(rounds * 125.0, ULTRA_DEPTHS["deep"].search_seconds + 125.0)
        self.assertEqual(headers["X-Retake-Ultra-Depth"], "deep")
        self.assertEqual(headers["X-Retake-Ultra-Search-Budget"], "1500")

    def test_refinement_keeps_a_better_step_and_reuses_the_weight(self):
        calls = []

        def fake_swap(*args):
            metadata = args[3][6]
            params = metadata.get("variationParams")
            calls.append((metadata["variation"], params))
            if params is None:
                return f"grid-{metadata['variation']}".encode(), score(.84 if metadata["variation"] == 0 else .80)
            # Slightly more detail transfer measures better; nothing else does.
            better = params[1] > srv.FACE_SWAP_VARIATIONS[0][1]
            headers = {**score(.851 if better else .83), "X-Retake-Variation-Params": ",".join(map(str, params))}
            return ("refined-%s" % params[1]).encode(), headers

        photo, headers, phases = self.run_search("deep", fake_swap)
        refinements = [item for item in calls if item[1] is not None]
        self.assertEqual(len(refinements), ULTRA_DEPTHS["deep"].refinements)
        self.assertTrue(all(variation == 0 for variation, _ in refinements), "around the winner")
        self.assertTrue(all(params[0] == srv.FACE_SWAP_VARIATIONS[0][0] for _, params in refinements))
        self.assertEqual(photo, b"refined-0.26")
        wins, attempts = headers["X-Retake-Ultra-Refinements"].split("/")
        self.assertEqual(attempts, str(ULTRA_DEPTHS["deep"].refinements))
        self.assertGreaterEqual(int(wins), 1)
        self.assertEqual(headers["X-Retake-Ultra-Winner"], "1")
        self.assertIn(("refine", 1, ULTRA_DEPTHS["deep"].refinements), phases)

    def test_a_worse_refinement_cannot_replace_the_grid_winner(self):
        def fake_swap(*args):
            metadata = args[3][6]
            if metadata.get("variationParams") is not None:
                return b"refined", score(.70)
            return b"grid", score(.84 if metadata["variation"] == 0 else .80)

        photo, headers, _ = self.run_search("max", fake_swap)
        self.assertEqual(photo, b"grid")
        self.assertTrue(headers["X-Retake-Ultra-Refinements"].startswith("0/"))

    def test_deep_gives_two_distinct_finalists_the_realism_pass(self):
        realism_calls = []

        def fake_swap(*args):
            metadata, caches = args[3][6], args[5]
            if caches and caches.get("realism"):
                realism_calls.append((metadata["variation"], metadata.get("variationParams"), metadata.get("realismSeedBase")))
                if metadata["variation"] == 5:
                    return b"second-finalist-real", score(.845, realism=True)
                return b"first-finalist-real", score(.70, realism=True)
            if metadata.get("variationParams") is not None:
                return b"refined", score(.80)
            return f"grid-{metadata['variation']}".encode(), score({0: .8400, 5: .8395}.get(metadata["variation"], .79))

        photo, headers, phases = self.run_search("deep", fake_swap, caches={"realism": {"python": Path("gpu")}})
        self.assertEqual(len(realism_calls), 2)
        self.assertEqual(realism_calls[0], (0, None, None), "the winner keeps the default seeds")
        self.assertEqual(realism_calls[1][0], 5)
        self.assertEqual(realism_calls[1][2], srv.REALISM_SEED_BASE + srv.REALISM_SEED_STRIDE)
        self.assertEqual(photo, b"second-finalist-real")
        self.assertEqual(headers["X-Retake-Ultra-Realism-Finalists"], "2")
        self.assertEqual(headers["X-Retake-Ultra-Winner"], "6")
        self.assertIn(("realism", 2, 2), phases)

    def test_extra_finalists_measure_new_variants_without_an_editor(self):
        now = [0.0]
        calls = []

        def fake_swap(*args):
            now[0] += 250.0  # only two search rounds fit in 420 s
            calls.append(args[3][6]["variation"])
            return b"x", {**score(.80), "X-Retake-Identity-Median-Similarity": "0.80"}

        _, headers, _ = self.run_search(None, fake_swap, clock=lambda: now[0])
        self.assertEqual(calls[:2], [0, 1])
        extras = calls[2:]
        self.assertTrue(extras, "an under-target result gets extra finalists")
        self.assertFalse(set(extras) & {0, 1}, "a measured variant is not repeated")
        self.assertEqual(len(extras), len(set(extras)))

    def test_standard_depth_is_unchanged(self):
        calls = []

        def fake_swap(*args):
            metadata, caches = args[3][6], args[5]
            calls.append((metadata["variation"], metadata.get("variationParams"), bool(caches and caches.get("realism"))))
            return b"x", score(.84 if metadata["variation"] == 3 else .74)

        _, headers, _ = self.run_search(None, fake_swap, caches={"realism": {"python": Path("gpu")}})
        self.assertEqual(calls[:24], [(variation, None, False) for variation in ULTRA_VARIATIONS])
        self.assertEqual(calls[24:], [(3, None, True)])
        self.assertEqual(headers["X-Retake-Ultra-Depth"], "standard")
        self.assertEqual(headers["X-Retake-Ultra-Refinements"], "0/0")


class JobAndCacheTest(unittest.TestCase):
    def tearDown(self):
        with srv.JOBS_LOCK:
            srv.JOBS.clear()

    def test_job_remembers_depth_and_search_budget(self):
        job = srv.create_job(False, "max")
        self.assertEqual(job["depth"], "max")
        self.assertEqual(job["search_budget_seconds"], ULTRA_DEPTHS["max"].search_seconds)
        self.assertEqual(srv.create_job(False, "bogus")["depth"], "standard")

    def test_eta_counts_the_job_search_window(self):
        class Timings:
            def remaining(self, pending, _index, _in_current):
                return 60.0 * len(pending)

        job = srv.create_job(False, "deep")
        job.update(status="running", step=0, stage_started=time.time())
        job["created"] = time.time() - 600
        eta = srv.estimate_remaining(job, Timings())
        # 1500 s window − 600 s elapsed + the current stage, not the 420 s window.
        self.assertGreater(eta, 800)
        self.assertLessEqual(eta, 1500 - 600 + 61)

    def test_pass_cache_holds_a_whole_job_within_a_byte_budget(self):
        cache = PassCache(max_entries=100, max_bytes=10)
        cache.put("a", b"1234", "1024x1024")
        cache.put("b", b"5678", "1024x1024")
        cache.put("c", b"90", "1024x1024")
        self.assertEqual(len(cache), 3)
        cache.put("d", b"abcd", "1024x1024")  # 14 bytes > 10: oldest goes
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("d"), (b"abcd", "1024x1024"))
        cache.put("huge", b"x" * 11, "1024x1024")
        self.assertIsNone(cache.get("huge"), "larger than the whole budget")
        info = cache.describe()
        self.assertLessEqual(info["entries"], 3)
        self.assertGreaterEqual(info["hits"], 1)
        cache.clear()
        self.assertEqual(cache.describe()["last_job"]["hits"], info["hits"])
        self.assertEqual(len(cache), 0)

    def test_default_cache_survives_three_swapper_weights(self):
        cache = PassCache()
        for weight in (1.0, 0.95, 0.90):
            for index in range(9):
                cache.put(PassCache.key(b"t", [bytes([index])], "m", weight, "1024x1024"), b"png" * 1000, "1024x1024")
        for weight in (1.0, 0.95, 0.90):
            self.assertIsNotNone(cache.get(PassCache.key(b"t", [bytes([0])], "m", weight, "1024x1024")))

    def test_cache_size_from_environment_is_bounded(self):
        self.assertEqual(PassCache.from_environment({"RETAKE_PASS_CACHE_MB": "512"}).max_bytes, 512 * 1024 * 1024)
        default = PassCache().max_bytes
        for bad in ("8", "999999", "lots", ""):
            self.assertEqual(PassCache.from_environment({"RETAKE_PASS_CACHE_MB": bad}).max_bytes, default)

    def test_another_finalist_passes_its_seed_to_the_realism_worker(self):
        seen = {}

        class Completed:
            stdout = 'RETAKE_REALISM_REPORT={"ok": true, "outputs": []}'
            stderr = ""
            returncode = 0

        def fake_run(command, **_kwargs):
            seen["command"] = command
            return Completed()

        import tempfile
        with tempfile.TemporaryDirectory() as directory, patch.object(srv.subprocess, "run", fake_run):
            temp = Path(directory)
            srv.run_realism({"python": Path("py"), "models": Path("m")}, temp / "a.png", temp / "b.png",
                            {"x": 0, "y": 0, "width": 10, "height": 10}, [1.0], temp, seed_base=107)
            self.assertIn("--seed-base", seen["command"])
            self.assertEqual(seen["command"][seen["command"].index("--seed-base") + 1], "107")
            srv.run_realism({"python": Path("py"), "models": Path("m")}, temp / "a.png", temp / "b.png",
                            {"x": 0, "y": 0, "width": 10, "height": 10}, [1.0], temp)
            self.assertNotIn("--seed-base", seen["command"])


if __name__ == "__main__":
    unittest.main()
