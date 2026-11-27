"""Ultra-mode scheduling and best-result integrity tests.

No GPU needed: evaluate the real HTTP orchestration with mocked FaceFusion.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from tools import facefusion_local_server as srv
from tools.ultra_mode import (
    ULTRA_VARIATIONS, prefer_ultra_candidate, rim_from_headers, score_from_headers,
    should_continue_ultra, with_variation
)


def fake_request():
    return (
        b"original photo", [b"reference"] * 25, ".png", [".png"] * 25,
        b"original appearance", ".png", {
            "variation": 3, "referenceWeights": [1] * 25,
            "faceBounds": {"x": 10, "y": 15, "width": 200, "height": 200}
        }
    )


def score(score_value, realism=False, texture="average"):
    return {
        "X-Retake-Similarity": str(score_value),
        "X-Retake-Realism": "on" if realism else "off",
        "X-Retake-Texture-Source": texture,
        "X-Retake-Compositing": "preblended",
    }


class UltraSelectionTest(unittest.TestCase):
    def test_first_pass_is_always_variation_zero_on_untouched_original(self):
        parsed = fake_request()
        variant = with_variation(parsed, 0)
        self.assertEqual(variant[0], b"original photo")
        self.assertEqual(variant[1], parsed[1])
        self.assertEqual(variant[6]["variation"], 0)
        self.assertEqual(parsed[6]["variation"], 3)

    def test_invalid_cosine_never_displaces_verified_result(self):
        best = score("0.802")
        for invalid in ["nan", "inf", "1.2", "", "-1.5"]:
            self.assertIsNone(score_from_headers(score(invalid)))
            self.assertFalse(prefer_ultra_candidate(score(invalid), best))

    def test_favors_significant_identity_gain_and_natural_tiebreaks(self):
        best = score("0.800")
        self.assertTrue(prefer_ultra_candidate(best, None))
        self.assertTrue(prefer_ultra_candidate(score("0.811"), best))
        self.assertTrue(prefer_ultra_candidate(score("0.803"), best), "measurable identity improvement must not be discarded")
        self.assertTrue(prefer_ultra_candidate(score("0.799", realism=True), best))
        self.assertFalse(prefer_ultra_candidate(score("0.790", realism=True), best))
        self.assertFalse(prefer_ultra_candidate(score("0.800"), score("0.801", realism=True)))

    def test_four_rounds_pick_best_verified_face_not_necessarily_last(self):
        visits = []
        results = [score(0.68), score(0.72), score(0.81, realism=True), score(0.76)]

        def fake_swap(*args):
            parsed, progress = args[3], args[4]
            variant = parsed[6]["variation"]
            visits.append((parsed[0], variant))
            for i in range(len(srv.JOB_STAGES)):
                progress(i)
            return f"photo-{variant}".encode(), results[variant % len(results)]

        positions = []
        with patch.object(srv, "process_swap", fake_swap):
            photo, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, fake_request(),
                lambda step, cached=False: positions.append(step),
                min_duration_seconds=0
            )
        self.assertEqual(photo, b"photo-2")
        self.assertEqual(headers["X-Retake-Ultra"], "on")
        self.assertEqual(headers["X-Retake-Ultra-Rounds"], "24")
        self.assertEqual(headers["X-Retake-Ultra-Winner"], "3")
        self.assertEqual(positions, list(range(len(srv.JOB_STAGES) * len(ULTRA_VARIATIONS))))
        self.assertEqual(visits, [(b"original photo", index) for index in ULTRA_VARIATIONS])


    def test_near_ties_cannot_accumulate_identity_loss_across_rounds(self):
        def fake_swap(*args):
            index = args[3][6]["variation"]
            return f"photo-{index}".encode(), {
                **score(.900 - index * .0015),
                "X-Retake-Rim-Fraction": str(max(0.0, .9 - index * .04))
            }
        with patch.object(srv, "process_swap", fake_swap):
            photo, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, fake_request(),
                lambda *_a, **_k: None, min_duration_seconds=0
            )
        self.assertGreaterEqual(score_from_headers(headers), .898)
        self.assertEqual(float(headers["X-Retake-Ultra-Peak-Identity"]), .9)
        self.assertEqual(headers["X-Retake-Ultra-Attempted"], "24")

    def test_invalid_scores_are_attempted_but_not_verified(self):
        def fake_swap(*args):
            index = args[3][6]["variation"]
            return b"verified" if index == 1 else b"invalid", score(.84 if index == 1 else "nan")
        with patch.object(srv, "process_swap", fake_swap):
            photo, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, fake_request(),
                lambda *_a, **_k: None, min_duration_seconds=0
            )
        self.assertEqual(photo, b"verified")
        self.assertEqual(headers["X-Retake-Ultra-Rounds"], "1")
        self.assertEqual(headers["X-Retake-Ultra-Attempted"], "24")

    def test_diffusion_is_run_only_for_best_identity_finalist(self):
        calls = []
        finalist_calls = []
        realism_config = {"python": Path("gpu"), "models": Path("weights")}

        def fake_swap(*args):
            variation = args[3][6]["variation"]
            active = bool(args[5] and args[5].get("realism"))
            calls.append((variation, active))
            if active:
                return b"realism", score(0.85, realism=True)
            return f"unfiltered-{variation}".encode(), score(0.84 if variation == 3 else 0.74)

        with patch.object(srv, "process_swap", fake_swap):
            photo, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, fake_request(),
                lambda *_args, **_kwargs: None,
                caches={"realism": realism_config},
                on_finalist=lambda: finalist_calls.append("gpu"),
                min_duration_seconds=0
            )
        self.assertEqual(calls[:24], [(variation, False) for variation in ULTRA_VARIATIONS])
        self.assertEqual(calls[24:], [(3, True)])
        self.assertEqual(photo, b"realism")
        self.assertEqual(finalist_calls, ["gpu"])
        self.assertEqual(headers["X-Retake-Ultra-Realism-Policy"], "finalist-only")

    def test_bad_finalist_diffusion_cannot_overwrite_sharp_identity_result(self):
        def fake_swap(*args):
            if args[5] and args[5].get("realism"):
                return b"fake-face", score(0.65, realism=True)
            return b"strong-identity", score(0.84)
        with patch.object(srv, "process_swap", fake_swap):
            photo, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, fake_request(),
                lambda *_args, **_kwargs: None,
                caches={"realism": {"python": Path("gpu")}},
                min_duration_seconds=0
            )
        self.assertEqual(photo, b"strong-identity")
        self.assertEqual(headers["X-Retake-Ultra-Rounds"], "24")

    def test_budget_stops_extra_rounds_without_sleep_or_fake_wait(self):
        now = [0.0]

        def clock():
            return now[0]

        def fake_swap(*args):
            now[0] += 125.0
            return b"best", score(0.83)

        with patch.object(srv, "process_swap", fake_swap):
            photo, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, fake_request(),
                lambda *_args, **_kwargs: None, clock=clock,
                min_duration_seconds=0
            )
        self.assertEqual(photo, b"best")
        self.assertEqual(headers["X-Retake-Ultra-Rounds"], "4")
        self.assertEqual(headers["X-Retake-Ultra-Elapsed"], "500.0")
        self.assertFalse(should_continue_ultra(4, 500))

    def test_failed_later_round_preserves_best_verified_result(self):
        attempts = []

        def fake_swap(*args):
            variation = args[3][6]["variation"]
            attempts.append(variation)
            if variation == 0:
                return b"good", score(0.85)
            raise srv.NoOpSwapError("bu varyasyonda yüz değişmedi")

        with patch.object(srv, "process_swap", fake_swap):
            photo, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, fake_request(),
                lambda *_args, **_kwargs: None,
                min_duration_seconds=0
            )
        self.assertEqual(photo, b"good")
        self.assertEqual(attempts, list(ULTRA_VARIATIONS))
        self.assertEqual(headers["X-Retake-Ultra-Winner"], "1")

    def test_seven_minimum_is_real_elapsed_time_and_does_not_reprocess_pixels(self):
        now = [0.0]
        holds = []
        seen = []

        def clock():
            return now[0]

        def advance(seconds):
            now[0] += seconds

        def fake_swap(*args):
            seen.append(args[3][6]["variation"])
            # The first GPU round costs 360s, second costs 80s.
            now[0] += 360 if len(seen) == 1 else 80
            return b"validated", score(0.86)

        with patch.object(srv, "process_swap", fake_swap):
            photo, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, fake_request(),
                lambda *_args, **_kwargs: None, clock=clock,
                sleeper=advance, on_hold=holds.append,
                min_duration_seconds=420
            )
        self.assertEqual(photo, b"validated")
        self.assertEqual(seen, [0, 1])
        self.assertEqual(headers["X-Retake-Ultra-Elapsed"], "440.0")
        self.assertEqual(holds[-1], 0.0)

    def test_finishing_all_variants_early_uses_visible_minimum_hold(self):
        now = [0.0]
        holds = []
        attempts = []

        def fake_swap(*args):
            now[0] += 2
            attempts.append(args[3][6]["variation"])
            return b"good", score(0.9)

        with patch.object(srv, "process_swap", fake_swap):
            photo, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, fake_request(),
                lambda *_args, **_kwargs: None,
                clock=lambda: now[0],
                sleeper=lambda delay: now.__setitem__(0, now[0] + delay),
                on_hold=holds.append,
                min_duration_seconds=420
            )
        self.assertEqual(len(attempts), 24)
        self.assertEqual(photo, b"good")
        self.assertEqual(headers["X-Retake-Ultra-Elapsed"], "420.0")
        self.assertGreater(len(holds), 1)
        self.assertGreater(holds[0], 0)
        self.assertEqual(holds[-1], 0.0)

    def test_detail_score_decides_ties_without_overriding_identity(self):
        a = {**score(.830), "X-Retake-Detail-Score": ".44"}
        b = {**score(.829), "X-Retake-Detail-Score": ".88"}
        c = {**score(.821), "X-Retake-Detail-Score": ".95"}
        self.assertTrue(prefer_ultra_candidate(b, a))
        self.assertFalse(prefer_ultra_candidate(c, a))

    def test_rejects_unverifiable_or_invalid_residual_halo_scores(self):
        for invalid in ("", "nan", "inf", "-0.1", "1.1", "bogus"):
            self.assertIsNone(rim_from_headers({"X-Retake-Rim-Fraction": invalid}))
        self.assertIsNone(rim_from_headers({}))
        self.assertAlmostEqual(rim_from_headers({"X-Retake-Rim-Fraction": ".19"}), .19)

    def test_visible_black_outline_breaks_only_identity_near_ties(self):
        dark = {**score(.838), "X-Retake-Rim-Fraction": ".19", "X-Retake-Detail-Score": ".85"}
        clean = {**score(.837), "X-Retake-Rim-Fraction": ".04", "X-Retake-Detail-Score": ".70"}
        worse = {**score(.821), "X-Retake-Rim-Fraction": ".00"}
        self.assertTrue(prefer_ultra_candidate(clean, dark), "visible seam improvement wins a narrow ArcFace tie")
        self.assertFalse(prefer_ultra_candidate(dark, clean), "same identity tie must reject darker outline")
        self.assertFalse(prefer_ultra_candidate(worse, dark), "a cleaner face must not replace a different identity")
        self.assertTrue(prefer_ultra_candidate(dark, worse), "meaningful identity gain stays first priority")

    def test_missing_halo_score_never_counts_as_perfect(self):
        dark = {**score(.83), "X-Retake-Rim-Fraction": ".19"}
        missing = score(.83)
        invalid = {**score(.83), "X-Retake-Rim-Fraction": "nan"}
        self.assertFalse(prefer_ultra_candidate(missing, dark))
        self.assertFalse(prefer_ultra_candidate(invalid, dark))
        self.assertFalse(prefer_ultra_candidate(dark, missing))

    def test_small_halo_difference_defers_to_existing_detail_tie_break(self):
        old = {**score(.83), "X-Retake-Rim-Fraction": ".10", "X-Retake-Detail-Score": ".40"}
        new = {**score(.83), "X-Retake-Rim-Fraction": ".09", "X-Retake-Detail-Score": ".88"}
        self.assertTrue(prefer_ultra_candidate(new, old))

    def test_every_failed_round_preserves_original_error_kind(self):
        with patch.object(srv, "process_swap", side_effect=srv.NoOpSwapError("no face")):
            with self.assertRaises(srv.NoOpSwapError):
                srv.run_ultra_search(
                    Path("."), "1024x1024", False, fake_request(),
                    lambda *_args, **_kwargs: None
                )


if __name__ == "__main__":
    unittest.main()


class UltraNaturalnessTest(unittest.TestCase):
    @staticmethod
    def headers(identity, naturalness=None):
        value = {"X-Retake-Similarity": f"{identity:.4f}"}
        if naturalness is not None:
            value["X-Retake-Naturalness"] = f"{naturalness:.1f}"
        return value

    def test_a_clearly_more_natural_photo_wins_a_small_identity_gap(self):
        from tools.ultra_mode import prefer_ultra_candidate
        natural = self.headers(0.795, 90)
        pasted = self.headers(0.800, 78)
        self.assertTrue(prefer_ultra_candidate(natural, pasted))
        self.assertFalse(prefer_ultra_candidate(pasted, natural))

    def test_below_the_floor_loses_to_a_natural_one_nearly_as_close(self):
        from tools.ultra_mode import prefer_ultra_candidate
        self.assertTrue(prefer_ultra_candidate(self.headers(0.78, 80), self.headers(0.80, 55)))
        self.assertFalse(prefer_ultra_candidate(self.headers(0.80, 55), self.headers(0.78, 80)))

    def test_identity_still_decides_a_real_gap(self):
        from tools.ultra_mode import prefer_ultra_candidate
        # 0.05 ArcFace is beyond every naturalness allowance.
        self.assertTrue(prefer_ultra_candidate(self.headers(0.85, 70), self.headers(0.80, 92)))
        self.assertFalse(prefer_ultra_candidate(self.headers(0.80, 92), self.headers(0.85, 70)))

    def test_without_naturalness_the_old_rules_apply(self):
        from tools.ultra_mode import prefer_ultra_candidate
        self.assertTrue(prefer_ultra_candidate(self.headers(0.81), self.headers(0.80)))


class FluxOutcomeTest(unittest.TestCase):
    def test_rejected_finalist_keeps_original_and_reports_real_outcome(self):
        def fake_swap(*args):
            if args[5] and args[5].get("nim_editors"):
                return b"rejected", {**score(.61), "X-Retake-FLUX-Status":"accepted"}
            return b"verified", score(.87)
        with patch.object(srv, "process_swap", fake_swap):
            image, headers = srv.run_ultra_search(
                Path("."), "1024x1024", False, fake_request(),
                lambda *_a, **_k: None, min_duration_seconds=0,
                caches={"nim_editors":{"retake-local/flux2-klein-4b":0}}
            )
        self.assertEqual(image,b"verified")
        self.assertEqual(headers["X-Retake-FLUX-Status"],"finalist-not-selected")


class AdaptiveFinalistsTest(unittest.TestCase):
    def test_profile_target_is_adaptive_and_unknown_is_not_invented(self):
        from tools.ultra_mode import identity_aspiration, identity_gain, needs_extra_finalist, choose_extra_variations
        headers = {**score(.796), "X-Retake-Identity-Median-Similarity": ".77", "X-Retake-Similarity-Before": ".06"}
        self.assertAlmostEqual(identity_aspiration(headers), .835)
        self.assertAlmostEqual(identity_gain(headers), .736)
        self.assertTrue(needs_extra_finalist(headers))
        self.assertIsNone(identity_aspiration(score(.8)))
        self.assertEqual(choose_extra_variations({0:.8,1:.79,3:.78},1),[3,1])
        self.assertEqual(choose_extra_variations({0:.8},1),[3,6])
        self.assertAlmostEqual(identity_aspiration({**headers,"X-Retake-Similarity-Before":".95"}),.95)

    def test_under_target_runs_extra_finalist_and_keeps_verified_improvement(self):
        calls = []
        def fake_swap(*args):
            variation = args[3][6]["variation"]
            editing = bool((args[5] or {}).get("nim_editors"))
            calls.append((variation,editing))
            value = .86 if editing and variation==3 else .796 if variation==0 else .77
            return str((variation,editing)).encode(), {
                **score(value),
                "X-Retake-Detail-Gate":"passed",
                "X-Retake-Identity-Median-Similarity":".77",
                "X-Retake-Similarity-Before":".06",
                "X-Retake-FLUX-Status":"accepted" if editing and variation==3 else "detail-gain-insufficient",
            }
        with patch.object(srv,"process_swap",fake_swap):
            image, headers = srv.run_ultra_search(
                Path("."),"1024x1024",False,fake_request(),lambda *_a,**_k:None,
                caches={"nim_editors":{"retake-local/flux2-klein-4b":0}}, min_duration_seconds=0)
        self.assertEqual(image,b"(3, True)")
        self.assertEqual(headers["X-Retake-Finalist-Attempts"],"2")
        self.assertEqual(headers["X-Retake-Ultra-Winner"],"4")
        self.assertEqual(headers["X-Retake-FLUX-Status"],"accepted")
        self.assertEqual(headers["X-Retake-Identity-Gain"],".8000".rjust(6,"0"))
        self.assertEqual(sum(editing for _,editing in calls),2)


class FluxRetryBudgetTest(unittest.TestCase):
    def test_defer_only_when_both_metrics_are_measured_and_bad(self):
        from tools.ultra_mode import defer_flux_retry
        self.assertTrue(defer_flux_retry("identity-loss",{"X-Retake-FLUX-Identity-Delta":"-.076","X-Retake-FLUX-Detail-Delta":"-.11"}))
        for identity,detail in (("-.01","-.11"),("-.076",".05"),("nan","-.11"),("-.076","")):
            self.assertFalse(defer_flux_retry("identity-loss",{"X-Retake-FLUX-Identity-Delta":identity,"X-Retake-FLUX-Detail-Delta":detail}))
        self.assertFalse(defer_flux_retry("accepted",{"X-Retake-FLUX-Identity-Delta":"-.076","X-Retake-FLUX-Detail-Delta":"-.11"}))


class LateIdentityPeakTest(unittest.TestCase):
    def test_realism_peak_is_reported_even_if_sufficient_cleaner_photo_wins(self):
        level={"X-Retake-Identity-Photo-Level":".86","X-Retake-Detail-Gate":"passed"}
        def fake_swap(*args):
            if (args[5] or {}).get("realism"):
                return b"new", {**score(.88,realism=True),**level,"X-Retake-Rim-Fraction":".4"}
            return b"clean", {**score(.85),**level,"X-Retake-Rim-Fraction":".01"}
        with patch.object(srv,"process_swap",fake_swap):
            photo,headers=srv.run_ultra_search(Path("."),"1024x1024",False,fake_request(),
                lambda *_a,**_k:None,caches={"realism":{"python":Path("gpu")}},min_duration_seconds=0)
        self.assertEqual(photo,b"clean")
        self.assertAlmostEqual(float(headers["X-Retake-Ultra-Peak-Identity"]),.88)


class FinishingIdentityFloorTest(unittest.TestCase):
    def test_finish_steps_cannot_accumulate_loss_or_drop_below_reference_floor(self):
        from tools.ultra_mode import finishing_identity_floor
        self.assertAlmostEqual(finishing_identity_floor(.832,None),.828)
        self.assertAlmostEqual(finishing_identity_floor(.871,.89),.87)
        self.assertAlmostEqual(finishing_identity_floor(.88,.89),.876)
        floor=finishing_identity_floor(.832,.89)
        self.assertGreaterEqual(.829,floor)
        self.assertLess(.826,floor)


class DistinctExtraWorkTest(unittest.TestCase):
    def test_full_grid_local_retry_uses_new_refinement(self):
        from tools.ultra_mode import plan_extra_spec
        grid=srv.FACE_SWAP_VARIATIONS
        scores={i:.80 for i in ULTRA_VARIATIONS}
        records=[((i,None),score(.80)) for i in ULTRA_VARIATIONS]
        plan=plan_extra_spec(scores,(0,None),grid,records,set(),{(0,None)},
                             ULTRA_VARIATIONS,True,True)
        spec,local_only=plan
        self.assertTrue(local_only)
        self.assertEqual(spec[0],0)
        self.assertIsNotNone(spec[1])
        self.assertNotEqual(spec[1],grid[0])
        second=plan_extra_spec(scores,(0,None),grid,records,{spec},{(0,None)},
                               ULTRA_VARIATIONS,True,True)
        self.assertNotEqual(second[0],spec)

    def test_editor_retry_skips_previously_edited_spec(self):
        from tools.ultra_mode import plan_extra_spec
        grid=srv.FACE_SWAP_VARIATIONS
        scores={0:.82,3:.81,1:.80}
        records=[((i,None),score(s)) for i,s in scores.items()]
        spec,local_only=plan_extra_spec(scores,(0,None),grid,records,set(),{(0,None),(3,None)},
                                        ULTRA_VARIATIONS,False,True)
        self.assertEqual(spec,(1,None))
        self.assertFalse(local_only)

    def test_without_editor_uses_untried_grid_then_refines(self):
        from tools.ultra_mode import plan_extra_spec
        spec,local_only=plan_extra_spec({0:.8},(0,None),srv.FACE_SWAP_VARIATIONS,
            [((0,None),score(.8))],set(),set(),ULTRA_VARIATIONS,False,False)
        self.assertEqual(spec,(1,None))
        self.assertTrue(local_only)

    def test_severe_diffusion_failure_does_not_repeat_deterministic_grid(self):
        calls=[]
        def fake_swap(*args):
            metadata=args[3][6];editor=bool((args[5] or {}).get("nim_editors"))
            variation=metadata["variation"];params=metadata.get("variationParams")
            calls.append((variation,params,editor))
            value=.87 if params is not None else .80 if variation==0 else .77
            return repr((variation,params,editor)).encode(), {
                **score(value),"X-Retake-Detail-Gate":"passed",
                "X-Retake-Identity-Photo-Level":".89",
                "X-Retake-FLUX-Status":"identity-loss" if editor else "not-configured",
                "X-Retake-FLUX-Identity-Delta":"-.076" if editor else "",
                "X-Retake-FLUX-Detail-Delta":"-.11" if editor else ""}
        with patch.object(srv,"process_swap",fake_swap):
            image,headers=srv.run_ultra_search(Path("."),"1024x1024",False,fake_request(),
                lambda *_a,**_k:None,caches={"nim_editors":{"retake-local/flux2-klein-4b":0}},
                min_duration_seconds=0)
        self.assertEqual(len(calls),26)
        self.assertTrue(calls[24][2])
        self.assertFalse(calls[25][2])
        self.assertIsNotNone(calls[25][1])
        self.assertAlmostEqual(float(headers["X-Retake-Similarity"]),.87)
        self.assertEqual(headers["X-Retake-FLUX-Status"],"identity-loss")


    def test_stronger_original_is_not_ignored_by_photo_level_sufficiency(self):
        from tools.ultra_mode import needs_extra_finalist
        headers={**score(.88),"X-Retake-Identity-Photo-Level":".89","X-Retake-Similarity-Before":".90"}
        self.assertTrue(needs_extra_finalist(headers))
        self.assertFalse(needs_extra_finalist({**headers,"X-Retake-Similarity":".90"}))


class ExplicitDetailFailureTest(unittest.TestCase):
    def test_failed_high_score_cannot_win_or_set_peak_in_grid_or_realism(self):
        def fake_swap(*args):
            variant=args[3][6]["variation"]
            if (args[5] or {}).get("realism") or variant==1:
                return b"bad",{**score(.98),"X-Retake-Detail-Gate":"failed"}
            return b"good",{**score(.83),"X-Retake-Detail-Gate":"passed"}
        with patch.object(srv,"process_swap",fake_swap):
            photo,headers=srv.run_ultra_search(Path("."),"1024x1024",False,fake_request(),
                lambda *_a,**_k:None,caches={"realism":{"python":Path("gpu")}},min_duration_seconds=0)
        self.assertEqual(photo,b"good")
        self.assertAlmostEqual(float(headers["X-Retake-Ultra-Peak-Identity"]),.83)
        self.assertEqual(headers["X-Retake-Ultra-Rounds"],"23")


class JobLocalScoreReuseTest(unittest.TestCase):
    def test_one_job_cache_is_shared_across_rounds_and_never_the_next_job(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"same.png"
            path.write_bytes(b"same pixels")
            seen=[]
            measured=[]
            def fake_swap(*args):
                cache=args[5]["identity_scores"]
                seen.append(cache)
                def measure(probes):
                    measured.extend(probes)
                    return [.92]*len(probes)
                value=cache.score([1.0,0.0],[path],measure)[0]
                return b"verified", {**score(value),"X-Retake-Detail-Gate":"passed",
                                    "X-Retake-Identity-Photo-Level":".90"}
            original_caches={}
            with patch.object(srv,"process_swap",fake_swap):
                _,first=srv.run_ultra_search(Path("."),"1024x1024",False,fake_request(),
                    lambda *_a,**_k:None,caches=original_caches,min_duration_seconds=0)
                first_count=len(seen)
                self.assertGreater(first_count,1)
                self.assertTrue(all(c is seen[0] for c in seen))
                self.assertEqual(len(measured),1)
                self.assertEqual(first["X-Retake-Identity-Score-Cache"],f"hits:{first_count-1},measured:1,failed:0")
                _,second=srv.run_ultra_search(Path("."),"1024x1024",False,fake_request(),
                    lambda *_a,**_k:None,caches=original_caches,min_duration_seconds=0)
                self.assertIsNot(seen[first_count],seen[0])
                self.assertEqual(len(measured),2)
                self.assertEqual(first["X-Retake-Similarity"],second["X-Retake-Similarity"])
                self.assertEqual(original_caches,{})


class NoGainFluxRetryTest(unittest.TestCase):
    def test_measured_zero_gain_defers_only_the_next_editor_retry(self):
        from tools.ultra_mode import defer_flux_retry
        base={"X-Retake-FLUX-Identity-Delta":"0.0","X-Retake-FLUX-Detail-Delta":"0.0","X-Retake-FLUX-Skin-Gain":"0.0"}
        self.assertTrue(defer_flux_retry("detail-gain-insufficient",base))
        self.assertFalse(defer_flux_retry("accepted",base))
        for field,value in (("Identity-Delta",".01"),("Detail-Delta",".07"),("Skin-Gain",".06"),("Skin-Gain",""),("Skin-Gain","nan")):
            self.assertFalse(defer_flux_retry("detail-gain-insufficient",{**base,"X-Retake-FLUX-"+field:value}))


class DistinctAttemptLedgerTest(unittest.TestCase):
    def test_failed_grid_and_refinement_are_not_replayed(self):
        from tools.ultra_mode import plan_extra_spec
        grid=srv.FACE_SWAP_VARIATIONS
        scores={0:.82}
        records=[((0,None),score(.82))]
        seen={(0,None),(1,None)}
        spec,local=plan_extra_spec(scores,(0,None),grid,records,set(),set(),
                                  ULTRA_VARIATIONS,True,False,seen)
        self.assertEqual(spec,(2,None))
        self.assertTrue(local)
        seen.update((v,None) for v in ULTRA_VARIATIONS)
        first,_=plan_extra_spec(scores,(0,None),grid,records,set(),set(),
                                ULTRA_VARIATIONS,True,False,seen)
        seen.add(first)
        second,_=plan_extra_spec(scores,(0,None),grid,records,set(),set(),
                                 ULTRA_VARIATIONS,True,False,seen)
        self.assertNotEqual(first,second)
        self.assertIsNotNone(second[1])

    def test_exhausted_winner_explores_other_verified_base(self):
        from tools.ultra_mode import plan_extra_spec, RefinePlanner
        grid=srv.FACE_SWAP_VARIATIONS
        scores={0:.83,3:.82}
        records=[((v,None),score(s)) for v,s in scores.items()]
        seen={(v,None) for v in ULTRA_VARIATIONS}
        planner=RefinePlanner(grid[0],budget=40)
        while True:
            params=planner.next()
            if params is None:
                break
            seen.add((0,params))
            planner.feedback(params,False)
        spec,local=plan_extra_spec(scores,(0,None),grid,records,set(),set(),
                                  ULTRA_VARIATIONS,True,False,seen)
        self.assertEqual(spec[0],3)
        self.assertTrue(local)
        self.assertNotIn(spec,seen)
