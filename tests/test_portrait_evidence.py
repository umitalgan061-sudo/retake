import unittest
import cv2
import numpy as np
from tools.portrait_evidence import reference_evidence, format_evidence
from tools.lower_face_transfer import zoned_transfer_amount, MOUSTACHE
from tools.ultra_mode import identity_aspiration, needs_extra_finalist, prefer_ultra_candidate
from tools import facefusion_local_server as srv


def mesh():
    pts=np.zeros((478,2),np.float32)
    pts[234]=[20,65];pts[454]=[120,65];pts[10]=[70,20];pts[152]=[70,130]
    return pts


class PortraitEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.b=dict(x=20,y=20,width=100,height=110)
        self.base=np.full((160,160,3),140,np.uint8)
        self.edited=self.base.copy()
        rng=np.random.default_rng(7)
        self.edited[100:120,42:100]=rng.integers(75,145,(20,58,1),dtype=np.uint8)

    def test_smooth_source_still_has_absolute_lower_face_evidence(self):
        result,fit=reference_evidence(self.base,self.edited,self.b)
        self.assertIsNone(fit)
        self.assertLess(result["lower_face"][0],1e-3)
        self.assertGreater(result["lower_face"][1],0)
        self.assertIn("lower_face:0.00>",format_evidence(result))

    def test_reference_fit_uses_actual_pixels_and_two_valid_refs(self):
        refs=[(self.edited,mesh()),(self.edited.copy(),mesh())]
        _,good=reference_evidence(self.base,self.edited,self.b,mesh(),refs)
        _,bad=reference_evidence(self.base,self.base,self.b,mesh(),refs)
        self.assertIsNotNone(good)
        self.assertGreater(good,bad)
        self.assertAlmostEqual(good,1.)
        self.assertIsNone(reference_evidence(self.base,self.edited,self.b,mesh(),refs[:1])[1])
        self.assertIsNone(reference_evidence(self.base,self.edited,self.b,None,refs)[1])

    def test_independent_moustache_and_jaw_strengths_never_escape_mask(self):
        pts=np.zeros((478,2),np.float32)
        for idx,xy in zip(MOUSTACHE,[(30,30),(40,30),(50,30),(50,40),(45,45),(40,45),(30,40),(30,35)]):
            pts[idx]=xy
        mask=np.zeros((100,100),np.float32);mask[20:80,20:80]=1
        amount=zoned_transfer_amount(mask,pts,100,.5,1.25,.8)
        self.assertGreater(amount[37,40],amount[70,40])
        self.assertEqual(float(amount[0,0]),0)
        self.assertLessEqual(float(amount.max()),1)

    def test_photo_level_is_the_search_target_when_available(self):
        h={"X-Retake-Similarity":".832","X-Retake-Identity-Median-Similarity":".766","X-Retake-Identity-Photo-Level":".89"}
        self.assertAlmostEqual(identity_aspiration(h),.87)
        self.assertTrue(needs_extra_finalist(h))
        self.assertFalse(needs_extra_finalist({**h,"X-Retake-Similarity":".88"}))

    def test_reference_appearance_breaks_only_measured_near_ties(self):
        h={"X-Retake-Similarity":".832","X-Retake-Reference-Appearance":".6"}
        better={**h,"X-Retake-Reference-Appearance":".9"}
        self.assertTrue(prefer_ultra_candidate(better,h))
        self.assertFalse(prefer_ultra_candidate({**better,"X-Retake-Similarity":".80"},h))
        self.assertFalse(prefer_ultra_candidate({**h,"X-Retake-Reference-Appearance":"nan"},h))

    def test_jury_does_not_regress_a_sufficient_winner(self):
        h={"X-Retake-Similarity":".871","X-Retake-Identity-Photo-Level":".89","X-Retake-Detail-Gate":"passed"}
        below={**h,"X-Retake-Similarity":".869"}
        self.assertEqual(len(srv.judge_pool(b"a",h,1,[(b"b",below,2)],.871)),1)


    def test_distinct_finalist_roles_preserve_measured_peaks(self):
        def row(name,score,appearance,fine):
            return (name.encode(),{"X-Retake-Similarity":str(score),
                 "X-Retake-Reference-Appearance":str(appearance),
                 "X-Retake-Fine-Quality":str(fine),"X-Retake-Detail-Gate":"passed"},1)
        rows=[row("local",.83,.5,.5),row("regional",.829,.95,.5),
              row("fine",.829,.5,.95),row("repeat",.83,.6,.6)]
        pool=srv.judge_pool(*rows[0],rows,.83)
        self.assertEqual(pool[0][0],b"local")
        self.assertEqual({r[0] for r in pool},{b"local",b"regional",b"fine",b"repeat"})
