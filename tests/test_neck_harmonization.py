import unittest
import cv2
import numpy as np
from tools.neck_harmonization import harmonize_neck

class NeckHarmonizationTest(unittest.TestCase):
    def fixture(self):
        original=np.full((220,220,3),40,np.uint8)
        original[20:140,40:160]=(90,130,180)
        original[140:180,78:122]=(90,130,180)
        original[180:,:]=(210,30,10)
        composite=np.dstack((original.copy(),np.zeros((220,220),np.uint8)))
        composite[20:140,40:160,:3]=(104,145,197)
        composite[20:140,40:160,3]=255
        points=np.zeros((478,2),np.float32)
        points[152]=[100,140]
        points[172]=[60,112]
        points[397]=[140,112]
        points[205]=[76,105]
        points[425]=[124,105]
        return original,composite,{"x":40,"y":20,"width":120,"height":120},points
    def test_neck_changes_only_connected_skin_and_keeps_existing_face(self):
        original,base,bounds,points=self.fixture()
        output,status,count=harmonize_neck(original,base,bounds,points)
        self.assertEqual(status,"applied")
        self.assertGreater(count,0)
        delta=np.any(output!=base,axis=2)
        allowed=np.zeros(delta.shape,bool)
        allowed[140:180,78:122]=True
        self.assertFalse(np.any(delta & ~allowed))
        self.assertEqual(int(np.count_nonzero(delta)),count)
        self.assertTrue(np.array_equal(output[:140],base[:140]))
    def test_no_visible_skin_fails_closed(self):
        original,base,bounds,points=self.fixture()
        original[140:180,78:122]=(40,40,40)
        base[140:180,78:122,:3]=(40,40,40)
        output,status,count=harmonize_neck(original,base,bounds,points)
        self.assertIsNone(output)
        self.assertEqual(count,0)
    def test_invalid_mesh_and_ambiguous_pose_do_not_edit(self):
        original,base,bounds,points=self.fixture()
        for mesh in (None,points[:40]):
            output,status,count=harmonize_neck(original,base,bounds,mesh)
            self.assertIsNone(output)
        points[172]=[60,60]
        output,status,count=harmonize_neck(original,base,bounds,points)
        self.assertEqual(status,"pose-uncertain")
        self.assertIsNone(output)


class NeckTransitionGuardTest(unittest.TestCase):
    def test_no_new_hard_edge_or_unmasked_change(self):
        from tools.neck_harmonization import neck_transition_safe
        src=np.full((40,40,3),120,np.uint8)
        base=np.dstack((src.copy(),np.zeros((40,40),np.uint8)))
        mask=np.zeros((40,40),bool);mask[10:30,10:30]=True
        output=base.copy();output[mask,:3]=124;output[mask,3]=255
        self.assertTrue(neck_transition_safe(src,base,output,mask))
        sharp=output.copy();sharp[mask,:3]=140
        self.assertFalse(neck_transition_safe(src,base,sharp,mask))
        spill=output.copy();spill[0,0,0]=121
        self.assertFalse(neck_transition_safe(src,base,spill,mask))
        base[15,15,3]=255
        self.assertFalse(neck_transition_safe(src,base,output,mask))


class NeckToneEvidenceTest(unittest.TestCase):
    def test_only_correction_towards_measured_offset_is_improvement(self):
        from tools.neck_harmonization import neck_tone_evidence
        source=np.full((20,20,3),130,np.uint8)
        mask=np.ones((20,20),bool)
        brighter=np.full_like(source,135)
        darker=np.full_like(source,125)
        desired=np.array([10,0,0],np.float32)
        self.assertTrue(neck_tone_evidence(source,brighter,mask,desired)["improved"])
        self.assertFalse(neck_tone_evidence(source,darker,mask,desired)["improved"])
        self.assertFalse(neck_tone_evidence(source,source,mask,desired)["improved"])
