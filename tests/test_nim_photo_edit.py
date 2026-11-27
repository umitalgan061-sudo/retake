import base64
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from tools.nim_photo_edit import configured_editors, candidate_from_editor, edit_payload

class NimPhotoEditTest(unittest.TestCase):
    def test_config_is_explicit_local_and_allowlisted(self):
        self.assertEqual(configured_editors(""),{})
        model="qwen/qwen-image-edit-2511"
        self.assertEqual(configured_editors(model+"=8001"),{model:8001})
        for value in ["https://example.com=8000",model+"=8765",model+"=http://localhost:8000",model+"=8000,"+model+"=8001"]:
            with self.assertRaises(ValueError): configured_editors(value)
    def test_only_face_interior_existing_matte_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=root/"base.png";out=root/"out.png";ref=root/"ref.png"
            src=np.full((100,100,4),120,np.uint8);src[:,:,3]=0;src[25:75,25:75,3]=255
            cv2.imwrite(str(base),src);cv2.imwrite(str(ref),src[:,:,:3])
            generated=np.full((100,100,3),180,np.uint8)
            data=base64.b64encode(cv2.imencode(".png",generated)[1]).decode("ascii")
            calls=[]
            def request(port,path,payload=None,timeout=180):
                calls.append((port,path,payload))
                return True if payload is None else {"data":[{"b64_json":data}]}
            candidate_from_editor("qwen/qwen-image-edit-2511",8001,base,[ref]*5,dict(x=20,y=20,width=60,height=60),out,request)
            actual=cv2.imread(str(out),cv2.IMREAD_UNCHANGED)
            self.assertEqual(len(calls[1][2]["image"]),3)
            self.assertTrue(np.array_equal(src[:,:,3],actual[:,:,3]))
            self.assertTrue(np.array_equal(src[src[:,:,3]==0],actual[src[:,:,3]==0]))
            self.assertGreater(actual[50,50,0],src[50,50,0])


    def test_multiple_blends_reuse_inference_and_preserve_matte(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); base=root/"base.png"; out=root/"out.png"
            src=np.full((100,100,4),120,np.uint8);src[:,:,3]=0;src[25:75,25:75,3]=255
            cv2.imwrite(str(base),src)
            generated=np.full((100,100,3),180,np.uint8)
            data=base64.b64encode(cv2.imencode(".png",generated)[1]).decode("ascii")
            calls=[]
            def request(port,path,payload=None,timeout=180):
                calls.append(path)
                return True if payload is None else {"data":[{"b64_json":data}]}
            candidate_from_editor("qwen/qwen-image-edit-2511",8001,base,[],dict(x=20,y=20,width=60,height=60),out,request,extra_strengths=(.20,.50))
            self.assertEqual(calls,["/v1/health/ready","/v1/images/edits"])
            values=[]
            for path in (root/"out-20.png",out,root/"out-50.png"):
                actual=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
                self.assertTrue(np.array_equal(actual[:,:,3],src[:,:,3]))
                self.assertTrue(np.array_equal(actual[src[:,:,3]==0],src[src[:,:,3]==0]))
                values.append(int(actual[50,50,0]))
            self.assertTrue(values[0]<values[1]<values[2])


class TextureOnlyCandidateTest(unittest.TestCase):
    def fixture(self):
        original=np.full((200,200,4),120,np.uint8)
        original[:,:,3]=255
        original[:,:25,3]=0
        yy,xx=np.mgrid[:200,:200]
        texture=(120+8*((xx+yy)%2*2-1)).astype(np.uint8)
        generated=np.repeat(texture[:,:,None],3,axis=2)
        return original,generated,dict(x=20,y=20,width=160,height=160)

    def test_texture_is_bounded_and_does_not_edit_eyes_mouth_or_outline(self):
        from tools.nim_photo_edit import texture_only_candidate
        base,generated,bounds=self.fixture()
        result=texture_only_candidate(base,generated,bounds,.5)
        self.assertGreater(np.count_nonzero(result!=base),20)
        self.assertLessEqual(np.max(np.abs(result.astype(int)-base.astype(int))),4)
        self.assertTrue(np.array_equal(base[:,:,3],result[:,:,3]))
        self.assertTrue(np.array_equal(base[base[:,:,3]==0],result[base[:,:,3]==0]))
        self.assertTrue(np.array_equal(base[65:88,40:160],result[65:88,40:160]))
        self.assertTrue(np.array_equal(base[125:,:],result[125:,:]))
        self.assertTrue(np.array_equal(base[:20],result[:20]))
        # Exposure or structure mismatch cannot transfer a new face.
        self.assertTrue(np.array_equal(texture_only_candidate(base,np.full_like(generated,200),bounds,.5),base))

    def test_texture_alternatives_share_one_local_inference(self):
        base,generated,bounds=self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);a=root/"base.png";out=root/"out.png"
            cv2.imwrite(str(a),base)
            data=base64.b64encode(cv2.imencode(".png",generated)[1]).decode("ascii")
            calls=[]
            def request(port,path,payload=None,timeout=180):
                calls.append(path)
                return True if payload is None else {"data":[{"b64_json":data}]}
            candidate_from_editor("qwen/qwen-image-edit-2511",8001,a,[],bounds,out,request,texture_strengths=(.25,.50))
            self.assertEqual(calls,["/v1/health/ready","/v1/images/edits"])
            self.assertTrue((root/"out-texture-25.png").is_file())
            self.assertTrue((root/"out-texture-50.png").is_file())

    def test_skin_gain_requires_native_texture_and_two_measured_regions(self):
        from tools.nim_photo_edit import interior_texture_gain,texture_only_candidate
        base,generated,bounds=self.fixture()
        candidate=texture_only_candidate(base,generated,bounds,.5)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);a,b,c=(root/f"{name}.png" for name in ("source","base","candidate"))
            cv2.imwrite(str(a),generated);cv2.imwrite(str(b),base);cv2.imwrite(str(c),candidate)
            self.assertGreater(interior_texture_gain(a,b,c,bounds),.05)
            self.assertAlmostEqual(interior_texture_gain(a,b,b,bounds),0)
            cv2.imwrite(str(a),base[:,:,:3])
            self.assertIsNone(interior_texture_gain(a,b,c,bounds))


class EditorSelectionTest(unittest.TestCase):
    def test_real_texture_gain_wins_only_inside_identity_band(self):
        from tools.nim_photo_edit import select_editor_candidate, editor_quality_rank
        def row(score, detail, skin, name):
            return (score, .8, "test", name, .25, score-.84, detail, "texture", skin)
        peak=row(.845, 0., 0., "peak")
        detail=row(.844, .01, .12, "texture")
        weak=row(.840, .4, .4, "outside")
        self.assertEqual(select_editor_candidate([peak, detail, weak]),detail)
        self.assertEqual(select_editor_candidate([peak, weak]),peak)
        self.assertEqual(select_editor_candidate([]),None)
        self.assertGreater(editor_quality_rank(0.,.02,.1),editor_quality_rank(.001,None,float("nan")))
