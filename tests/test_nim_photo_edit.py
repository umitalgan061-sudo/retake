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
