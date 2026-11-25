import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from tools.facefusion_quality import measure_final_face_edit

class FinalFaceEditTest(unittest.TestCase):
    def test_original_and_invisible_changes_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/"a.png", Path(folder)/"b.png"
            image = np.full((100,100,3),120,np.uint8)
            cv2.imwrite(str(a),image)
            result = cv2.cvtColor(image,cv2.COLOR_BGR2BGRA)
            bounds = dict(x=20,y=20,width=60,height=60)
            cv2.imwrite(str(b),result)
            self.assertFalse(measure_final_face_edit(a,b,bounds)["meaningful"])
            result[35:65,35:65,:3]=150
            result[:,:,3]=0
            cv2.imwrite(str(b),result)
            self.assertFalse(measure_final_face_edit(a,b,bounds)["meaningful"])
            result[35:65,35:65,3]=255
            cv2.imwrite(str(b),result)
            self.assertTrue(measure_final_face_edit(a,b,bounds)["meaningful"])
