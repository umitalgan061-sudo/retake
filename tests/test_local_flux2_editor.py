import base64
import io
import unittest

from PIL import Image
from tools.local_flux2_editor import MODEL_ALIAS, decode_image, parse_edit, model_dimensions, edit_image

def png_data(size=(128,128), color=(30,80,150)):
    buffer=io.BytesIO()
    Image.new("RGB",size,color).save(buffer,format="PNG")
    return "data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode()

class LocalFlux2EditorTest(unittest.TestCase):
    def test_model_payload_requires_local_images_and_small_reference_pool(self):
        payload={"model":MODEL_ALIAS,"prompt":"edit","image":[png_data(),png_data()],"n":1,"response_format":"b64_json"}
        prompt,images=parse_edit(payload)
        self.assertEqual(prompt,"edit")
        self.assertEqual(len(images),2)
        for bad in [dict(payload,image="https://example.com/photo.png"),dict(payload,image=[png_data()]*4),
                    dict(payload,model="other"),dict(payload,extra=True),dict(payload,n=2)]:
            with self.assertRaises(ValueError):parse_edit(bad)
    def test_invalid_encoding_is_rejected(self):
        for data in ["data:image/png;base64,!","data:image/gif;base64,AAAA","file:///tmp/private.png"]:
            with self.assertRaises(ValueError):decode_image(data)
    def test_dimensions_are_bounded_multiples(self):
        self.assertEqual(model_dimensions(1536,1024,768),(768,512))
        self.assertEqual(model_dimensions(511,513,512),(512,512))
        with self.assertRaises(ValueError):model_dimensions(10,100,768)
    def test_inference_wrapper_uses_real_image_api_and_preserves_size(self):
        from contextlib import nullcontext
        class Generator:
            def __init__(self,device):self.device=device
            def manual_seed(self,seed):return self
        class Torch:
            inference_mode=staticmethod(nullcontext)
        Torch.Generator=Generator
        calls=[]
        def pipeline(**kwargs):
            calls.append(kwargs)
            return type("Output",(),{"images":[Image.new("RGB",(kwargs["width"],kwargs["height"]),(99,40,80))]})()
        result=edit_image(pipeline,Torch,"edit",[Image.new("RGB",(1000,700)),Image.new("RGB",(600,600))],512)
        self.assertEqual(result.size,(1000,700))
        self.assertEqual(calls[0]["num_inference_steps"],4)
        self.assertEqual(len(calls[0]["image"]),2)
        self.assertEqual(calls[0]["width"],512)
        self.assertEqual(calls[0]["image"][1].size,(512,512))
