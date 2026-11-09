import test from "node:test";
import assert from "node:assert/strict";
import {
  MAX_INPUT_BYTES,
  decodePngDataUrl,
  imageDataUrlFromNvidiaResponse,
  describeNvidiaFailure
} from "../functions/lib/nvidia.js";

const PNG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00]);

test("accepts PNG data URL and decodes the binary image", () => {
  const result = decodePngDataUrl("data:image/png;base64," + PNG.toString("base64"));
  assert.deepEqual(result, PNG);
});

test("rejects non-PNG data URLs and malformed base64", () => {
  assert.throws(() => decodePngDataUrl("data:image/jpeg;base64,AAAA"), /PNG/);
  assert.throws(() => decodePngDataUrl("data:image/png;base64,%%%"), /PNG/);
});

test("rejects empty and oversized image data", () => {
  assert.throws(() => decodePngDataUrl("data:image/png;base64,"), /PNG/);
  const oversized = "data:image/png;base64," + Buffer.alloc(MAX_INPUT_BYTES + 1, 1).toString("base64");
  assert.throws(() => decodePngDataUrl(oversized), /10 MB/);
});

test("parses the NVIDIA artifacts base64 response", () => {
  assert.equal(
    imageDataUrlFromNvidiaResponse({ artifacts: [{ base64: "YWJj" }] }),
    "data:image/png;base64,YWJj"
  );
});

test("accepts a provider-returned image data URL", () => {
  assert.equal(
    imageDataUrlFromNvidiaResponse({ image: "data:image/jpeg;base64,YWJj" }),
    "data:image/jpeg;base64,YWJj"
  );
});

test("explains permission, quota and unavailable endpoint failures", () => {
  assert.match(describeNvidiaFailure(403), /invoke_function/);
  assert.match(describeNvidiaFailure(402), /kota/);
  assert.match(describeNvidiaFailure(404), /FLUX.1-Kontext-dev/);
});
