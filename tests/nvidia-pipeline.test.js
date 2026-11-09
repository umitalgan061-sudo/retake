import test from "node:test";
import assert from "node:assert/strict";
import {
  NVIDIA_BOARD_WIDTH,
  NVIDIA_BOARD_HEIGHT,
  getFaceCropRect,
  createFaceAlphaMask,
  createNvidiaEditPrompt,
  imageDataUrlFromNvidiaResponse
} from "../src/nvidia-pipeline.js";

test("uses a supported 1568 by 1024 NVIDIA image board", () => {
  assert.equal(NVIDIA_BOARD_WIDTH, 1568);
  assert.equal(NVIDIA_BOARD_HEIGHT, 1024);
});

test("creates a crop with the board panel aspect ratio and contains the whole face", () => {
  const face = { x: 0.35, y: 0.2, width: 0.2, height: 0.3 };
  const crop = getFaceCropRect(2400, 3600, face);
  assert.ok(Math.abs(crop.width / crop.height - 0.765625) < 0.01);
  assert.ok(crop.x <= face.x * 2400);
  assert.ok(crop.y <= face.y * 3600);
  assert.ok(crop.x + crop.width >= (face.x + face.width) * 2400 - 1);
  assert.ok(crop.y + crop.height >= (face.y + face.height) * 3600 - 1);
});

test("rejects invalid face bounds", () => {
  assert.throws(() => getFaceCropRect(100, 100, { x: 0, y: 0, width: 0, height: 0 }), /valid normalized face/);
});

test("creates a soft elliptical face mask and leaves distant pixels untouched", () => {
  const mask = createFaceAlphaMask(100, 120, { x: 35, y: 30, width: 30, height: 50 });
  const center = mask[55 * 100 + 50];
  const outside = mask[5 * 100 + 5];
  assert.equal(center, 1);
  assert.equal(outside, 0);
  assert.ok(mask.some((alpha) => alpha > 0 && alpha < 1));
});

test("prompt explicitly distinguishes target and twenty-five identity references", () => {
  const prompt = createNvidiaEditPrompt();
  assert.match(prompt, /twenty-five different reference photographs/i);
  assert.match(prompt, /LEFT PANEL/);
  assert.match(prompt, /RIGHT PANEL/);
  assert.match(prompt, /eyes are clearly closed/i);
});


test("normalizes NVIDIA base64 image responses in the browser", () => {
  assert.equal(
    imageDataUrlFromNvidiaResponse({ artifacts: [{ base64: "AQID" }] }),
    "data:image/png;base64,AQID"
  );
  assert.equal(
    imageDataUrlFromNvidiaResponse({ images: [{ b64_json: "data:image/webp;base64,AQID" }] }),
    "data:image/webp;base64,AQID"
  );
  assert.throws(() => imageDataUrlFromNvidiaResponse({ artifacts: [] }), /base64 görsel çıktısı/);
});
