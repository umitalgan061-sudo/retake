import test from "node:test";
import assert from "node:assert/strict";
import {
  descriptorDistance,
  MESH_POINTS,
  meshToPixels,
  pickRealReference,
  poseDescriptor,
  rankRealReferences,
  referenceCrop
} from "../src/real-face.js";

function mesh({ cx = 0.5, cy = 0.5, scale = 0.1, mouthOpen = 0, turn = 0 } = {}) {
  // Deterministic pseudo-face: every point on a fixed layout, eyes at ±scale.
  const points = [];
  for (let index = 0; index < MESH_POINTS; index += 1) {
    const angle = index * 2.399963;
    const radius = Math.sqrt(index / MESH_POINTS) * 2.2;
    points.push({ x: cx + Math.cos(angle) * radius * scale + turn * scale * (1 - Math.abs(Math.cos(angle))), y: cy + Math.sin(angle) * radius * scale });
  }
  points[33] = { x: cx - scale, y: cy };
  points[263] = { x: cx + scale, y: cy };
  points[13] = { x: cx, y: cy + 1.2 * scale };
  points[14] = { x: cx, y: cy + (1.25 + mouthOpen) * scale };
  return points;
}

test("pose signature ignores position and size but sees expression and turn", () => {
  const a = poseDescriptor(mesh(), 1000, 1000);
  const b = poseDescriptor(mesh({ cx: 0.3, cy: 0.6, scale: 0.05 }), 1000, 1000);
  assert.ok(descriptorDistance(a, b) < 1e-3);
  assert.ok(descriptorDistance(a, poseDescriptor(mesh({ mouthOpen: 0.6 }), 1000, 1000)) > 0.005);
  assert.ok(descriptorDistance(a, poseDescriptor(mesh({ turn: 0.4 }), 1000, 1000)) > 0.02);
  assert.equal(poseDescriptor(new Array(10).fill({ x: 0, y: 0 }), 100, 100), null);
});

test("the closest pose wins; weight breaks near ties; far poses are refused", () => {
  const target = poseDescriptor(mesh(), 1000, 1000);
  const near = poseDescriptor(mesh({ turn: 0.042 }), 1000, 1000);
  const nearer = poseDescriptor(mesh({ turn: 0.04 }), 1000, 1000);
  const far = poseDescriptor(mesh({ turn: 1.5, mouthOpen: 1 }), 1000, 1000);
  assert.equal(pickRealReference([far, near, nearer], [1, 1, 1], target), 2);
  assert.equal(pickRealReference([near, nearer], [3, 1], target), 0, "a much better photo wins a near tie");
  assert.equal(pickRealReference([far], [3], target), -1);
  assert.equal(pickRealReference([], [], target), -1);
});

test("up to three candidates are ranked best first, far poses never included", () => {
  const target = poseDescriptor(mesh(), 1000, 1000);
  const at = (turn) => poseDescriptor(mesh({ turn }), 1000, 1000);
  const far = poseDescriptor(mesh({ turn: 1.5, mouthOpen: 1 }), 1000, 1000);
  const descriptors = [at(0.08), far, at(0.02), at(0.05), at(0.035), far];
  assert.deepEqual(rankRealReferences(descriptors, new Array(6).fill(1), target), [2, 4, 3]);
  assert.deepEqual(rankRealReferences(descriptors, new Array(6).fill(1), target, undefined, 5), [2, 4, 3, 0]);
  assert.deepEqual(rankRealReferences([far, far], [1, 1], target), []);
});

test("reference crop surrounds the face and is capped in size", () => {
  const crop = referenceCrop(mesh({ scale: 0.1 }), 4000, 3000);
  assert.ok(crop.outWidth <= 1024 && crop.outHeight <= 1024);
  assert.ok(crop.scale < 1);
  assert.ok(crop.x >= 0 && crop.y >= 0 && crop.x + crop.width <= 4000 && crop.y + crop.height <= 3000);
});

test("mesh is mapped into pixel space", () => {
  const pixels = meshToPixels(mesh(), 200, 100, (p) => ({ x: p.x - 10, y: p.y * 2 }));
  assert.equal(pixels.length, MESH_POINTS);
  assert.deepEqual(pixels[33], [200 * 0.4 - 10, 100]);
});
