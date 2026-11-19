import test from "node:test";
import assert from "node:assert/strict";
import {
  estimateHeadFit,
  getTargetRenderSize,
  mapLandmarksFromRegion,
  measureFaceShapeRatio,
  orderDetectedFaces,
  rectIoU,
  rectOverlap,
  summarizeReferenceShape,
  weightedMedian
} from "../src/head-geometry.js";

function mesh({ forehead, chin, left, right }) {
  const landmarks = new Array(478).fill(null);
  landmarks[10] = forehead;
  landmarks[152] = chin;
  landmarks[234] = left;
  landmarks[454] = right;
  return landmarks;
}

test("face shape ratio is face length over face width in pixels", () => {
  const landmarks = mesh({
    forehead: { x: 0.5, y: 0.2, z: 0 }, chin: { x: 0.5, y: 0.8, z: 0 },
    left: { x: 0.3, y: 0.5, z: 0 }, right: { x: 0.7, y: 0.5, z: 0 }
  });
  // 600 px tall, 400 px wide on a 1000×1000 image.
  assert.ok(Math.abs(measureFaceShapeRatio(landmarks, 1000, 1000) - 1.5) < 1e-9);
  // Non-square images must be measured in pixels, not normalized units:
  // 600 px tall, 500 px wide on a 1250×1000 image.
  assert.ok(Math.abs(measureFaceShapeRatio(landmarks, 1250, 1000) - 1.2) < 1e-9);
});

test("depth keeps the ratio stable when the head is turned", () => {
  const frontal = mesh({
    forehead: { x: 0.5, y: 0.2, z: 0 }, chin: { x: 0.5, y: 0.8, z: 0 },
    left: { x: 0.3, y: 0.5, z: 0 }, right: { x: 0.7, y: 0.5, z: 0 }
  });
  // 40° yaw: the cheek-to-cheek span shortens on screen but gains depth.
  const yaw = 40 * Math.PI / 180;
  const half = 0.2;
  const turned = mesh({
    forehead: { x: 0.5, y: 0.2, z: 0 }, chin: { x: 0.5, y: 0.8, z: 0 },
    left: { x: 0.5 - half * Math.cos(yaw), y: 0.5, z: -half * Math.sin(yaw) },
    right: { x: 0.5 + half * Math.cos(yaw), y: 0.5, z: half * Math.sin(yaw) }
  });
  const a = measureFaceShapeRatio(frontal, 1000, 1000);
  const b = measureFaceShapeRatio(turned, 1000, 1000);
  assert.ok(Math.abs(a - b) < 1e-9);
});

test("unusable meshes yield no ratio", () => {
  assert.equal(measureFaceShapeRatio(new Array(478).fill(null), 100, 100), null);
  assert.equal(measureFaceShapeRatio(null, 100, 100), null);
  const degenerate = mesh({
    forehead: { x: 0.5, y: 0.5 }, chin: { x: 0.5, y: 0.5 },
    left: { x: 0.3, y: 0.5 }, right: { x: 0.7, y: 0.5 }
  });
  assert.equal(measureFaceShapeRatio(degenerate, 1000, 1000), null);
});

test("weighted median ignores outliers and honours weights", () => {
  assert.equal(weightedMedian([{ value: 1, weight: 1 }, { value: 2, weight: 1 }, { value: 50, weight: 1 }]), 2);
  assert.equal(weightedMedian([{ value: 1, weight: 5 }, { value: 2, weight: 1 }, { value: 3, weight: 1 }]), 1);
  assert.equal(weightedMedian([]), null);
});

test("reference shape needs enough full-mesh references", () => {
  const few = [1.4, 1.42, 1.38].map((value) => ({ value, weight: 1 }));
  assert.equal(summarizeReferenceShape(few), null);
  const enough = [1.4, 1.42, 1.38, 1.41, 1.39, 2.1].map((value) => ({ value, weight: 1 }));
  assert.equal(summarizeReferenceShape(enough), 1.4);
});

test("head fit lengthens and narrows for a longer reference face", () => {
  const fit = estimateHeadFit(1.5, 1.4);
  assert.ok(fit.scaleY > 1 && fit.scaleX < 1);
  assert.ok(Math.abs(fit.scaleX * fit.scaleY - 1) < 1e-3);
});

test("head fit shortens and widens for a rounder reference face", () => {
  const fit = estimateHeadFit(1.3, 1.45);
  assert.ok(fit.scaleY < 1 && fit.scaleX > 1);
});

test("head fit is capped and neutral without measurements", () => {
  const extreme = estimateHeadFit(2.0, 1.0);
  assert.equal(extreme.scaleY, 1.08);
  assert.equal(extreme.scaleX, 0.92);
  assert.deepEqual(estimateHeadFit(null, 1.4), { scaleX: 1, scaleY: 1 });
  assert.deepEqual(estimateHeadFit(1.4, NaN), { scaleX: 1, scaleY: 1 });
  assert.deepEqual(estimateHeadFit(1.4, 1.4), { scaleX: 1, scaleY: 1 });
  // Within measurement noise: do not resample the head for nothing.
  assert.deepEqual(estimateHeadFit(1.41, 1.4), { scaleX: 1, scaleY: 1 });
});

test("target crops in range are sent pixel-for-pixel", () => {
  assert.deepEqual(getTargetRenderSize(1100, 1437), { width: 1100, height: 1437, scale: 1 });
});

test("small crops are raised to the minimum and huge crops capped", () => {
  const small = getTargetRenderSize(392, 512);
  assert.equal(small.height, 1024);
  assert.equal(small.width, 784);
  const huge = getTargetRenderSize(3062, 4000);
  assert.equal(huge.height, 2048);
  assert.equal(huge.width, 1568);
  const wide = getTargetRenderSize(3000, 1500);
  assert.equal(wide.width, 1600);
  assert.equal(wide.height, 800);
  assert.throws(() => getTargetRenderSize(0, 10), TypeError);
});

test("rectangle overlap", () => {
  const a = { x: 0, y: 0, width: 2, height: 2 };
  assert.equal(rectIoU(a, a), 1);
  assert.equal(rectIoU(a, { x: 5, y: 5, width: 1, height: 1 }), 0);
  assert.ok(Math.abs(rectIoU(a, { x: 1, y: 0, width: 2, height: 2 }) - 1 / 3) < 1e-9);
});

test("a distant face inside an oversized detector box counts as the same face", () => {
  // Mirror selfie: BlazeFace's box is ~3x the face and shifted down to the neck.
  const face = { x: 0.38, y: 0.43, width: 0.09, height: 0.1 };
  const box = { x: 0.28, y: 0.47, width: 0.24, height: 0.18 };
  assert.ok(rectIoU(face, box) < 0.25);
  assert.ok(rectOverlap(face, box) > 0.5);
  assert.equal(rectOverlap(face, { x: 0.6, y: 0.43, width: 0.09, height: 0.1 }), 0);
  assert.equal(rectOverlap(face, { ...face, width: 0 }), 0);
});

test("landmarks found in a zoomed region map back to the full image", () => {
  const mapped = mapLandmarksFromRegion(
    [{ x: 0.5, y: 0.5, z: 0.1 }, { x: 0, y: 1 }],
    { x: 0.2, y: 0.4, width: 0.2, height: 0.1 }
  );
  assert.ok(Math.abs(mapped[0].x - 0.3) < 1e-12 && Math.abs(mapped[0].y - 0.45) < 1e-12);
  assert.ok(Math.abs(mapped[0].z - 0.02) < 1e-12);
  assert.deepEqual(mapped[1], { x: 0.2, y: 0.5, z: 0 });
});

import {
  JAW_INDICES,
  canonicalJaw,
  eyeFrame,
  fromFrame,
  jawPointsFromMesh,
  meshFrontalness,
  planJawWarp,
  summarizeReferenceJaw,
  toFrame
} from "../src/head-geometry.js";

// A synthetic mesh: eyes, nose, mouth and a jaw shaped as a half ellipse.
function faceMesh({ cx = 0.5, eyeY = 0.4, eyeHalf = 0.1, jawW = 0.22, jawH = 0.32, noseX = 0, chinPoint = 1 } = {}) {
  const mesh = new Array(478).fill(null);
  mesh[33] = { x: cx - eyeHalf, y: eyeY };
  mesh[263] = { x: cx + eyeHalf, y: eyeY };
  mesh[1] = { x: cx + noseX, y: eyeY + 0.12 };
  mesh[168] = { x: cx, y: eyeY - 0.01 };
  mesh[61] = { x: cx - 0.05, y: eyeY + 0.2 };
  mesh[291] = { x: cx + 0.05, y: eyeY + 0.2 };
  mesh[13] = { x: cx, y: eyeY + 0.19 };
  mesh[14] = { x: cx, y: eyeY + 0.21 };
  const last = JAW_INDICES.length - 1;
  JAW_INDICES.forEach((index, i) => {
    const angle = Math.PI * (i / last);          // 0 → left cheek, π → right cheek
    const point = { x: cx - Math.cos(angle) * jawW, y: eyeY + 0.4 * eyeHalf + Math.sin(angle) * jawH * (i === last / 2 ? chinPoint : 1) };
    mesh[index] = point;
  });
  return mesh;
}

test("eye frame round-trips points", () => {
  const frame = eyeFrame({ x: 100, y: 200 }, { x: 180, y: 260 });
  const point = { x: 133, y: 321 };
  const back = fromFrame(frame, toFrame(frame, point));
  assert.ok(Math.abs(back.x - point.x) < 1e-9 && Math.abs(back.y - point.y) < 1e-9);
  assert.equal(eyeFrame({ x: 1, y: 1 }, { x: 1, y: 1 }), null);
});

test("canonical jaw ignores position, size and roll", () => {
  const a = canonicalJaw(faceMesh(), 1000, 1000);
  const b = canonicalJaw(faceMesh({ cx: 0.3, eyeHalf: 0.05, jawW: 0.11, jawH: 0.16 }), 1000, 1000);
  a.forEach((point, i) => {
    assert.ok(Math.abs(point.x - b[i].x) < 1e-9 && Math.abs(point.y - b[i].y) < 1e-9);
  });
  assert.equal(canonicalJaw(new Array(478).fill(null), 100, 100), null);
});

test("frontalness drops as the nose leaves the eye midpoint", () => {
  assert.equal(meshFrontalness(faceMesh(), 1000, 1000), 1);
  assert.ok(meshFrontalness(faceMesh({ noseX: 0.05 }), 1000, 1000) < 0.8);
  assert.equal(meshFrontalness(faceMesh({ noseX: 0.2 }), 1000, 1000), 0);
});

test("the reference jaw is a weighted median that ignores an outlier", () => {
  const normal = canonicalJaw(faceMesh(), 1000, 1000);
  const odd = canonicalJaw(faceMesh({ jawH: 0.6 }), 1000, 1000);
  const samples = [...Array(5)].map(() => ({ points: normal, weight: 1 })).concat([{ points: odd, weight: 1 }]);
  const summary = summarizeReferenceJaw(samples);
  summary.forEach((point, i) => assert.ok(Math.abs(point.y - normal[i].y) < 1e-9));
  assert.equal(summarizeReferenceJaw(samples.slice(0, 4)), null);
});

test("a longer reference chin pulls the photo's chin down, cheeks move less", () => {
  const reference = canonicalJaw(faceMesh({ jawH: 0.36 }), 1000, 1000);
  const target = jawPointsFromMesh(faceMesh(), 1000, 1000);
  const plan = planJawWarp(target, reference);
  const chin = plan.controls[10];
  const cheek = plan.controls[1];
  assert.ok(chin.to[1] > chin.from[1], "chin moves down");
  const shift = (control) => Math.hypot(control.to[0] - control.from[0], control.to[1] - control.from[1]);
  assert.ok(shift(chin) > shift(cheek) * 1.5);
  assert.equal(plan.anchors.length, 8);
});

test("shifts are capped, same shapes and turned heads give no warp", () => {
  const target = jawPointsFromMesh(faceMesh(), 1000, 1000);
  const huge = planJawWarp(target, canonicalJaw(faceMesh({ jawH: 0.9 }), 1000, 1000));
  const width = 2 * 0.22 * 1000;
  for (const control of huge.controls) {
    assert.ok(Math.hypot(control.to[0] - control.from[0], control.to[1] - control.from[1]) <= width * 0.07 + 0.02);
  }
  assert.equal(planJawWarp(target, canonicalJaw(faceMesh(), 1000, 1000)), null);
  const turned = jawPointsFromMesh(faceMesh({ noseX: 0.06 }), 1000, 1000);
  assert.equal(planJawWarp(turned, canonicalJaw(faceMesh({ jawH: 0.36 }), 1000, 1000)), null);
});

test("small jaw differences within pose noise are ignored", () => {
  const target = jawPointsFromMesh(faceMesh(), 1000, 1000);
  // 0.32 → 0.328: an 8 px chin difference on a 440 px wide face.
  assert.equal(planJawWarp(target, canonicalJaw(faceMesh({ jawH: 0.328 }), 1000, 1000)), null);
});

test("a face both detectors agree on is preselected over a larger lone mesh 'face'", () => {
  const rust = { x: 0.58, y: 0.32, width: 0.23, height: 0.22, confirmed: false };
  const person = { x: 0.15, y: 0.45, width: 0.12, height: 0.12, confirmed: true };
  const small = { x: 0.7, y: 0.7, width: 0.05, height: 0.05, confirmed: true };
  assert.deepEqual(orderDetectedFaces([rust, small, person]), [person, small, rust]);
  const a = { x: 0, y: 0, width: 0.1, height: 0.1, confirmed: true };
  const b = { x: 0.5, y: 0, width: 0.2, height: 0.2, confirmed: true };
  assert.deepEqual(orderDetectedFaces([a, b]), [b, a]);
});
