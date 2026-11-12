import test from "node:test";
import assert from "node:assert/strict";
import {
  estimateHeadFit,
  getTargetRenderSize,
  mapLandmarksFromRegion,
  measureFaceShapeRatio,
  rectIoU,
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

test("landmarks found in a zoomed region map back to the full image", () => {
  const mapped = mapLandmarksFromRegion(
    [{ x: 0.5, y: 0.5, z: 0.1 }, { x: 0, y: 1 }],
    { x: 0.2, y: 0.4, width: 0.2, height: 0.1 }
  );
  assert.ok(Math.abs(mapped[0].x - 0.3) < 1e-12 && Math.abs(mapped[0].y - 0.45) < 1e-12);
  assert.ok(Math.abs(mapped[0].z - 0.02) < 1e-12);
  assert.deepEqual(mapped[1], { x: 0.2, y: 0.5, z: 0 });
});
