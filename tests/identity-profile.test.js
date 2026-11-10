import test from "node:test";
import assert from "node:assert/strict";
import {
  calculateReferenceWeight,
  getSimilarityTransform,
  weightedAverageRGBA
} from "../src/identity-profile.js";

test("aligns two eye points to a canonical horizontal eye line", () => {
  const transform = getSimilarityTransform(
    { x: 10, y: 20 },
    { x: 10, y: 40 },
    { x: 100, y: 100 },
    { x: 140, y: 100 }
  );
  const apply = (point) => ({
    x: transform.a * point.x + transform.c * point.y + transform.e,
    y: transform.b * point.x + transform.d * point.y + transform.f
  });
  const left = apply({ x: 10, y: 20 });
  const right = apply({ x: 10, y: 40 });
  assert.ok(Math.abs(left.x - 100) < 1e-8);
  assert.ok(Math.abs(left.y - 100) < 1e-8);
  assert.ok(Math.abs(right.x - 140) < 1e-8);
  assert.ok(Math.abs(right.y - 100) < 1e-8);
});

test("rejects invalid eye alignment geometry", () => {
  assert.throws(
    () => getSimilarityTransform({ x: 1, y: 1 }, { x: 1, y: 1 }, { x: 0, y: 0 }, { x: 5, y: 5 }),
    /too small/
  );
});

test("weights every usable reference while favoring clear, frontal faces", () => {
  const low = calculateReferenceWeight({
    faceAreaRatio: 0.02, frontalness: 0.2, sharpness: 0.1, exposure: 0.4
  });
  const high = calculateReferenceWeight({
    faceAreaRatio: 0.12, frontalness: 1, sharpness: 1, exposure: 1
  });
  assert.ok(low >= 1 && low <= 3);
  assert.equal(high, 3);
  assert.ok(high > low);
});

test("creates a weighted RGBA profile from all aligned reference images", () => {
  const result = weightedAverageRGBA([
    { data: new Uint8Array([0, 10, 20, 255]), weight: 1 },
    { data: new Uint8Array([100, 110, 120, 255]), weight: 3 }
  ]);
  assert.deepEqual(Array.from(result), [75, 85, 95, 255]);
});

test("rejects mismatched profile buffers", () => {
  assert.throws(() => weightedAverageRGBA([
    { data: new Uint8Array([0, 0, 0, 255]), weight: 1 },
    { data: new Uint8Array([0, 0, 0, 255, 0, 0, 0, 255]), weight: 1 }
  ]), /equal RGBA size/);
});
