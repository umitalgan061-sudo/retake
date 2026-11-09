import test from "node:test";
import assert from "node:assert/strict";
import { ACCEPTED_IMAGE_TYPES, getContainedRect, getLandmarkBounds, formatBytes, validateImageFile } from "../src/core.js";

test("accepts supported image files within the size limit", () => {
  assert.deepEqual(validateImageFile({ type: "image/jpeg", size: 1024 }), { valid: true, reason: "" });
});

test("rejects unsupported types, empty files and oversized files", () => {
  assert.equal(validateImageFile({ type: "image/gif", size: 100 }).valid, false);
  assert.equal(validateImageFile({ type: "image/png", size: 0 }).valid, false);
  assert.equal(validateImageFile({ type: "image/png", size: 26 * 1024 * 1024 }).valid, false);
  assert.equal(ACCEPTED_IMAGE_TYPES.size, 3);
});

test("computes a normalized face rectangle with padding", () => {
  const result = getLandmarkBounds([{ x: .2, y: .25 }, { x: .8, y: .75 }, { x: .5, y: .4 }]);
  assert.ok(result);
  assert.ok(result.x < .2 && result.y < .25);
  assert.ok(result.x + result.width > .8 && result.y + result.height > .75);
});

test("clamps landmark rectangles to image bounds and rejects invalid data", () => {
  const result = getLandmarkBounds([{ x: -.3, y: .1 }, { x: 1.2, y: .9 }]);
  assert.equal(result.x, 0);
  assert.ok(result.y >= 0 && result.x + result.width <= 1);
  assert.equal(getLandmarkBounds([]), null);
  assert.equal(getLandmarkBounds([{ x: Number.NaN, y: 0 }]), null);
});

test("fits image rectangles without changing aspect ratio", () => {
  assert.deepEqual(getContainedRect(400, 200, 300, 300), { x: 0, y: 75, width: 300, height: 150, scale: .75 });
  assert.equal(getContainedRect(0, 200, 300, 300), null);
});

test("formats byte counts for Turkish UI", () => {
  assert.equal(formatBytes(0), "0 B");
  assert.match(formatBytes(1024), /1 KB/);
  assert.match(formatBytes(1024 * 1024), /1 MB/);
});