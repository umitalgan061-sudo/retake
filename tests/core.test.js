import test from "node:test";
import assert from "node:assert/strict";
import {
  settleWithLimit,
  ACCEPTED_IMAGE_TYPES,
  getContainedRect,
  getLandmarkBounds,
  formatBytes,
  MAX_IMAGE_EDGE,
  MAX_IMAGE_PIXELS,
  MAX_REFERENCE_FACES,
  validateImageFile,
  validateImageDimensions,
  validateReferenceFiles
} from "../src/core.js";

test("accepts supported image files within the size limit", () => {
  assert.deepEqual(validateImageFile({ type: "image/jpeg", size: 1024 }), { valid: true, reason: "" });
});

test("rejects unsupported types, empty files and oversized files", () => {
  assert.equal(validateImageFile({ type: "image/gif", size: 100 }).valid, false);
  assert.equal(validateImageFile({ type: "image/png", size: 0 }).valid, false);
  assert.equal(validateImageFile({ type: "image/png", size: 26 * 1024 * 1024 }).valid, false);
  assert.equal(ACCEPTED_IMAGE_TYPES.size, 3);
});

test("requires exactly 25 supported, non-empty personal reference photos", () => {
  const files = Array.from({ length: MAX_REFERENCE_FACES }, (_, index) => ({
    name: `face-${index + 1}.jpg`,
    type: "image/jpeg",
    size: 1024
  }));
  assert.equal(validateReferenceFiles(files).valid, true);
  assert.match(validateReferenceFiles(files.slice(0, 24)).reason, /tam 25/);
  assert.match(validateReferenceFiles([...files, files[0]]).reason, /tam 25/);
  assert.match(
    validateReferenceFiles(files.map((file, index) => index === 4 ? { ...file, type: "image/gif" } : file)).reason,
    /face-5\.jpg/
  );
});

test("rejects images that exceed safe pixel dimensions", () => {
  assert.deepEqual(validateImageDimensions(4000, 3000), { valid: true, reason: "" });
  assert.match(validateImageDimensions(MAX_IMAGE_EDGE + 1, 100).reason, /boyutu çok büyük/);
  assert.match(validateImageDimensions(MAX_IMAGE_PIXELS, 2).reason, /boyutu çok büyük/);
  assert.equal(validateImageDimensions(0, 10).valid, false);
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
test("settleWithLimit keeps order, reports failures and never exceeds the limit", async () => {
  let running = 0;
  let peak = 0;
  const results = await settleWithLimit([30, 5, 20, 1, 10], 2, async (delay, index) => {
    running += 1;
    peak = Math.max(peak, running);
    await new Promise((resolve) => setTimeout(resolve, delay));
    running -= 1;
    if (index === 3) throw new Error("bad");
    return index;
  });
  assert.equal(peak, 2);
  assert.deepEqual(results.map((item) => item.status), ["fulfilled", "fulfilled", "fulfilled", "rejected", "fulfilled"]);
  assert.deepEqual(results.filter((item) => item.value !== undefined).map((item) => item.value), [0, 1, 2, 4]);
  assert.deepEqual(await settleWithLimit([], 3, async () => 1), []);
});
