import test from "node:test";
import assert from "node:assert/strict";
import { alphaMaskFromRGBA, compositeMaskedRGBA, inspectOutsideMaskIntegrity } from "../src/face-only.js";

test("copies every pixel outside the face mask byte-for-byte", () => {
  const source = new Uint8Array([
    10, 20, 30, 255,
    40, 50, 60, 255,
    70, 80, 90, 255
  ]);
  const generated = new Uint8Array([
    200, 210, 220, 255,
    180, 170, 160, 255,
    150, 140, 130, 255
  ]);
  const mask = new Float32Array([0, 1, 0]);
  const output = compositeMaskedRGBA(source, generated, mask, 3, 1);

  assert.deepEqual(Array.from(output.slice(0, 4)), Array.from(source.slice(0, 4)));
  assert.deepEqual(Array.from(output.slice(4, 8)), Array.from(generated.slice(4, 8)));
  assert.deepEqual(Array.from(output.slice(8, 12)), Array.from(source.slice(8, 12)));
  assert.deepEqual(inspectOutsideMaskIntegrity(source, output, mask, 3, 1), {
    ok: true,
    changedPixels: 0
  });
});

test("supports soft alpha only on the face-mask edge", () => {
  const source = new Uint8Array([0, 0, 0, 255]);
  const generated = new Uint8Array([200, 100, 50, 255]);
  const output = compositeMaskedRGBA(source, generated, new Float32Array([0.5]), 1, 1);
  assert.deepEqual(Array.from(output), [100, 50, 25, 255]);
});

test("rejects mismatched buffers and invalid mask values", () => {
  assert.throws(() => compositeMaskedRGBA(
    new Uint8Array(4),
    new Uint8Array(4),
    new Float32Array([0, 1]),
    1,
    1
  ), /alphaMask/);

  assert.throws(() => compositeMaskedRGBA(
    new Uint8Array(4),
    new Uint8Array(4),
    new Float32Array([1.5]),
    1,
    1
  ), /between 0 and 1/);
});

test("integrity check detects accidental edits outside the mask", () => {
  const source = new Uint8Array([1, 2, 3, 255, 5, 6, 7, 255]);
  const output = new Uint8Array([9, 2, 3, 255, 5, 6, 7, 255]);
  const mask = new Float32Array([0, 1]);
  assert.deepEqual(inspectOutsideMaskIntegrity(source, output, mask, 2, 1), {
    ok: false,
    changedPixels: 1
  });
});

test("accepts ImageData-style clamped pixel buffers", () => {
  const source = new Uint8ClampedArray([1, 2, 3, 255]);
  const generated = new Uint8ClampedArray([10, 20, 30, 255]);
  const output = compositeMaskedRGBA(source, generated, new Float32Array([0]), 1, 1);
  assert.ok(output instanceof Uint8ClampedArray);
  assert.deepEqual(Array.from(output), Array.from(source));
});

test("extracts FaceFusion PNG alpha matte as a normalized mask", () => {
  const rgba = new Uint8ClampedArray([
    200, 10, 20, 0,
    100, 110, 120, 128,
    30, 40, 50, 255
  ]);
  const mask = alphaMaskFromRGBA(rgba, 3, 1);
  assert.equal(mask[0], 0);
  assert.ok(Math.abs(mask[1] - 128 / 255) < 1e-7);
  assert.equal(mask[2], 1);
});

test("rejects an invalid RGBA alpha matte", () => {
  assert.throws(() => alphaMaskFromRGBA(new Uint8Array([0, 0, 0]), 1, 1), /width \* height \* 4/);
});
