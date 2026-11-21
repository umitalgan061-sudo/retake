import test from "node:test";
import assert from "node:assert/strict";
import { alphaMaskFromRGBA, compositeMaskedRGBA, compositePreblendedRGBA, inspectOutsideMaskIntegrity } from "../src/face-only.js";

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

test("a preblended engine crop is copied once; its matte is not squared", () => {
  const original = new Uint8ClampedArray([
    20, 30, 40, 255, 20, 30, 40, 255, 20, 30, 40, 255
  ]);
  // Engine already applied its own feather: the middle face pixel is 120,
  // not the raw swap's 220. An extra 0.5 blend would incorrectly make it 70.
  const engine = new Uint8ClampedArray([
    0, 0, 0, 0, 120, 130, 140, 128, 220, 230, 240, 255
  ]);
  const mask = alphaMaskFromRGBA(engine, 3, 1);
  const output = compositePreblendedRGBA(original, engine, mask, 3, 1);
  assert.deepEqual([...output], [
    20, 30, 40, 255, 120, 130, 140, 255, 220, 230, 240, 255
  ]);
  assert.deepEqual(inspectOutsideMaskIntegrity(original, output, mask, 3, 1),
    { ok: true, changedPixels: 0 });
  const doubled = compositeMaskedRGBA(original, engine, mask, 3, 1);
  assert.notDeepEqual([...doubled.slice(4, 7)], [...output.slice(4, 7)]);
});

test("preblended copy rejects corrupt matte and preserves untouched pixels", () => {
  const original = new Uint8Array([10, 20, 30, 255]);
  const composite = new Uint8Array([180, 190, 200, 128]);
  assert.deepEqual([...compositePreblendedRGBA(original, composite, new Float32Array([0]), 1, 1)],
    [...original]);
  assert.throws(() => compositePreblendedRGBA(original, composite, new Float32Array([NaN]), 1, 1), /finite/);
  assert.throws(() => compositePreblendedRGBA(original, composite, new Float32Array([0, 1]), 1, 1), /extentMask/);
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

test("the photo stays fully opaque: the patch's matte alpha never leaks into the result", () => {
  // 3 pixels: outside the matte, in the feathered ring, fully inside.
  const source = new Uint8ClampedArray([10, 20, 30, 255, 10, 20, 30, 255, 10, 20, 30, 255]);
  const generated = new Uint8ClampedArray([0, 0, 0, 0, 200, 100, 50, 128, 200, 100, 50, 255]);
  const mask = new Float32Array([0, 128 / 255, 1]);
  const out = compositeMaskedRGBA(source, generated, mask, 3, 1);
  assert.deepEqual([out[3], out[7], out[11]], [255, 255, 255]);
  assert.deepEqual([...out.slice(0, 3)], [10, 20, 30]);
  assert.deepEqual([...out.slice(8, 11)], [200, 100, 50]);
  assert.ok(out[4] > 10 && out[4] < 200, "ring pixel is a colour blend");
});
