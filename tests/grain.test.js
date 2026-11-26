import test from "node:test";
import assert from "node:assert/strict";
import {
  MAX_GRAIN, MIN_UPSCALE, describeNativeFinish, finishUpscaledFace, grainLevel, mulberry32
} from "../src/grain.js";

function photo(width, height, sigma, seed = 7) {
  const random = mulberry32(seed);
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < data.length; i += 4) {
    let u = 0;
    while (u === 0) u = random();
    const noise = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random()) * sigma;
    data[i] = 170 + noise;
    data[i + 1] = 140 + noise;
    data[i + 2] = 120 + noise;
    data[i + 3] = 255;
  }
  return data;
}

function matte(width, height, x0, x1) {
  const alpha = new Float32Array(width * height);
  for (let y = 0; y < height; y += 1) for (let x = x0; x < x1; x += 1) alpha[y * width + x] = 1;
  return alpha;
}

test("grain level is the robust noise sigma", () => {
  const alpha = matte(120, 120, 0, 120);
  const level = grainLevel(photo(120, 120, 4), alpha, 120, 120);
  assert.ok(level > 3.2 && level < 4.4, String(level));
  assert.ok(grainLevel(photo(120, 120, 0), alpha, 120, 120) < 0.5);
});

test("an up-scaled smooth face gets the photo's grain back, inside the matte only", () => {
  const width = 160;
  const height = 140;
  const source = photo(width, height, 4);
  const generated = photo(width, height, 0.5, 9);
  const untouched = new Uint8ClampedArray(generated);
  const alpha = matte(width, height, 40, 140);
  const report = finishUpscaledFace(source, generated, alpha, width, height, 1.4, 42);
  assert.equal(report.applied, true);
  assert.ok(report.sharpen > 0 && report.sharpen <= 0.35);
  assert.ok(report.addedGrain > 2.5 && report.addedGrain <= MAX_GRAIN, JSON.stringify(report));
  const after = grainLevel(generated, alpha, width, height);
  assert.ok(Math.abs(after - report.sourceGrain) < 1.0, `${after} vs ${report.sourceGrain}`);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (alpha[y * width + x] > 0) continue;
      const i = (y * width + x) * 4;
      assert.deepEqual([...generated.subarray(i, i + 4)], [...untouched.subarray(i, i + 4)]);
    }
  }
  // Seeded: the same photo gets the same grain on every export.
  const again = photo(width, height, 0.5, 9);
  finishUpscaledFace(source, again, alpha, width, height, 1.4, 42);
  assert.deepEqual([...again], [...generated]);
});

test("no up-scale, or already enough grain, changes nothing", () => {
  const source = photo(80, 80, 3);
  const generated = photo(80, 80, 3, 11);
  const copy = new Uint8ClampedArray(generated);
  const alpha = matte(80, 80, 0, 80);
  assert.equal(finishUpscaledFace(source, generated, alpha, 80, 80, 1.0).applied, false);
  assert.equal(finishUpscaledFace(source, generated, alpha, 80, 80, MIN_UPSCALE - 0.01).applied, false);
  assert.deepEqual([...generated], [...copy]);
  const report = finishUpscaledFace(source, generated, alpha, 80, 80, 1.06);
  assert.equal(report.addedGrain, 0, "grain already matches the photo");
  assert.equal(finishUpscaledFace(source, generated, new Float32Array(6400), 80, 80, 2).applied, false);
});

test("describes the native-resolution finish", () => {
  assert.equal(describeNativeFinish(null), "");
  assert.equal(describeNativeFinish({ applied: false }), "");
  assert.equal(
    describeNativeFinish({ applied: true, scale: 1.42, sharpen: 0.25, sourceGrain: 3.1, resultGrain: 1.4, addedGrain: 2.8 }),
    "Doğal çözünürlük: yüz 1,4× büyütüldü · büyütme yumuşaması giderildi · fotoğrafın greni eklendi (1,4 → 3,1)"
  );
});
