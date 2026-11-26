// Native-resolution finishing (v27).
//
// The engine works on at most 1600×2048 px. A close-up portrait from a phone
// gives a larger face crop, so the result is scaled up again in the browser:
// the new face then looks smoother than the rest of the photo — a classic
// swap tell (no pores, no sensor grain where the cheek meets the original
// skin). After the up-scale this restores, inside the edit matte only:
//
// 1. a mild sharpening that undoes the interpolation blur, bounded by the
//    up-scale factor;
// 2. the photo's own grain level: the robust (median-absolute-deviation)
//    fine-detail level of the original face region at native resolution is
//    compared with the result's, and only the missing part is added as
//    luminance grain. No texture is copied from the original face, so the
//    original person's features cannot reappear.
//
// Pixels outside the matte are never touched.

export const MIN_UPSCALE = 1.05;
export const MAX_SHARPEN = 0.35;
export const SHARPEN_PER_SCALE = 0.6;
export const MAX_GRAIN = 6;
export const MIN_GRAIN_DEFICIT = 0.6;
// Estimates of two noisy images differ by a few percent: only a clearly
// smoother result (below 85 % of the photo's grain) gets grain.
export const GRAIN_TRIGGER = 0.85;
// Fine band of white noise (pixel − 3×3 mean) keeps sqrt(8/9) of its sigma.
const WHITE_NOISE_FINE_GAIN = Math.sqrt(8 / 9);
const MAD_TO_SIGMA = 1.4826;
const SAMPLE_LIMIT = 200000;

function luma(data, index) {
  return 0.299 * data[index] + 0.587 * data[index + 1] + 0.114 * data[index + 2];
}

function maskBounds(alpha, width, height) {
  let x0 = width;
  let y0 = height;
  let x1 = -1;
  let y1 = -1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (alpha[y * width + x] > 0) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  return x1 < 0 ? null : { x0, y0, x1, y1 };
}

// Luma of the 3×3 neighbourhood mean (borders clamp), for the box region.
function boxMeanLuma(data, width, height, box) {
  const w = box.x1 - box.x0 + 1;
  const h = box.y1 - box.y0 + 1;
  const result = new Float32Array(w * h);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      let sum = 0;
      for (let dy = -1; dy <= 1; dy += 1) {
        const yy = Math.min(height - 1, Math.max(0, box.y0 + y + dy));
        for (let dx = -1; dx <= 1; dx += 1) {
          const xx = Math.min(width - 1, Math.max(0, box.x0 + x + dx));
          sum += luma(data, (yy * width + xx) * 4);
        }
      }
      result[y * w + x] = sum / 9;
    }
  }
  return result;
}

function median(values) {
  if (!values.length) return 0;
  const sorted = Float32Array.from(values).sort();
  return sorted[Math.floor(sorted.length / 2)];
}

// Robust grain level of the fine band inside the fully opaque matte.
export function grainLevel(data, alpha, width, height, box = maskBounds(alpha, width, height)) {
  if (!box) return 0;
  const w = box.x1 - box.x0 + 1;
  const inside = [];
  const total = w * (box.y1 - box.y0 + 1);
  // A sample (≤ 200k pixels) is plenty for a robust level, even at 12 MP.
  const stride = Math.max(1, Math.floor(total / SAMPLE_LIMIT));
  for (let index = 0; index < total; index += stride) {
    const x = box.x0 + (index % w);
    const y = box.y0 + Math.floor(index / w);
    if (alpha[y * width + x] < 0.999) continue;
    let sum = 0;
    for (let dy = -1; dy <= 1; dy += 1) {
      const yy = Math.min(height - 1, Math.max(0, y + dy));
      for (let dx = -1; dx <= 1; dx += 1) {
        sum += luma(data, (yy * width + Math.min(width - 1, Math.max(0, x + dx))) * 4);
      }
    }
    inside.push(luma(data, (y * width + x) * 4) - sum / 9);
  }
  if (inside.length < 64) return 0;
  const centre = median(inside);
  return MAD_TO_SIGMA * median(inside.map((value) => Math.abs(value - centre)));
}

// Small, seeded PRNG: the same photo gets the same grain on every export.
export function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(random) {
  let u = 0;
  while (u === 0) u = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
}

/**
 * Finish an up-scaled result in place (generatedRGBA) inside its matte.
 * sourceRGBA is the original crop at the same size; alpha is 0..1 per pixel.
 * Returns what was done, for the result badge and the diagnostics report.
 */
export function finishUpscaledFace(sourceRGBA, generatedRGBA, alpha, width, height, scale, seed = 1) {
  const report = { applied: false, scale: Math.round(scale * 100) / 100, sharpen: 0, sourceGrain: 0, resultGrain: 0, addedGrain: 0 };
  if (!(scale >= MIN_UPSCALE) || !Number.isFinite(scale)) return report;
  const box = maskBounds(alpha, width, height);
  if (!box) return report;

  const sharpen = Math.min(MAX_SHARPEN, Math.max(0, (scale - 1) * SHARPEN_PER_SCALE));
  if (sharpen > 0.01) {
    const w = box.x1 - box.x0 + 1;
    const mean = boxMeanLuma(generatedRGBA, width, height, box);
    for (let y = box.y0; y <= box.y1; y += 1) {
      for (let x = box.x0; x <= box.x1; x += 1) {
        const weight = alpha[y * width + x];
        if (weight <= 0) continue;
        const index = (y * width + x) * 4;
        // Luma-only unsharp: colour edges do not fringe.
        const delta = sharpen * weight * (luma(generatedRGBA, index) - mean[(y - box.y0) * w + (x - box.x0)]);
        generatedRGBA[index] += delta;
        generatedRGBA[index + 1] += delta;
        generatedRGBA[index + 2] += delta;
      }
    }
    report.sharpen = Math.round(sharpen * 100) / 100;
  }

  const sourceGrain = grainLevel(sourceRGBA, alpha, width, height, box);
  const resultGrain = grainLevel(generatedRGBA, alpha, width, height, box);
  report.sourceGrain = Math.round(sourceGrain * 100) / 100;
  report.resultGrain = Math.round(resultGrain * 100) / 100;
  const deficit = Math.sqrt(Math.max(0, sourceGrain * sourceGrain - resultGrain * resultGrain));
  if (deficit >= MIN_GRAIN_DEFICIT && resultGrain < GRAIN_TRIGGER * sourceGrain) {
    const sigma = Math.min(MAX_GRAIN, deficit) / WHITE_NOISE_FINE_GAIN;
    const random = mulberry32(seed);
    for (let y = box.y0; y <= box.y1; y += 1) {
      for (let x = box.x0; x <= box.x1; x += 1) {
        const weight = alpha[y * width + x];
        if (weight <= 0) continue;
        const index = (y * width + x) * 4;
        const noise = gaussian(random) * sigma * weight;
        generatedRGBA[index] += noise;
        generatedRGBA[index + 1] += noise;
        generatedRGBA[index + 2] += noise;
      }
    }
    report.addedGrain = Math.round(Math.min(MAX_GRAIN, deficit) * 100) / 100;
  }
  report.applied = report.sharpen > 0 || report.addedGrain > 0;
  return report;
}

export function describeNativeFinish(report) {
  if (!report || typeof report !== "object" || !report.applied) return "";
  const decimal = (value) => Number(value).toFixed(1).replace(".", ",");
  const parts = [`Doğal çözünürlük: yüz ${decimal(report.scale)}× büyütüldü`];
  if (report.sharpen > 0) parts.push("büyütme yumuşaması giderildi");
  if (report.addedGrain > 0) parts.push(`fotoğrafın greni eklendi (${decimal(report.resultGrain)} → ${decimal(report.sourceGrain)})`);
  return parts.join(" · ");
}
