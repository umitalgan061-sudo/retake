const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

export function getSimilarityTransform(sourceLeft, sourceRight, targetLeft, targetRight) {
  const points = [sourceLeft, sourceRight, targetLeft, targetRight];
  if (points.some((point) => !point || !Number.isFinite(point.x) || !Number.isFinite(point.y))) {
    throw new TypeError("Eye alignment points must contain finite x/y coordinates.");
  }

  const sourceDx = sourceRight.x - sourceLeft.x;
  const sourceDy = sourceRight.y - sourceLeft.y;
  const sourceDistance = Math.hypot(sourceDx, sourceDy);
  const targetDx = targetRight.x - targetLeft.x;
  const targetDy = targetRight.y - targetLeft.y;
  const targetDistance = Math.hypot(targetDx, targetDy);

  if (sourceDistance < 1 || targetDistance < 1) {
    throw new RangeError("Eye distance is too small for face alignment.");
  }

  const scale = targetDistance / sourceDistance;
  const angle = Math.atan2(sourceDy, sourceDx);
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const a = scale * cos;
  const b = -scale * sin;
  const c = scale * sin;
  const d = scale * cos;
  const sourceMidX = (sourceLeft.x + sourceRight.x) / 2;
  const sourceMidY = (sourceLeft.y + sourceRight.y) / 2;
  const targetMidX = (targetLeft.x + targetRight.x) / 2;
  const targetMidY = (targetLeft.y + targetRight.y) / 2;

  return {
    a,
    b,
    c,
    d,
    e: targetMidX - a * sourceMidX - c * sourceMidY,
    f: targetMidY - b * sourceMidX - d * sourceMidY,
    scale
  };
}

export function calculateReferenceWeight(metrics) {
  const values = [
    metrics?.faceAreaRatio,
    metrics?.frontalness,
    metrics?.sharpness,
    metrics?.exposure
  ];
  if (values.some((value) => !Number.isFinite(value))) {
    throw new TypeError("Reference quality metrics must be finite numbers.");
  }

  const area = clamp(metrics.faceAreaRatio / 0.08, 0, 1);
  const frontalness = clamp(metrics.frontalness, 0, 1);
  const sharpness = clamp(metrics.sharpness, 0, 1);
  const exposure = clamp(metrics.exposure, 0, 1);
  const score = area * 0.20 + frontalness * 0.35 + sharpness * 0.30 + exposure * 0.15;

  // Every detected face remains in the identity profile. Better views receive
  // two or three votes instead of allowing one blurry/oblique image to dominate.
  return clamp(1 + Math.round(score * 2), 1, 3);
}

export function weightedAverageRGBA(images) {
  if (!Array.isArray(images) || images.length === 0) {
    throw new TypeError("At least one aligned reference image is required.");
  }

  const length = images[0]?.data?.length;
  if (!Number.isInteger(length) || length === 0 || length % 4 !== 0) {
    throw new TypeError("Reference pixel data must be a non-empty RGBA buffer.");
  }

  const normalized = images.map((image) => {
    if (!image?.data || image.data.length !== length ||
        !Number.isFinite(image.weight) || image.weight <= 0) {
      throw new TypeError("Reference buffers must have equal RGBA size and positive weights.");
    }
    return { data: image.data, weight: image.weight };
  });

  const totalWeight = normalized.reduce((sum, image) => sum + image.weight, 0);
  const output = new Uint8ClampedArray(length);
  for (let index = 0; index < length; index += 1) {
    let value = 0;
    for (const image of normalized) {
      value += image.data[index] * image.weight;
    }
    output[index] = Math.round(value / totalWeight);
  }
  return output;
}
