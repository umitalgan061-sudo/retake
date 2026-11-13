// Pure geometry helpers for fitting the head to the new face and for sending
// the target crop to the local engine without needless resampling.

export const MIN_TARGET_HEIGHT = 1024;
export const MAX_TARGET_HEIGHT = 2048;
export const MAX_TARGET_WIDTH = 1600;
export const MAX_HEAD_FIT_CHANGE = 0.08;
export const MIN_SHAPE_REFERENCES = 5;
// Differences below this are within the mesh's measurement noise.
export const HEAD_FIT_DEAD_ZONE = 0.02;

const FOREHEAD = 10;
const CHIN = 152;
const LEFT_CHEEK = 234;
const RIGHT_CHEEK = 454;

function distance3d(a, b, width, height) {
  // MediaPipe's z uses the same scale as x, so it is multiplied by the width.
  // Measuring in 3D keeps the ratio stable when the head is turned or tilted.
  return Math.hypot(
    (a.x - b.x) * width,
    (a.y - b.y) * height,
    ((Number.isFinite(a.z) ? a.z : 0) - (Number.isFinite(b.z) ? b.z : 0)) * width
  );
}

/** Face length ÷ face width from the 478-point mesh, or null if unusable. */
export function measureFaceShapeRatio(landmarks, imageWidth, imageHeight) {
  if (!Array.isArray(landmarks) || !(imageWidth > 0) || !(imageHeight > 0)) return null;
  const points = [FOREHEAD, CHIN, LEFT_CHEEK, RIGHT_CHEEK].map((index) => landmarks[index]);
  if (points.some((point) => !point || !Number.isFinite(point.x) || !Number.isFinite(point.y))) return null;
  const [forehead, chin, leftCheek, rightCheek] = points;
  const length = distance3d(forehead, chin, imageWidth, imageHeight);
  const breadth = distance3d(leftCheek, rightCheek, imageWidth, imageHeight);
  if (!(breadth > 1) || !(length > 1)) return null;
  const ratio = length / breadth;
  return ratio > 0.8 && ratio < 2.2 ? ratio : null;
}

export function weightedMedian(items) {
  const valid = (Array.isArray(items) ? items : [])
    .filter((item) => Number.isFinite(item?.value) && Number.isFinite(item?.weight) && item.weight > 0)
    .sort((a, b) => a.value - b.value);
  if (valid.length === 0) return null;
  const half = valid.reduce((sum, item) => sum + item.weight, 0) / 2;
  let running = 0;
  for (const item of valid) {
    running += item.weight;
    if (running >= half) return item.value;
  }
  return valid[valid.length - 1].value;
}

/**
 * The reference person's typical face proportion, from every reference that
 * has a full mesh. Returns null when too few references carry one.
 */
export function summarizeReferenceShape(samples) {
  const valid = (Array.isArray(samples) ? samples : [])
    .filter((item) => Number.isFinite(item?.value) && Number.isFinite(item?.weight) && item.weight > 0);
  return valid.length >= MIN_SHAPE_REFERENCES ? weightedMedian(valid) : null;
}

/**
 * How much the head in the photo must be lengthened/shortened and
 * narrowed/widened so its proportions match the reference person's.
 * Area is preserved (scaleX * scaleY ≈ 1) and each axis is capped.
 */
export function estimateHeadFit(
  referenceRatio,
  targetRatio,
  { damping = 0.8, maxChange = MAX_HEAD_FIT_CHANGE, deadZone = HEAD_FIT_DEAD_ZONE } = {}
) {
  if (!(referenceRatio > 0) || !(targetRatio > 0) ||
      !Number.isFinite(referenceRatio) || !Number.isFinite(targetRatio)) {
    return { scaleX: 1, scaleY: 1 };
  }
  // Leave the photo alone when the head already has the right proportions.
  if (Math.abs(referenceRatio / targetRatio - 1) < deadZone) return { scaleX: 1, scaleY: 1 };
  const change = Math.pow(referenceRatio / targetRatio, damping);
  const limit = (value) => Math.min(1 + maxChange, Math.max(1 - maxChange, value));
  const round = (value) => Math.round(value * 10000) / 10000;
  return {
    scaleX: round(limit(1 / Math.sqrt(change))),
    scaleY: round(limit(Math.sqrt(change)))
  };
}

/**
 * Size at which the target crop is sent to the local engine. Crops inside the
 * supported range go through pixel-for-pixel, so the photo around the face is
 * never blurred by a down/up-scale round trip.
 */
export function getTargetRenderSize(cropWidth, cropHeight) {
  if (!Number.isInteger(cropWidth) || !Number.isInteger(cropHeight) || cropWidth <= 0 || cropHeight <= 0) {
    throw new TypeError("Crop dimensions must be positive integers.");
  }
  let scale = 1;
  if (cropHeight < MIN_TARGET_HEIGHT) scale = MIN_TARGET_HEIGHT / cropHeight;
  if (cropHeight > MAX_TARGET_HEIGHT) scale = MAX_TARGET_HEIGHT / cropHeight;
  if (cropWidth * scale > MAX_TARGET_WIDTH) scale = MAX_TARGET_WIDTH / cropWidth;
  return {
    width: Math.max(1, Math.round(cropWidth * scale)),
    height: Math.max(1, Math.round(cropHeight * scale)),
    scale
  };
}

/** Intersection-over-union of two {x, y, width, height} rectangles. */
export function rectIoU(a, b) {
  const left = Math.max(a.x, b.x);
  const top = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.width, b.x + b.width);
  const bottom = Math.min(a.y + a.height, b.y + b.height);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const union = a.width * a.height + b.width * b.height - intersection;
  return union > 0 ? intersection / union : 0;
}

/** Map mesh landmarks found inside a normalized sub-rectangle back to the full image. */
export function mapLandmarksFromRegion(landmarks, region) {
  return landmarks.map((point) => ({
    x: region.x + point.x * region.width,
    y: region.y + point.y * region.height,
    // z shares x's scale, which shrinks with the region width.
    z: Number.isFinite(point.z) ? point.z * region.width : 0
  }));
}
