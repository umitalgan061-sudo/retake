// Pure geometry helpers for fitting the head to the new face and for sending
// the target crop to the local engine without needless resampling.

export const MIN_TARGET_HEIGHT = 1024;
export const MAX_TARGET_HEIGHT = 2048;
export const MAX_TARGET_WIDTH = 1600;
export const MAX_HEAD_FIT_CHANGE = 0.15;
export const MIN_SHAPE_REFERENCES = 5;
// Differences below this are within the mesh's measurement noise.
export const HEAD_FIT_DEAD_ZONE = 0.015;

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
  { damping = 1.0, maxChange = MAX_HEAD_FIT_CHANGE, deadZone = HEAD_FIT_DEAD_ZONE } = {}
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

/**
 * Faces for the picker: those both detectors agree on first (a lone mesh
 * "face" in a texture should not be preselected), then the largest.
 */
export function orderDetectedFaces(faces) {
  return [...faces].sort((a, b) =>
    (Number(Boolean(b.confirmed)) - Number(Boolean(a.confirmed))) ||
    ((b.width * b.height) - (a.width * a.height)));
}

/** Share of the smaller rectangle covered by the other (1 = one inside the other). */
export function rectOverlap(a, b) {
  const left = Math.max(a.x, b.x);
  const top = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.width, b.x + b.width);
  const bottom = Math.min(a.y + a.height, b.y + b.height);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const smaller = Math.min(a.width * a.height, b.width * b.height);
  return smaller > 0 ? intersection / smaller : 0;
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

// ---------------------------------------------------------------------------
// Jawline: make the jaw/chin contour in the photo follow the reference person.

// MediaPipe face-mesh face-oval points from the left cheek (234) along the jaw
// and chin (152) to the right cheek (454).
export const JAW_INDICES = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454];
// Points that must not move: eye corners, nose tip, mouth corners, nose bridge.
export const JAW_ANCHOR_INDICES = [33, 263, 1, 61, 291, 168, 13, 14];
const LEFT_EYE = 33;
const RIGHT_EYE = 263;
const NOSE_TIP = 1;
export const MIN_JAW_REFERENCES = 5;
export const MIN_JAW_FRONTALNESS = 0.55;
export const MAX_JAW_SHIFT = 0.07; // of the cheek-to-cheek width
// Shifts below this are within what a nod or tilt of the head does to the 2D
// contour; they are ignored so the jaw is not reshaped on pose noise.
export const JAW_DEAD_ZONE = 0.02;

function toPixels(point, width, height) {
  return { x: point.x * width, y: point.y * height };
}

/** Eye-based similarity frame: origin between the eyes, x along the eye line, unit = eye distance. */
export function eyeFrame(leftEye, rightEye) {
  const dx = rightEye.x - leftEye.x;
  const dy = rightEye.y - leftEye.y;
  const scale = Math.hypot(dx, dy);
  if (!(scale > 1e-6)) return null;
  return {
    origin: { x: (leftEye.x + rightEye.x) / 2, y: (leftEye.y + rightEye.y) / 2 },
    cos: dx / scale,
    sin: dy / scale,
    scale
  };
}

export function toFrame(frame, point) {
  const x = point.x - frame.origin.x;
  const y = point.y - frame.origin.y;
  return { x: (x * frame.cos + y * frame.sin) / frame.scale, y: (-x * frame.sin + y * frame.cos) / frame.scale };
}

export function fromFrame(frame, point) {
  const x = point.x * frame.scale;
  const y = point.y * frame.scale;
  return { x: frame.origin.x + x * frame.cos - y * frame.sin, y: frame.origin.y + x * frame.sin + y * frame.cos };
}

/** 1 for a frontal face, falling towards 0 as the head turns (nose offset from the eye midpoint). */
export function meshFrontalness(landmarks, width, height) {
  const left = landmarks?.[LEFT_EYE];
  const right = landmarks?.[RIGHT_EYE];
  const nose = landmarks?.[NOSE_TIP];
  if (!left || !right || !nose) return 0;
  const frame = eyeFrame(toPixels(left, width, height), toPixels(right, width, height));
  if (!frame) return 0;
  const offset = Math.abs(toFrame(frame, toPixels(nose, width, height)).x);
  return Math.max(0, Math.min(1, 1 - offset / 0.34));
}

/**
 * The jaw contour of one face in the eye frame (scale-, position- and
 * roll-free), or null when the mesh lacks the points.
 */
export function canonicalJaw(landmarks, width, height) {
  if (!Array.isArray(landmarks)) return null;
  const left = landmarks[LEFT_EYE];
  const right = landmarks[RIGHT_EYE];
  if (!left || !right) return null;
  const frame = eyeFrame(toPixels(left, width, height), toPixels(right, width, height));
  if (!frame) return null;
  const points = [];
  for (const index of JAW_INDICES) {
    const point = landmarks[index];
    if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) return null;
    points.push(toFrame(frame, toPixels(point, width, height)));
  }
  return points;
}

/** Weighted median of each contour point over the references; null if too few. */
export function summarizeReferenceJaw(samples, minimum = MIN_JAW_REFERENCES) {
  const valid = (Array.isArray(samples) ? samples : []).filter((sample) =>
    Array.isArray(sample?.points) && sample.points.length === JAW_INDICES.length &&
    Number.isFinite(sample.weight) && sample.weight > 0
  );
  if (valid.length < minimum) return null;
  return JAW_INDICES.map((_, index) => ({
    x: weightedMedian(valid.map((sample) => ({ value: sample.points[index].x, weight: sample.weight }))),
    y: weightedMedian(valid.map((sample) => ({ value: sample.points[index].y, weight: sample.weight })))
  }));
}

/** Pick the pixel-space points the jaw warp needs from a full mesh. */
export function jawPointsFromMesh(landmarks, width, height) {
  const pick = (indices) => indices.map((index) => landmarks?.[index] ? toPixels(landmarks[index], width, height) : null);
  const jaw = pick(JAW_INDICES);
  const anchors = pick(JAW_ANCHOR_INDICES);
  const eyes = pick([LEFT_EYE, RIGHT_EYE]);
  if ([...jaw, ...anchors, ...eyes].some((point) => !point || !Number.isFinite(point.x) || !Number.isFinite(point.y))) {
    return null;
  }
  return { jaw, anchors, eyes, frontalness: meshFrontalness(landmarks, width, height) };
}

/**
 * Where each jaw point of the photo should move so the contour follows the
 * reference person. Points near the cheeks move less than the chin; every
 * shift is capped; turned heads get no warp (one side of the jaw is hidden).
 * Coordinates are in whatever pixel space `target` is in.
 */
export function planJawWarp(
  target,
  referenceJaw,
  { strength = 0.8, maxShift = MAX_JAW_SHIFT, deadZone = JAW_DEAD_ZONE } = {}
) {
  if (!target || !Array.isArray(referenceJaw) || referenceJaw.length !== JAW_INDICES.length) return null;
  if (!(target.frontalness >= MIN_JAW_FRONTALNESS)) return null;
  const frame = eyeFrame(target.eyes[0], target.eyes[1]);
  if (!frame) return null;
  const cheekWidth = Math.hypot(target.jaw[0].x - target.jaw.at(-1).x, target.jaw[0].y - target.jaw.at(-1).y);
  const limit = cheekWidth * maxShift;
  // Turned slightly: trust the contour less.
  const confidence = Math.min(1, (target.frontalness - MIN_JAW_FRONTALNESS) / (1 - MIN_JAW_FRONTALNESS) * 1.6);
  const last = JAW_INDICES.length - 1;
  const controls = target.jaw.map((point, index) => {
    const desired = fromFrame(frame, referenceJaw[index]);
    // 0.35 at the cheeks, 1 at the chin.
    const along = 1 - Math.abs(index - last / 2) / (last / 2);
    const weight = strength * confidence * (0.35 + 0.65 * along);
    let dx = (desired.x - point.x) * weight;
    let dy = (desired.y - point.y) * weight;
    let length = Math.hypot(dx, dy);
    // Soft threshold: subtract the pose-noise floor, keep the direction.
    const kept = Math.max(0, length - cheekWidth * deadZone);
    if (length > 0) {
      dx *= kept / length;
      dy *= kept / length;
      length = kept;
    }
    if (length > limit) {
      dx *= limit / length;
      dy *= limit / length;
    }
    const round = (value) => Math.round(value * 100) / 100;
    return { from: [round(point.x), round(point.y)], to: [round(point.x + dx), round(point.y + dy)] };
  });
  const moved = controls.some((control) => Math.hypot(control.to[0] - control.from[0], control.to[1] - control.from[1]) >= 0.75);
  if (!moved) return null;
  return {
    controls,
    anchors: target.anchors.map((point) => [Math.round(point.x * 100) / 100, Math.round(point.y * 100) / 100])
  };
}
