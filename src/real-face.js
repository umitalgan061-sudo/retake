// Real-photo face: pick the reference photo whose pose and expression best
// match the face being replaced, and prepare it (crop + 478-point mesh) for
// the engine, which warps it onto the target face triangle by triangle.

import { eyeFrame, toFrame } from "./head-geometry.js";

export const MESH_POINTS = 478;
// Eyes, brows, nose, mouth and jaw: what decides whether a photo can be
// warped onto another face without distortion.
export const POSE_KEY_POINTS = [1, 4, 6, 10, 33, 133, 152, 159, 145, 263, 362, 386, 374, 61, 291, 0, 17, 13, 14,
  78, 308, 234, 454, 58, 288, 105, 334, 98, 327, 168, 197, 195, 5, 2, 94];
// Generous: the mesh warp absorbs expression differences (a frown, a tilt),
// and the engine measures how well each candidate actually fits and drops
// those that do not.
export const MAX_POSE_DISTANCE = 0.13;
export const REFERENCE_CROP_MAX_EDGE = 1024;

/** Pose/expression signature: key points in the eye frame (position, size and roll free). */
export function poseDescriptor(landmarks, width, height) {
  if (!Array.isArray(landmarks) || landmarks.length < MESH_POINTS) return null;
  const left = landmarks[33];
  const right = landmarks[263];
  if (!left || !right) return null;
  const frame = eyeFrame({ x: left.x * width, y: left.y * height }, { x: right.x * width, y: right.y * height });
  if (!frame) return null;
  const points = [];
  for (const index of POSE_KEY_POINTS) {
    const point = landmarks[index];
    if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) return null;
    const local = toFrame(frame, { x: point.x * width, y: point.y * height });
    points.push(Math.round(local.x * 10000) / 10000, Math.round(local.y * 10000) / 10000);
  }
  return points;
}

export function descriptorDistance(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length || !a.length) return Infinity;
  let sum = 0;
  for (let index = 0; index < a.length; index += 2) sum += Math.hypot(a[index] - b[index], a[index + 1] - b[index + 1]);
  return sum / (a.length / 2);
}

export const REAL_FACE_CANDIDATES = 3;

/**
 * Indices of the references that can be warped onto the target face, best
 * first (at most `count`). A very different pose would warp badly, so only
 * references within `maxDistance` qualify. Sharper, better-lit references
 * (higher weight) win close calls. The engine warps every candidate and keeps
 * the one whose features line up best with the photo.
 */
export function rankRealReferences(descriptors, weights, target, maxDistance = MAX_POSE_DISTANCE, count = REAL_FACE_CANDIDATES) {
  const ranked = [];
  (Array.isArray(descriptors) ? descriptors : []).forEach((descriptor, index) => {
    const distance = descriptorDistance(descriptor, target);
    if (!(distance <= maxDistance)) return;
    const weight = Number(weights?.[index]) || 1;
    ranked.push({ index, score: distance * (1.15 - 0.05 * Math.min(3, weight)) });
  });
  return ranked.sort((a, b) => a.score - b.score).slice(0, Math.max(0, count)).map((item) => item.index);
}

/** Index of the best reference for the target face, or -1 if none is close enough. */
export function pickRealReference(descriptors, weights, target, maxDistance = MAX_POSE_DISTANCE) {
  return rankRealReferences(descriptors, weights, target, maxDistance, 1)[0] ?? -1;
}

/** Crop around the face (1.8× its size) at no more than REFERENCE_CROP_MAX_EDGE px. */
export function referenceCrop(landmarks, width, height, maxEdge = REFERENCE_CROP_MAX_EDGE) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const point of landmarks) {
    minX = Math.min(minX, point.x * width); maxX = Math.max(maxX, point.x * width);
    minY = Math.min(minY, point.y * height); maxY = Math.max(maxY, point.y * height);
  }
  const size = Math.max(maxX - minX, maxY - minY) * 1.8;
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const x = Math.max(0, Math.floor(cx - size / 2));
  const y = Math.max(0, Math.floor(cy - size / 2));
  const w = Math.min(width - x, Math.ceil(size));
  const h = Math.min(height - y, Math.ceil(size));
  const scale = Math.min(1, maxEdge / Math.max(w, h));
  return { x, y, width: w, height: h, scale, outWidth: Math.max(1, Math.round(w * scale)), outHeight: Math.max(1, Math.round(h * scale)) };
}

/** Mesh (normalized, full image) → pixel pairs in another space. */
export function meshToPixels(landmarks, width, height, map) {
  return landmarks.slice(0, MESH_POINTS).map((point) => {
    const mapped = map({ x: point.x * width, y: point.y * height });
    return [Math.round(mapped.x * 100) / 100, Math.round(mapped.y * 100) / 100];
  });
}
