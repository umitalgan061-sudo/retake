// Pose-free 3D lower-face shape: jaw, chin and cheeks of the reference person.
//
// The 2D jaw plan (head-geometry.js planJawWarp) compares contours in an
// eye-aligned frame. A nod moves the chin relative to the eyes and a turned
// head foreshortens one side, so that plan needs a dead zone, a frontalness
// threshold and weak cheek weights. MediaPipe's 478-point mesh carries depth:
// here every reference is aligned in 3D on rigid points (eye corners, nose
// bridge, nose, forehead), which removes yaw, pitch, roll and scale. The
// person's jaw, chin and cheek contour is the weighted median of the aligned
// references. For the photo, the same rigid alignment gives its 3D pose; the
// person's contour is projected into that pose, so the target contour can
// follow it even on turned heads. A wider or narrower mouth opening than in
// the references moves the mandible, not the identity: it is compensated.
//
// Output uses the existing engine format {controls, anchors}; the engine's
// variants scale it and its hard cap limits any displacement.

import { JAW_INDICES, JAW_ANCHOR_INDICES, weightedMedian } from "./head-geometry.js";

export const FACE_SHAPE_VERSION = 1;
// Do not move with expression or the jaw.
export const RIGID_INDICES = [33, 133, 263, 362, 168, 6, 197, 195, 5, 4, 1, 10, 151, 9];
const UPPER_INNER_LIP = 13;
const LOWER_INNER_LIP = 14;
export const MIN_SHAPE_REFERENCES = 5;
// Rigid-fit residual (of the template RMS) above which a mesh is not trusted.
export const MAX_RIGID_RESIDUAL = 0.09;
export const SHAPE_MAX_SHIFT = 0.10; // of the cheek-to-cheek width (engine cap: 0.10 of face width)
export const SHAPE_DEAD_ZONE = 0.008;
export const CHEEK_WEIGHT = 0.75; // 0.75 at the cheeks rising to 1 at the chin
export const NECK_FOLLOW = 0.6;
// Indices into JAW_INDICES of the jaw angles whose sideways move the neck follows.
const NECK_FROM = [4, 5, 15, 16];

const point3 = (landmark, width, height) => (
  landmark && Number.isFinite(landmark.x) && Number.isFinite(landmark.y)
    ? [landmark.x * width, landmark.y * height, (Number.isFinite(landmark.z) ? landmark.z : 0) * width]
    : null
);

/** Pixel-space 3D points the shape plan needs from one full mesh, or null. */
export function shapeSampleFromMesh(landmarks, width, height) {
  if (!Array.isArray(landmarks) || landmarks.length < 468) return null;
  const pick = (indices) => indices.map((index) => point3(landmarks[index], width, height));
  const rigid = pick(RIGID_INDICES);
  const jaw = pick(JAW_INDICES);
  const lips = pick([UPPER_INNER_LIP, LOWER_INNER_LIP]);
  const anchors = pick(JAW_ANCHOR_INDICES);
  if ([...rigid, ...jaw, ...lips, ...anchors].some((point) => !point || point.some((value) => !Number.isFinite(value)))) return null;
  return { rigid, jaw, lips, anchors };
}

/** Map a full-image sample into another pixel space (crop canvas): x/y by `map`, depth by `scale`. */
export function mapShapeSample(sample, map, scale) {
  if (!sample) return null;
  const convert = (point) => {
    const moved = map({ x: point[0], y: point[1] });
    return [moved.x, moved.y, point[2] * scale];
  };
  return {
    rigid: sample.rigid.map(convert),
    jaw: sample.jaw.map(convert),
    lips: sample.lips.map(convert),
    anchors: sample.anchors.map(convert)
  };
}

// ---------------------------------------------------------------- 3D algebra

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);
const centroid = (points) => mul(points.reduce(add, [0, 0, 0]), 1 / points.length);
const apply = (matrix, v) => [
  matrix[0][0] * v[0] + matrix[0][1] * v[1] + matrix[0][2] * v[2],
  matrix[1][0] * v[0] + matrix[1][1] * v[1] + matrix[1][2] * v[2],
  matrix[2][0] * v[0] + matrix[2][1] * v[1] + matrix[2][2] * v[2]
];
const transpose = (m) => [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]];

/** Eigenvector of the largest eigenvalue of a symmetric 4×4 matrix (cyclic Jacobi). */
function dominantEigenvector(input) {
  const a = input.map((row) => row.slice());
  const v = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < 4; p++) for (let q = p + 1; q < 4; q++) off += a[p][q] * a[p][q];
    if (off < 1e-22) break;
    for (let p = 0; p < 4; p++) {
      for (let q = p + 1; q < 4; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 4; k++) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 4; k++) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 4; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  let best = 0;
  for (let i = 1; i < 4; i++) if (a[i][i] > a[best][best]) best = i;
  return [v[0][best], v[1][best], v[2][best], v[3][best]];
}

/**
 * Least-squares similarity B ≈ s·R·A + t (Horn's quaternion method).
 * Returns {s, R, t, residual} with residual = RMS error / RMS spread of B.
 */
export function similarity3d(source, target) {
  if (!Array.isArray(source) || source.length !== target?.length || source.length < 4) return null;
  const ca = centroid(source);
  const cb = centroid(target);
  const a = source.map((p) => sub(p, ca));
  const b = target.map((p) => sub(p, cb));
  const spreadA = Math.sqrt(a.reduce((sum, p) => sum + p[0] ** 2 + p[1] ** 2 + p[2] ** 2, 0) / a.length);
  const spreadB = Math.sqrt(b.reduce((sum, p) => sum + p[0] ** 2 + p[1] ** 2 + p[2] ** 2, 0) / b.length);
  if (!(spreadA > 1e-9) || !(spreadB > 1e-9)) return null;
  const m = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < a.length; i++) for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) m[r][c] += a[i][r] * b[i][c];
  const [[sxx, sxy, sxz], [syx, syy, syz], [szx, szy, szz]] = m;
  const n = [
    [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
    [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
    [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
    [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz]
  ];
  const [w, x, y, z] = dominantEigenvector(n);
  const R = [
    [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z]
  ];
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += apply(R, a[i]).reduce((sum, value, k) => sum + value * b[i][k], 0);
  const s = dot / a.reduce((sum, p) => sum + p[0] ** 2 + p[1] ** 2 + p[2] ** 2, 0);
  if (!(s > 0)) return null;
  const t = sub(cb, mul(apply(R, ca), s));
  let error = 0;
  for (let i = 0; i < source.length; i++) {
    const mapped = add(mul(apply(R, source[i]), s), t);
    const d = sub(mapped, target[i]);
    error += d[0] ** 2 + d[1] ** 2 + d[2] ** 2;
  }
  return { s, R, t, residual: Math.sqrt(error / source.length) / spreadB };
}

const toCanonical = (fit, point) => mul(apply(transpose(fit.R), sub(point, fit.t)), 1 / fit.s);
const fromCanonical = (fit, point) => add(mul(apply(fit.R, point), fit.s), fit.t);

function normalizeTemplate(points) {
  const c = centroid(points);
  const centered = points.map((p) => sub(p, c));
  const spread = Math.sqrt(centered.reduce((sum, p) => sum + p[0] ** 2 + p[1] ** 2 + p[2] ** 2, 0) / centered.length);
  return centered.map((p) => mul(p, 1 / spread));
}

/**
 * The person's pose-free jaw/chin/cheek contour from reference samples
 * [{sample, weight}], or null with too few trustworthy meshes.
 */
export function summarizeFaceShape3d(entries, minimum = MIN_SHAPE_REFERENCES) {
  const valid = (Array.isArray(entries) ? entries : []).filter((entry) =>
    entry?.sample && Array.isArray(entry.sample.rigid) && entry.sample.rigid.length === RIGID_INDICES.length &&
    Array.isArray(entry.sample.jaw) && entry.sample.jaw.length === JAW_INDICES.length &&
    Number.isFinite(entry.weight) && entry.weight > 0
  );
  if (valid.length < minimum) return null;
  // Generalized Procrustes on the rigid points. Start from the reference that
  // agrees best with the others (median residual), so one broken mesh, however
  // heavily weighted, cannot become the template.
  const medianOf = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
  const seed = valid.map((entry) => {
    const candidate = normalizeTemplate(entry.sample.rigid);
    return { candidate, score: medianOf(valid.map((other) => similarity3d(candidate, other.sample.rigid)?.residual ?? Infinity)) };
  }).sort((a, b) => a.score - b.score)[0];
  let template = seed.candidate;
  let fits = [];
  for (let round = 0; round < 4; round++) {
    fits = valid.map((entry) => similarity3d(template, entry.sample.rigid));
    const kept = valid.map((entry, index) => ({ entry, fit: fits[index] })).filter((item) => item.fit && item.fit.residual <= MAX_RIGID_RESIDUAL);
    if (kept.length < minimum) return null;
    const aligned = kept.map(({ entry, fit }) => entry.sample.rigid.map((point) => toCanonical(fit, point)));
    template = normalizeTemplate(RIGID_INDICES.map((_, i) => centroid(aligned.map((points) => points[i]))));
  }
  const used = [];
  valid.forEach((entry, index) => {
    const fit = similarity3d(template, entry.sample.rigid);
    if (fit && fit.residual <= MAX_RIGID_RESIDUAL) used.push({ entry, fit });
  });
  if (used.length < minimum) return null;
  const median = (values) => weightedMedian(values);
  const jaw = JAW_INDICES.map((_, i) => [0, 1, 2].map((axis) => median(used.map(({ entry, fit }) => ({
    value: toCanonical(fit, entry.sample.jaw[i])[axis], weight: entry.weight
  })))));
  const mouthOpen = median(used.map(({ entry, fit }) => ({
    value: Array.isArray(entry.sample.lips) && entry.sample.lips.length === 2
      ? norm(sub(toCanonical(fit, entry.sample.lips[1]), toCanonical(fit, entry.sample.lips[0])))
      : 0,
    weight: entry.weight
  })));
  const round = (value) => Math.round(value * 1e5) / 1e5;
  return {
    version: FACE_SHAPE_VERSION,
    template: template.map((p) => p.map(round)),
    jaw: jaw.map((p) => p.map(round)),
    mouthOpen: round(mouthOpen),
    references: used.length
  };
}

const unit2 = (x, y) => {
  const length = Math.hypot(x, y);
  return length > 1e-9 ? [x / length, y / length] : [0, 0];
};

/**
 * Where the photo's jaw/chin/cheek contour should move so it follows the
 * person, for the photo's own 3D pose. `target` is a shapeSampleFromMesh
 * result in the engine's pixel space (the crop canvas). Returns
 * {controls, anchors, method, unchanged, yaw, references}; null only when the
 * 3D fit itself cannot be trusted (the caller may then use the 2D plan).
 */
export function planShapeWarp(target, shape, {
  strength = 1.0, maxShift = SHAPE_MAX_SHIFT, deadZone = SHAPE_DEAD_ZONE,
  cheekWeight = CHEEK_WEIGHT, neckFollow = NECK_FOLLOW, bounds = null
} = {}) {
  if (!target || !shape || shape.version !== FACE_SHAPE_VERSION) return null;
  if (!Array.isArray(shape.template) || shape.template.length !== RIGID_INDICES.length) return null;
  if (!Array.isArray(shape.jaw) || shape.jaw.length !== JAW_INDICES.length) return null;
  const fit = similarity3d(shape.template, target.rigid);
  if (!fit || fit.residual > MAX_RIGID_RESIDUAL) return null;
  const last = JAW_INDICES.length - 1;
  const cheekWidth = Math.hypot(target.jaw[0][0] - target.jaw[last][0], target.jaw[0][1] - target.jaw[last][1]);
  if (!(cheekWidth > 1)) return null;
  // Expression: a more open mouth than the person's median lowers the mandible.
  const lips = target.lips.map((point) => toCanonical(fit, point));
  const open = norm(sub(lips[1], lips[0]));
  const jawDrop = Number.isFinite(shape.mouthOpen) ? open - shape.mouthOpen : 0;
  const drop = unit2(lips[1][0] - lips[0][0], lips[1][1] - lips[0][1]);
  const dropDirection = open > 1e-6 ? [drop[0], drop[1], 0] : [0, 1, 0];
  // Yaw: the side turned away is foreshortened and partly hidden; trust it less.
  const depth = (point) => apply(fit.R, point)[2];
  const leftDepth = depth(shape.jaw[0]);
  const rightDepth = depth(shape.jaw[last]);
  const canonicalWidth = norm(sub(shape.jaw[0], shape.jaw[last]));
  const asymmetry = canonicalWidth > 1e-9 ? Math.abs(leftDepth - rightDepth) / canonicalWidth : 0;
  const farWeight = Math.max(0, Math.min(1, 1 - (asymmetry - 0.15) / 0.45));
  const farIsLeft = leftDepth > rightDepth;
  const forward = apply(fit.R, [0, 0, 1]);
  const yaw = Math.round(Math.atan2(forward[0], Math.abs(forward[2]) || 1e-9) * 180 / Math.PI);

  const limit = cheekWidth * maxShift;
  const floor = cheekWidth * deadZone;
  const round = (value) => Math.round(value * 100) / 100;
  const shifts = [];
  const controls = target.jaw.map((point, index) => {
    const along = 1 - Math.abs(index - last / 2) / (last / 2); // 0 at the cheeks, 1 at the chin
    const onFarSide = farIsLeft ? index < last / 2 : index > last / 2;
    const visibility = onFarSide ? farWeight + (1 - farWeight) * along : 1;
    const person = add(shape.jaw[index], mul(dropDirection, jawDrop * along));
    const desired = fromCanonical(fit, person);
    const weight = strength * (cheekWeight + (1 - cheekWeight) * along) * visibility;
    let dx = (desired[0] - point[0]) * weight;
    let dy = (desired[1] - point[1]) * weight;
    let length = Math.hypot(dx, dy);
    const kept = Math.max(0, length - floor);
    if (length > 0) {
      dx *= kept / length;
      dy *= kept / length;
      length = kept;
    }
    if (length > limit) {
      dx *= limit / length;
      dy *= limit / length;
    }
    shifts.push([dx, dy]);
    return { from: [round(point[0]), round(point[1])], to: [round(point[0] + dx), round(point[1] + dy)] };
  });
  const anchors = target.anchors.map((point) => [round(point[0]), round(point[1])]);
  // A trusted fit that needs no change is an answer too ("already the
  // person's shape"); callers must not fall back to the pose-sensitive 2D plan.
  if (!controls.some((control) => Math.hypot(control.to[0] - control.from[0], control.to[1] - control.from[1]) >= 0.75)) {
    return { controls: [], anchors, method: "3d", unchanged: true, yaw, references: shape.references };
  }

  // The sides of the neck follow the jaw angles sideways, so a wider or
  // narrower jaw does not pinch into or bulge over an unchanged neck.
  const lateral = unit2(...apply(fit.R, [1, 0, 0]).slice(0, 2));
  const down = unit2(...apply(fit.R, [0, 1, 0]).slice(0, 2));
  const eyeMid = [(target.rigid[0][0] + target.rigid[2][0]) / 2, (target.rigid[0][1] + target.rigid[2][1]) / 2];
  const chin = target.jaw[last / 2];
  const faceDrop = Math.hypot(chin[0] - eyeMid[0], chin[1] - eyeMid[1]);
  const inside = (x, y) => !bounds || (x >= 0 && y >= 0 && x <= bounds.width && y <= bounds.height);
  if (neckFollow > 0) {
    for (const index of NECK_FROM) {
      const from = target.jaw[index];
      const sideways = (shifts[index][0] * lateral[0] + shifts[index][1] * lateral[1]) * neckFollow;
      if (Math.abs(sideways) < 0.5) continue;
      const x = from[0] + down[0] * faceDrop * 0.55;
      const y = from[1] + down[1] * faceDrop * 0.55;
      if (!inside(x, y)) continue;
      controls.push({ from: [round(x), round(y)], to: [round(x + lateral[0] * sideways), round(y + lateral[1] * sideways)] });
    }
  }
  return { controls, anchors, method: "3d", unchanged: false, yaw, references: shape.references };
}

/** One line for the diagnostics report (numbers only). */
export function describeShapePlan(plan) {
  if (!plan) return "";
  if (plan.method !== "3d") return "2B çene hattı (3B profil yok)";
  if (plan.unchanged) return `3B; ${plan.references} referans; baş açısı ${plan.yaw}°; zaten kişinin şeklinde`;
  const jaw = Math.min(plan.controls.length, 21);
  let largest = 0;
  let width = 0;
  if (plan.controls.length >= 21) {
    const first = plan.controls[0].from;
    const lastPoint = plan.controls[20].from;
    width = Math.hypot(first[0] - lastPoint[0], first[1] - lastPoint[1]);
  }
  for (const control of plan.controls.slice(0, jaw)) {
    largest = Math.max(largest, Math.hypot(control.to[0] - control.from[0], control.to[1] - control.from[1]));
  }
  const percent = width > 0 ? Math.round((largest / width) * 1000) / 10 : 0;
  return `3B; ${plan.references} referans; baş açısı ${plan.yaw}°; ${jaw}+${plan.controls.length - jaw} nokta; en büyük kayma %${percent}`;
}
