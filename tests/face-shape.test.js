import test from "node:test";
import assert from "node:assert/strict";
import { JAW_INDICES, JAW_ANCHOR_INDICES } from "../src/head-geometry.js";
import {
  RIGID_INDICES, mapShapeSample, planShapeWarp, shapeSampleFromMesh, similarity3d, summarizeFaceShape3d
} from "../src/face-shape.js";

// A synthetic head in model units (x right, y down, z away from the camera).
const RIGID_MODEL = {
  33: [-0.95, 0, 0.1], 133: [-0.35, 0.02, 0.05], 263: [0.95, 0, 0.1], 362: [0.35, 0.02, 0.05],
  168: [0, -0.05, -0.05], 6: [0, 0.15, -0.12], 197: [0, 0.35, -0.2], 195: [0, 0.5, -0.27],
  5: [0, 0.68, -0.33], 4: [0, 0.8, -0.38], 1: [0, 0.9, -0.4], 10: [0, -0.9, -0.02], 151: [0, -0.6, -0.08], 9: [0, -0.3, -0.06]
};
function jawModel({ chin = 0, width = 1, drop = 0 } = {}) {
  const last = JAW_INDICES.length - 1;
  return JAW_INDICES.map((_, index) => {
    const t = index / last; // 0 left cheek → 1 right cheek
    const angle = Math.PI * t;
    const along = 1 - Math.abs(index - last / 2) / (last / 2);
    const x = -Math.cos(angle) * 1.25 * (1 - (1 - width) * along);
    const y = 0.35 + Math.sin(angle) * (1.25 + chin) + drop * along;
    const z = 0.55 - Math.sin(angle) * 0.45;
    return [x, y, z];
  });
}
const ANCHOR_MODEL = { 33: RIGID_MODEL[33], 263: RIGID_MODEL[263], 1: RIGID_MODEL[1], 61: [-0.45, 1.15, -0.2], 291: [0.45, 1.15, -0.2], 168: RIGID_MODEL[168], 13: [0, 1.12, -0.28], 14: [0, 1.16, -0.28] };

function rotation(yaw, pitch, roll) {
  const [a, b, c] = [yaw, pitch, roll].map((deg) => deg * Math.PI / 180);
  const ry = [[Math.cos(a), 0, Math.sin(a)], [0, 1, 0], [-Math.sin(a), 0, Math.cos(a)]];
  const rx = [[1, 0, 0], [0, Math.cos(b), -Math.sin(b)], [0, Math.sin(b), Math.cos(b)]];
  const rz = [[Math.cos(c), -Math.sin(c), 0], [Math.sin(c), Math.cos(c), 0], [0, 0, 1]];
  const m = (p, q) => p.map((row) => [0, 1, 2].map((j) => row.reduce((sum, value, k) => sum + value * q[k][j], 0)));
  return m(rz, m(rx, ry));
}
const W = 1000;
const H = 1200;
function view(jaw, { yaw = 0, pitch = 0, roll = 0, scale = 120, cx = 500, cy = 520, mouth = 0, noise = 0, seed = 1 } = {}) {
  const R = rotation(yaw, pitch, roll);
  let state = seed;
  const jitter = () => { state = (state * 16807) % 2147483647; return ((state / 2147483647) - 0.5) * noise; };
  const project = (p) => {
    const r = [0, 1, 2].map((i) => R[i][0] * p[0] + R[i][1] * p[1] + R[i][2] * p[2]);
    return { x: (cx + scale * r[0] + jitter()) / W, y: (cy + scale * r[1] + jitter()) / H, z: (scale * r[2]) / W };
  };
  const landmarks = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  for (const [index, p] of Object.entries(RIGID_MODEL)) landmarks[index] = project(p);
  for (const [index, p] of Object.entries(ANCHOR_MODEL)) landmarks[index] = project(p);
  landmarks[14] = project([0, 1.16 + mouth, -0.28]);
  JAW_INDICES.forEach((index, i) => { landmarks[index] = project(jaw[i]); });
  return landmarks;
}
const POSES = [
  { yaw: 0 }, { yaw: 18, pitch: 5 }, { yaw: -15, pitch: -8 }, { yaw: 8, roll: 6 },
  { yaw: -22, pitch: 3, roll: -4 }, { pitch: 12 }, { yaw: 12, pitch: -12 }
];
function profileOf(jaw, options = {}) {
  return summarizeFaceShape3d(POSES.map((pose, index) => ({
    sample: shapeSampleFromMesh(view(jaw, { ...pose, noise: options.noise ?? 0.6, seed: index + 3, scale: 100 + index * 9 }), W, H),
    weight: 1 + (index % 3)
  })));
}
const targetSample = (jaw, pose) => shapeSampleFromMesh(view(jaw, pose), W, H);
const expectedPoints = (jaw, pose) => shapeSampleFromMesh(view(jaw, pose), W, H).jaw;

test("similarity3d recovers rotation, scale and translation", () => {
  const source = Object.values(RIGID_MODEL);
  const R = rotation(25, -10, 7);
  const target = source.map((p) => [0, 1, 2].map((i) => 3 * (R[i][0] * p[0] + R[i][1] * p[1] + R[i][2] * p[2]) + [10, -4, 2][i]));
  const fit = similarity3d(source, target);
  assert.ok(Math.abs(fit.s - 3) < 1e-6);
  assert.ok(fit.residual < 1e-9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) assert.ok(Math.abs(fit.R[i][j] - R[i][j]) < 1e-6);
});

test("the same person is not reshaped, whatever the head pose", () => {
  const person = jawModel();
  const shape = profileOf(person);
  assert.equal(shape.references, POSES.length);
  for (const pose of [{ yaw: 0 }, { yaw: 25 }, { yaw: -28, pitch: 10 }, { pitch: -15 }, { yaw: 15, roll: 10 }]) {
    const plan = planShapeWarp(targetSample(person, pose), shape);
    assert.equal(plan.unchanged, true, JSON.stringify(pose));
    assert.deepEqual(plan.controls, []);
  }
});

test("a different chin and jaw width are moved to the person's, in the photo's pose", () => {
  const person = jawModel({ chin: 0.14, width: 0.9 });
  const photo = jawModel();
  const shape = profileOf(person);
  const pose = { yaw: 0, pitch: 4, scale: 130 };
  const plan = planShapeWarp(targetSample(photo, pose), shape);
  assert.equal(plan.method, "3d");
  const expected = expectedPoints(person, pose);
  const chin = (JAW_INDICES.length - 1) / 2;
  const cheekWidth = Math.hypot(expected[0][0] - expected.at(-1)[0], expected[0][1] - expected.at(-1)[1]);
  const error = Math.hypot(plan.controls[chin].to[0] - expected[chin][0], plan.controls[chin].to[1] - expected[chin][1]);
  assert.ok(error < cheekWidth * 0.015, `chin error ${error}`);
  // Chin moves down (longer chin), jaw sides move inwards (narrower).
  assert.ok(plan.controls[chin].to[1] > plan.controls[chin].from[1] + 5);
  assert.ok(plan.controls[6].to[0] > plan.controls[6].from[0]);
  assert.ok(plan.controls[14].to[0] < plan.controls[14].from[0]);
  // Every shift is capped at 10 % of the cheek width.
  for (const control of plan.controls) {
    assert.ok(Math.hypot(control.to[0] - control.from[0], control.to[1] - control.from[1]) <= cheekWidth * 0.1 + 0.02);
  }
  assert.deepEqual(plan.anchors.length, JAW_ANCHOR_INDICES.length);
});

test("turned heads are reshaped too, the far side less than the near side", () => {
  const person = jawModel({ width: 0.88 });
  const shape = profileOf(person);
  const plan = planShapeWarp(targetSample(jawModel(), { yaw: 30 }), shape);
  assert.ok(plan);
  assert.ok(Math.abs(plan.yaw) >= 20);
  const shift = (index) => Math.hypot(plan.controls[index].to[0] - plan.controls[index].from[0], plan.controls[index].to[1] - plan.controls[index].from[1]);
  const frontal = planShapeWarp(targetSample(jawModel(), { yaw: 0 }), shape);
  const frontalShift = (index) => Math.hypot(frontal.controls[index].to[0] - frontal.controls[index].from[0], frontal.controls[index].to[1] - frontal.controls[index].from[1]);
  const left = shift(5) / frontalShift(5);
  const right = shift(15) / frontalShift(15);
  assert.ok(Math.min(left, right) < 0.8 * Math.max(left, right), `left ${left} right ${right}`);
});

test("an open mouth lowers the mandible instead of being read as a short chin", () => {
  const person = jawModel();
  const shape = profileOf(person);
  const open = 0.22;
  const photo = jawModel({ drop: open });
  const pose = { yaw: 5 };
  const sample = shapeSampleFromMesh(view(photo, { ...pose, mouth: open }), W, H);
  assert.equal(planShapeWarp(sample, shape).unchanged, true);
  // The same open mouth on a person with a longer chin still lengthens it.
  const longer = profileOf(jawModel({ chin: 0.15 }));
  const plan = planShapeWarp(sample, longer);
  const chin = (JAW_INDICES.length - 1) / 2;
  assert.ok(plan.controls[chin].to[1] > plan.controls[chin].from[1] + 5);
});

test("the neck follows the jaw angles sideways only", () => {
  const shape = profileOf(jawModel({ width: 0.86 }));
  const plan = planShapeWarp(targetSample(jawModel(), { yaw: 0 }), shape);
  const neck = plan.controls.slice(JAW_INDICES.length);
  assert.equal(neck.length, 4);
  for (const control of neck) {
    assert.ok(Math.abs(control.to[1] - control.from[1]) < 0.6);
    assert.ok(control.from[1] > plan.controls[5].from[1]);
  }
  const chinOnly = planShapeWarp(targetSample(jawModel(), { yaw: 0 }), profileOf(jawModel({ chin: 0.14 })));
  assert.ok(chinOnly.controls.length < JAW_INDICES.length + 4);
  const clipped = planShapeWarp(targetSample(jawModel(), { yaw: 0 }), shape, { bounds: { width: 1000, height: 700 } });
  assert.equal(clipped.controls.length, JAW_INDICES.length);
  assert.ok(planShapeWarp(targetSample(jawModel(), { yaw: 0 }), shape, { neckFollow: 0 }).controls.length === JAW_INDICES.length);
});

test("too few or untrustworthy references give no 3D shape", () => {
  const person = jawModel();
  const few = POSES.slice(0, 4).map((pose, index) => ({ sample: shapeSampleFromMesh(view(person, pose), W, H), weight: 1 + index }));
  assert.equal(summarizeFaceShape3d(few), null);
  const broken = view(person, { yaw: 5 });
  broken[1] = { x: 0.9, y: 0.1, z: 0.3 };
  broken[10] = { x: 0.1, y: 0.9, z: -0.3 };
  const entries = POSES.map((pose, index) => ({ sample: shapeSampleFromMesh(view(person, pose), W, H), weight: 1 }));
  entries.push({ sample: shapeSampleFromMesh(broken, W, H), weight: 50 });
  const shape = summarizeFaceShape3d(entries);
  assert.equal(shape.references, POSES.length);
  assert.equal(planShapeWarp(shapeSampleFromMesh(broken, W, H), shape), null);
  assert.equal(shapeSampleFromMesh(new Array(100).fill({ x: 0, y: 0, z: 0 }), W, H), null);
  assert.equal(planShapeWarp(targetSample(person, {}), { ...shape, version: 99 }), null);
});

test("mapping into the crop canvas scales depth with the canvas", () => {
  const sample = targetSample(jawModel(), { yaw: 10 });
  const mapped = mapShapeSample(sample, (p) => ({ x: (p.x - 100) * 2, y: (p.y - 50) * 2 }), 2);
  assert.deepEqual(mapped.rigid[0], [(sample.rigid[0][0] - 100) * 2, (sample.rigid[0][1] - 50) * 2, sample.rigid[0][2] * 2]);
  assert.equal(mapped.jaw.length, JAW_INDICES.length);
  assert.equal(RIGID_INDICES.length, mapped.rigid.length);
});
