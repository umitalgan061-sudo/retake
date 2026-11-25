import test from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULT_QUALITY_DEPTH, ENGINE_DEFAULT_DEPTH, QUALITY_DEPTHS,
  depthLabel, depthMinutes, depthTimeoutMs, describeDepthDuration, normalizeDepth, requestedDepth
} from "../src/depth.js";

test("deep is the default, unknown values fall back", () => {
  assert.equal(DEFAULT_QUALITY_DEPTH, "deep");
  assert.equal(normalizeDepth("max"), "max");
  for (const bad of [undefined, null, "", "ultra", 3, "__proto__", "toString"]) {
    assert.equal(normalizeDepth(bad), "deep");
  }
});

test("old engines and self-tests always get the seven-minute depth", () => {
  assert.equal(ENGINE_DEFAULT_DEPTH, "standard");
  assert.equal(requestedDepth("max", { supported: false }), "standard");
  assert.equal(requestedDepth("max", { supported: true, benchmark: true }), "standard");
  assert.equal(requestedDepth("max", { supported: true }), "max");
  assert.equal(requestedDepth("bogus", { supported: true }), "deep");
});

test("the browser waits longer than every engine depth needs", () => {
  let previous = 0;
  for (const depth of ["standard", "deep", "max"]) {
    const timeout = depthTimeoutMs(depth);
    assert.ok(timeout > previous, depth);
    assert.ok(timeout >= QUALITY_DEPTHS[depth].minutes[1] * 60 * 1000, depth);
    previous = timeout;
  }
  // An engine may ask for longer, never for a uselessly short wait.
  const health = { deep: { client_timeout_seconds: 2 * 60 * 60 }, max: { client_timeout_seconds: 5 } };
  assert.equal(depthTimeoutMs("deep", health), 2 * 60 * 60 * 1000);
  assert.equal(depthTimeoutMs("max", health), QUALITY_DEPTHS.max.timeoutSeconds * 1000);
});

test("durations come from the engine when sane", () => {
  assert.deepEqual(depthMinutes("deep", { deep: { expected_minutes: [20, 30] } }), [20, 30]);
  assert.deepEqual(depthMinutes("deep", { deep: { expected_minutes: [30, 20] } }), QUALITY_DEPTHS.deep.minutes);
  assert.deepEqual(depthMinutes("deep", { deep: { expected_minutes: ["a", 2] } }), QUALITY_DEPTHS.deep.minutes);
  assert.equal(depthLabel("max"), "En yüksek");
  assert.match(describeDepthDuration("deep"), /^Derin · yaklaşık 25–40 dakika$/);
});
