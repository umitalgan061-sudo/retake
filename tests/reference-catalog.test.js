import test from "node:test";
import assert from "node:assert/strict";
import {
  REFERENCE_CATALOG,
  getReferenceImageUrl,
  resolveReferenceApiBase
} from "../src/reference-catalog.js";

test("catalog contains the 20 existing database image IDs", () => {
  assert.equal(REFERENCE_CATALOG.length, 20);
  assert.equal(new Set(REFERENCE_CATALOG.map((item) => item.id)).size, 20);
  assert.ok(REFERENCE_CATALOG.every((item) => /^img_[a-f0-9]{20}$/.test(item.id)));
});

test("builds a same-origin database image URL", () => {
  assert.equal(
    getReferenceImageUrl("img_689e31cf6bf4dbf904cc", "", "https://retake.example"),
    "https://retake.example/api/images/img_689e31cf6bf4dbf904cc"
  );
});

test("supports a separately configured API origin", () => {
  assert.equal(
    getReferenceImageUrl("img_689e31cf6bf4dbf904cc", "https://api.retake.example", "https://retake.example"),
    "https://api.retake.example/api/images/img_689e31cf6bf4dbf904cc"
  );
});

test("rejects malformed image IDs and unsafe URL schemes", () => {
  assert.throws(() => getReferenceImageUrl("../private", "", "https://retake.example"), TypeError);
  assert.throws(() => resolveReferenceApiBase("file:///tmp", "https://retake.example"), /HTTP veya HTTPS/);
});
