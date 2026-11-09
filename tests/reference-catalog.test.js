import test from "node:test";
import assert from "node:assert/strict";
import {
  getReferenceImageUrl,
  parseReferenceManifest,
  REFERENCE_MANIFEST_URL
} from "../src/reference-catalog.js";
import { MAX_REFERENCE_FACES } from "../src/core.js";

const manifest = { images: Array.from({ length: 25 }, (_, index) => `${String(index + 1).padStart(2, "0")}.jpg`) };

test("loads the 25-image manifest from the local reference folder", () => {
  assert.equal(MAX_REFERENCE_FACES, 25);
  assert.equal(REFERENCE_MANIFEST_URL, "./reference-faces/manifest.json");
  assert.equal(parseReferenceManifest(manifest).length, 25);
});

test("builds a same-origin local reference image URL", () => {
  assert.equal(
    getReferenceImageUrl("01.jpg", "https://retake.example/reference-faces/"),
    "https://retake.example/reference-faces/01.jpg"
  );
});

test("uses names and labels supplied by the manifest", () => {
  const parsed = parseReferenceManifest({
    images: [
      { file: "01.png", name: "Önden" },
      ...manifest.images.slice(1)
    ]
  });
  assert.deepEqual(parsed[0], { file: "01.png", name: "Önden" });
});

test("rejects missing, extra, duplicate, or unsafe manifest entries", () => {
  assert.throws(() => parseReferenceManifest({ images: manifest.images.slice(0, 24) }), /tam olarak 25/);
  assert.throws(() => parseReferenceManifest({ images: [...manifest.images, "26.jpg"] }), /tam olarak 25/);
  assert.throws(() => parseReferenceManifest({ images: ["../private.jpg", ...manifest.images.slice(1)] }), /dosya adı geçersiz/);
  assert.throws(() => parseReferenceManifest({ images: ["01.jpg", "01.jpg", ...manifest.images.slice(2)] }), /yinelenen/);
  assert.throws(() => getReferenceImageUrl("../private.jpg", "https://retake.example/reference-faces/"), TypeError);
});
