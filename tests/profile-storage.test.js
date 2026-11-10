import test from "node:test";
import assert from "node:assert/strict";
import { createReferenceSetKey } from "../src/profile-storage.js";

test("creates the same reference-set key for identical files", async () => {
  const references = [
    { file: "face-01.jpg", blob: new Blob(["first"], { type: "image/jpeg" }) },
    { file: "face-02.jpg", blob: new Blob(["second"], { type: "image/jpeg" }) }
  ];
  assert.equal(
    await createReferenceSetKey(references),
    await createReferenceSetKey(references)
  );
});

test("invalidates the profile key when a reference image changes", async () => {
  const first = [
    { file: "face-01.jpg", blob: new Blob(["first"], { type: "image/jpeg" }) }
  ];
  const changed = [
    { file: "face-01.jpg", blob: new Blob(["changed"], { type: "image/jpeg" }) }
  ];
  assert.notEqual(await createReferenceSetKey(first), await createReferenceSetKey(changed));
});

test("invalidates the profile key when reference membership changes", async () => {
  const one = [
    { file: "face-01.jpg", blob: new Blob(["first"]) }
  ];
  const two = [
    ...one,
    { file: "face-02.jpg", blob: new Blob(["second"]) }
  ];
  assert.notEqual(await createReferenceSetKey(one), await createReferenceSetKey(two));
});

test("rejects an empty reference set", async () => {
  await assert.rejects(createReferenceSetKey([]), /boş olamaz/);
});
