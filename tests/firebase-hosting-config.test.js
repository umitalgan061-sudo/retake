import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const firebase = JSON.parse(await readFile(new URL("../firebase.json", import.meta.url), "utf8"));
const firebaserc = JSON.parse(await readFile(new URL("../.firebaserc", import.meta.url), "utf8"));

test("maps only the Retake Hosting target to the retake06 site", () => {
  assert.equal(firebaserc.projects.default, "hafize-23465");
  assert.deepEqual(firebaserc.targets["hafize-23465"].hosting.retake06, ["retake06"]);
  assert.equal(firebase.hosting.target, "retake06");
  assert.equal(firebase.hosting.public, ".");
});

test("excludes server, native app, tests and private docs from public Hosting", () => {
  const ignored = firebase.hosting.ignore;
  for (const path of ["functions/**", "ios/**", "docs/**", "tests/**", "tools/**", "models/**", "reference-faces/**", "worker/**", "wrangler.toml"]) {
    assert.ok(ignored.includes(path), path + " must not be deployed to Hosting");
  }
});

test("retains the NVIDIA function config alongside one Hosting site", () => {
  assert.equal(firebase.functions.source, "functions");
  assert.equal(firebase.functions.runtime, "nodejs22");
});
