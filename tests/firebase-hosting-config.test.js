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

test("excludes source and tooling while publishing the consented reference assets", () => {
  const ignored = firebase.hosting.ignore;
  for (const path of ["functions/**", "ios/**", "docs/**", "tests/**", "tools/**", "models/**", "worker/**", "wrangler.toml"]) {
    assert.ok(ignored.includes(path), path + " must not be deployed to Hosting");
  }
  assert.ok(!ignored.includes("reference-faces/**"), "approved reference photos must be served by Hosting");
});

test("retains the NVIDIA function config alongside one Hosting site", () => {
  assert.equal(firebase.functions.source, "functions");
  assert.equal(firebase.functions.runtime, "nodejs22");
});

test("reference photos are cached long-term by browsers", () => {
  const rule = firebase.hosting.headers.find((item) => item.source.startsWith("/reference-faces/"));
  assert.ok(rule, "reference photo cache rule missing");
  const value = rule.headers.find((header) => header.key === "Cache-Control").value;
  assert.match(value, /max-age=31536000/);
  assert.ok(!rule.source.includes("json"), "the manifest must stay revalidated");
  const last = firebase.hosting.headers.at(-1);
  assert.equal(last, rule, "the specific rule must come after the catch-all no-cache rule");
});
