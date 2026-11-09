import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("../", import.meta.url));
const FORBIDDEN = [
  "getUserMedia",
  "getDisplayMedia",
  "navigator.mediaDevices",
  "MediaStream",
  "<video",
  "AVCaptureSession",
  "AVCaptureDeviceInput",
  "AVCaptureVideoDataOutput",
  "AVCapturePhotoOutput",
  "UIImagePickerController",
  "cameraFlipBtn",
  "startCamera"
];

function collectFiles(directory, predicate) {
  const files = [];
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const absolute = join(directory, entry.name);
    if (entry.isDirectory()) files.push(...collectFiles(absolute, predicate));
    else if (entry.isFile() && predicate(absolute)) files.push(absolute);
  }
  return files;
}

test("Retake photo flow contains no camera or live-video APIs", () => {
  const candidates = [
    join(ROOT, "index.html"),
    join(ROOT, "app.js"),
    ...collectFiles(join(ROOT, "ios", "RetakeApp"), (path) => path.endsWith(".swift"))
  ];
  const violations = [];
  for (const path of candidates) {
    const source = readFileSync(path, "utf8");
    for (const token of FORBIDDEN) {
      if (source.includes(token)) violations.push(relative(ROOT, path) + ": " + token);
    }
  }
  assert.deepEqual(violations, [], "Camera APIs must not be introduced: " + violations.join(", "));
});