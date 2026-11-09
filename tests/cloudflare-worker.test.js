import test from "node:test";
import assert from "node:assert/strict";
import worker from "../worker/index.js";

const env = {
  NVIDIA_API_KEY: "test-only-nvidia-key",
  RETAKE_ALLOWED_ORIGINS: "https://retake06.web.app,https://retake06.firebaseapp.com"
};

test("rejects browser requests from an unapproved origin", async () => {
  const request = new Request("https://retake-nvidia-edit.example.workers.dev/status", {
    method: "GET",
    headers: { Origin: "https://unapproved.example" }
  });
  const response = await worker.fetch(request, env);
  assert.equal(response.status, 403);
});

test("reports NVIDIA secret configuration without requiring a user login", async () => {
  const request = new Request("https://retake-nvidia-edit.example.workers.dev/status", {
    method: "GET",
    headers: { Origin: "https://retake06.web.app" }
  });
  const response = await worker.fetch(request, env);
  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.configured, true);
  assert.equal(response.headers.get("Access-Control-Allow-Origin"), "https://retake06.web.app");
});

test("reports an unconfigured NVIDIA secret without a login flow", async () => {
  const request = new Request("https://retake-nvidia-edit.example.workers.dev/status", {
    method: "GET",
    headers: { Origin: "https://retake06.web.app" }
  });
  const response = await worker.fetch(request, {
    RETAKE_ALLOWED_ORIGINS: env.RETAKE_ALLOWED_ORIGINS
  });
  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.configured, false);
});

test("requires explicit image consent for edit requests", async () => {
  const request = new Request("https://retake-nvidia-edit.example.workers.dev/edit", {
    method: "POST",
    headers: {
      Origin: "https://retake06.web.app",
      "Content-Type": "image/png"
    },
    body: new Uint8Array([137,80,78,71,13,10,26,10])
  });
  const response = await worker.fetch(request, env);
  assert.equal(response.status, 400);
  const payload = await response.json();
  assert.match(payload.error, /açık onay/);
});

test("rejects non-PNG editing input", async () => {
  const request = new Request("https://retake-nvidia-edit.example.workers.dev/edit", {
    method: "POST",
    headers: {
      Origin: "https://retake06.web.app",
      "Content-Type": "image/jpeg",
      "X-Retake-Consent": "true"
    },
    body: new Uint8Array([1,2,3])
  });
  const response = await worker.fetch(request, env);
  assert.equal(response.status, 415);
});
