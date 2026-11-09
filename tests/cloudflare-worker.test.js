import test from "node:test";
import assert from "node:assert/strict";
import worker, { createSessionToken, verifySessionToken } from "../worker/index.js";

const password = "a-unique-retake-test-password-that-is-long";
const env = {
  RETAKE_ACCESS_PASSWORD: password,
  NVIDIA_API_KEY: "test-only-nvidia-key",
  RETAKE_ALLOWED_ORIGINS: "https://retake06.web.app,https://retake06.firebaseapp.com"
};

test("creates a signed session token that expires after twelve hours", async () => {
  const now = 1_800_000_000_000;
  const token = await createSessionToken(password, now);
  assert.equal(await verifySessionToken(token, password, now + 1), true);
  assert.equal(await verifySessionToken(token, password, now + 12 * 60 * 60 * 1000 + 1), false);
  assert.equal(await verifySessionToken(token, "wrong-password", now + 1), false);
});

test("rejects a request from an unapproved web origin", async () => {
  const request = new Request("https://retake-nvidia-edit.example.workers.dev/status", {
    method: "GET",
    headers: { Origin: "https://unapproved.example" }
  });
  const response = await worker.fetch(request, env);
  assert.equal(response.status, 403);
});

test("requires a private session token for status and editing", async () => {
  const request = new Request("https://retake-nvidia-edit.example.workers.dev/status", {
    method: "GET",
    headers: { Origin: "https://retake06.web.app" }
  });
  const response = await worker.fetch(request, env);
  assert.equal(response.status, 401);
});

test("authenticates the personal password and returns an expiring token", async () => {
  const request = new Request("https://retake-nvidia-edit.example.workers.dev/auth", {
    method: "POST",
    headers: {
      Origin: "https://retake06.web.app",
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ password })
  });
  const response = await worker.fetch(request, env);
  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(typeof payload.token, "string");
  assert.equal(await verifySessionToken(payload.token, password), true);
});
