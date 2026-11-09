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

test("uploads an asset with metadata matching its signed description", async (t) => {
  const assetUrl = "https://api.nvcf.nvidia.com/v2/nvcf/assets";
  const uploadUrl = "https://upload.example/presigned";
  const requests = [];

  t.mock.method(globalThis, "fetch", async (input, init = {}) => {
    const url = input instanceof Request ? input.url : String(input);
    requests.push({ url, init });

    if (url === assetUrl && init.method === "POST") {
      return Response.json({ assetId: "test-asset", uploadUrl });
    }
    if (url === uploadUrl && init.method === "PUT") return new Response(null, { status: 200 });
    if (url.endsWith("/test-asset") && init.method === "DELETE") return new Response(null, { status: 204 });
    if (url === "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev") {
      return Response.json({ artifacts: [{ base64: "AQID" }] });
    }
    throw new Error("Unexpected fetch: " + url);
  });

  const request = new Request("https://retake-nvidia-edit.example.workers.dev/edit", {
    method: "POST",
    headers: {
      Origin: "https://retake06.web.app",
      "Content-Type": "image/png",
      "X-Retake-Consent": "true"
    },
    body: new Uint8Array([137,80,78,71,13,10,26,10])
  });

  const response = await worker.fetch(request, env);
  assert.equal(response.status, 200);

  const assetCreate = requests.find((entry) => entry.url === assetUrl && entry.init.method === "POST");
  const upload = requests.find((entry) => entry.url === uploadUrl && entry.init.method === "PUT");
  assert.equal(
    JSON.parse(assetCreate.init.body).description,
    upload.init.headers["x-amz-meta-nvcf-asset-description"]
  );
});
