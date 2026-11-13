import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const source = readFileSync(new URL("../sw.js", import.meta.url), "utf8");

// Load sw.js with a fake service-worker global and drive its fetch handler.
function loadWorker({ network, cached = {} }) {
  const listeners = {};
  const store = new Map(Object.entries(cached));
  const cache = {
    match: async (request) => store.get(typeof request === "string" ? request : request.url)?.clone(),
    put: async (request, response) => store.set(typeof request === "string" ? request : request.url, response),
    addAll: async () => {}
  };
  const self = {
    location: new URL("https://retake.example/sw.js"),
    addEventListener: (type, handler) => { listeners[type] = handler; },
    skipWaiting: () => {},
    clients: { claim: () => {} }
  };
  const caches = { open: async () => cache, match: cache.match, keys: async () => [], delete: async () => true };
  new Function("self", "caches", "fetch", source)(self, caches, network);
  return async (url, mode = "no-cors") => {
    let responded;
    listeners.fetch({ request: { method: "GET", url, mode }, respondWith: (value) => { responded = value; } });
    return responded ? await responded : undefined;
  };
}

const ok = (body) => new Response(body, { status: 200 });
Object.defineProperty(Response.prototype, "type", { get: () => "basic", configurable: true });

test("a new deploy of app code is served instead of the cached copy", async () => {
  const fetchFromWorker = loadWorker({
    network: async () => ok("new app.js"),
    cached: { "https://retake.example/app.js": ok("old app.js") }
  });
  const response = await fetchFromWorker("https://retake.example/app.js");
  assert.equal(await response.text(), "new app.js");
});

test("the cached app code is used when the site is offline or over quota", async () => {
  const offline = loadWorker({
    network: async () => { throw new TypeError("offline"); },
    cached: { "https://retake.example/app.js": ok("cached app.js") }
  });
  assert.equal(await (await offline("https://retake.example/app.js")).text(), "cached app.js");
  const overQuota = loadWorker({
    network: async () => new Response("quota", { status: 509 }),
    cached: { "https://retake.example/styles.css": ok("cached css") }
  });
  assert.equal(await (await overQuota("https://retake.example/styles.css")).text(), "cached css");
});

test("sculpture images are downloaded once", async () => {
  let calls = 0;
  const fetchFromWorker = loadWorker({ network: async () => { calls += 1; return ok("image"); } });
  await fetchFromWorker("https://retake.example/assets/forma/sculpture-dark-wide.webp");
  await fetchFromWorker("https://retake.example/assets/forma/sculpture-dark-wide.webp");
  assert.equal(calls, 1);
});

test("the reference photo cache survives service-worker updates", () => {
  assert.match(source, /\[SHELL_CACHE, AI_CACHE, REFERENCE_CACHE\]\.includes\(key\)/);
});
