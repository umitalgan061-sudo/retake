import test from "node:test";
import assert from "node:assert/strict";
import {
  clearReferenceCache,
  fetchCachedReference,
  fetchReferenceManifest,
  pruneReferenceCache,
  REFERENCE_CACHE_NAME,
  requestPersistentStorage
} from "../src/reference-cache.js";

const BASE = "https://retake06.web.app/reference-faces/";

function fakeCacheStorage() {
  const stores = new Map();
  return {
    stores,
    async open(name) {
      if (!stores.has(name)) stores.set(name, new Map());
      const entries = stores.get(name);
      return {
        async match(url) { return entries.get(String(url))?.clone() ?? undefined; },
        async put(url, response) { entries.set(String(url), response); },
        async keys() { return [...entries.keys()].map((url) => ({ url })); },
        async delete(request) { return entries.delete(request.url); }
      };
    },
    async delete(name) { return stores.delete(name); }
  };
}

function countingFetcher(status = 200, body = "photo") {
  const calls = [];
  const fetcher = async (url, init) => {
    calls.push({ url, init });
    return new Response(body, { status, headers: { "Content-Type": "image/jpeg" } });
  };
  return { fetcher, calls };
}

test("a reference photo is downloaded once and then served from the cache", async () => {
  const cacheStorage = fakeCacheStorage();
  const { fetcher, calls } = countingFetcher();
  const url = BASE + "01.jpeg";
  const first = await fetchCachedReference(url, { cacheStorage, fetcher });
  assert.equal(first.fromCache, false);
  assert.equal(await first.response.text(), "photo");
  for (let visit = 0; visit < 3; visit += 1) {
    const again = await fetchCachedReference(url, { cacheStorage, fetcher });
    assert.equal(again.fromCache, true);
    assert.equal(await again.response.text(), "photo");
  }
  assert.equal(calls.length, 1);
  assert.ok(cacheStorage.stores.has(REFERENCE_CACHE_NAME));
});

test("force re-downloads and replaces the cached photo", async () => {
  const cacheStorage = fakeCacheStorage();
  await fetchCachedReference(BASE + "01.jpeg", { cacheStorage, fetcher: countingFetcher(200, "old").fetcher });
  const { fetcher, calls } = countingFetcher(200, "new");
  const forced = await fetchCachedReference(BASE + "01.jpeg", { cacheStorage, fetcher, force: true });
  assert.equal(forced.fromCache, false);
  assert.equal(calls[0].init.cache, "reload");
  const later = await fetchCachedReference(BASE + "01.jpeg", { cacheStorage, fetcher });
  assert.equal(await later.response.text(), "new");
});

test("failed downloads are not cached", async () => {
  const cacheStorage = fakeCacheStorage();
  const failed = await fetchCachedReference(BASE + "01.jpeg", { cacheStorage, fetcher: countingFetcher(509, "quota").fetcher });
  assert.equal(failed.response.status, 509);
  const { fetcher, calls } = countingFetcher();
  await fetchCachedReference(BASE + "01.jpeg", { cacheStorage, fetcher });
  assert.equal(calls.length, 1);
});

test("without Cache Storage the photo is still fetched", async () => {
  const { fetcher, calls } = countingFetcher();
  const result = await fetchCachedReference(BASE + "01.jpeg", { cacheStorage: null, fetcher });
  assert.equal(result.response.ok, true);
  assert.equal(calls.length, 1);
});

test("the manifest falls back to its cached copy when hosting is over quota or offline", async () => {
  const cacheStorage = fakeCacheStorage();
  const url = BASE + "manifest.json";
  await fetchReferenceManifest(url, { cacheStorage, fetcher: countingFetcher(200, '{"images":[]}').fetcher });
  const overQuota = await fetchReferenceManifest(url, { cacheStorage, fetcher: countingFetcher(509, "quota").fetcher });
  assert.equal(overQuota.fromCache, true);
  assert.equal(await overQuota.response.text(), '{"images":[]}');
  const offline = await fetchReferenceManifest(url, { cacheStorage, fetcher: async () => { throw new TypeError("offline"); } });
  assert.equal(offline.fromCache, true);
});

test("the manifest prefers the network so a changed photo list is picked up", async () => {
  const cacheStorage = fakeCacheStorage();
  const url = BASE + "manifest.json";
  await fetchReferenceManifest(url, { cacheStorage, fetcher: countingFetcher(200, "v1").fetcher });
  const fresh = await fetchReferenceManifest(url, { cacheStorage, fetcher: countingFetcher(200, "v2").fetcher });
  assert.equal(fresh.fromCache, false);
  assert.equal(await fresh.response.text(), "v2");
});

test("pruning removes photos that left the manifest", async () => {
  const cacheStorage = fakeCacheStorage();
  const { fetcher } = countingFetcher();
  for (const name of ["01.jpeg", "02.jpeg", "old.jpeg"]) {
    await fetchCachedReference(BASE + name, { cacheStorage, fetcher });
  }
  await fetchReferenceManifest(BASE + "manifest.json", { cacheStorage, fetcher });
  const removed = await pruneReferenceCache([BASE + "01.jpeg", BASE + "02.jpeg"], BASE + "manifest.json", { cacheStorage });
  assert.equal(removed, 1);
  const keys = [...cacheStorage.stores.get(REFERENCE_CACHE_NAME).keys()];
  assert.deepEqual(keys.sort(), [BASE + "01.jpeg", BASE + "02.jpeg", BASE + "manifest.json"].sort());
});

test("clearing deletes the photo cache", async () => {
  const cacheStorage = fakeCacheStorage();
  await fetchCachedReference(BASE + "01.jpeg", { cacheStorage, fetcher: countingFetcher().fetcher });
  assert.equal(await clearReferenceCache({ cacheStorage }), true);
  assert.equal(cacheStorage.stores.has(REFERENCE_CACHE_NAME), false);
});

test("persistent storage is requested once", async () => {
  let asked = 0;
  assert.equal(await requestPersistentStorage({ persisted: async () => false, persist: async () => { asked += 1; return true; } }), true);
  assert.equal(await requestPersistentStorage({ persisted: async () => true, persist: async () => { asked += 1; return true; } }), true);
  assert.equal(asked, 1);
  assert.equal(await requestPersistentStorage(undefined), false);
});
