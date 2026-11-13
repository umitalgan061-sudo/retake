// Default reference photos are downloaded once and then served from the
// browser's Cache Storage, so later visits (and the installed PWA) do not
// spend hosting bandwidth on ~15 MB of photos again.

export const REFERENCE_CACHE_NAME = "retake-reference-photos-v1";

function defaultCaches() {
  return globalThis.caches && typeof globalThis.caches.open === "function" ? globalThis.caches : null;
}

async function openCache(cacheStorage) {
  if (!cacheStorage) return null;
  try {
    return await cacheStorage.open(REFERENCE_CACHE_NAME);
  } catch {
    return null; // e.g. private mode or storage disabled
  }
}

/**
 * Cache-first fetch of one reference photo. With `force`, the network copy
 * replaces the cached one. Returns { response, fromCache }.
 */
export async function fetchCachedReference(url, {
  force = false,
  cacheStorage = defaultCaches(),
  fetcher = globalThis.fetch.bind(globalThis),
  init = {}
} = {}) {
  const cache = await openCache(cacheStorage);
  if (cache && !force) {
    const cached = await cache.match(url);
    if (cached && cached.ok) return { response: cached, fromCache: true };
  }
  const response = await fetcher(url, { ...init, cache: force ? "reload" : "default" });
  if (cache && response.ok) {
    try {
      await cache.put(url, response.clone());
    } catch {
      // Quota full: the photo is still used for this session.
    }
  }
  return { response, fromCache: false };
}

/**
 * The manifest is small and decides which photos belong to the set, so it is
 * fetched from the network first; when the site is unreachable (offline or
 * hosting quota exceeded) the last cached copy keeps the app working.
 */
export async function fetchReferenceManifest(url, {
  force = false,
  cacheStorage = defaultCaches(),
  fetcher = globalThis.fetch.bind(globalThis)
} = {}) {
  const cache = await openCache(cacheStorage);
  let response = null;
  try {
    response = await fetcher(url, { cache: force ? "reload" : "no-cache" });
  } catch (error) {
    response = null;
    if (!cache) throw error;
  }
  if (response && response.ok) {
    if (cache) {
      try {
        await cache.put(url, response.clone());
      } catch {
        // Not fatal.
      }
    }
    return { response, fromCache: false };
  }
  const cached = cache ? await cache.match(url) : null;
  if (cached && cached.ok) return { response: cached, fromCache: true };
  if (response) return { response, fromCache: false };
  throw new Error("Referans listesi indirilemedi ve önbellekte yok.");
}

/** Remove cached photos that are no longer listed in the manifest. */
export async function pruneReferenceCache(keepUrls, manifestUrl, { cacheStorage = defaultCaches() } = {}) {
  const cache = await openCache(cacheStorage);
  if (!cache) return 0;
  const keep = new Set([...keepUrls, manifestUrl].map((url) => new URL(url, globalThis.location?.href).href));
  let removed = 0;
  for (const request of await cache.keys()) {
    if (!keep.has(request.url)) {
      await cache.delete(request);
      removed += 1;
    }
  }
  return removed;
}

export async function clearReferenceCache({ cacheStorage = defaultCaches() } = {}) {
  if (!cacheStorage) return false;
  try {
    return await cacheStorage.delete(REFERENCE_CACHE_NAME);
  } catch {
    return false;
  }
}

/** Ask the browser not to evict the photo cache under storage pressure. */
export async function requestPersistentStorage(storageManager = globalThis.navigator?.storage) {
  try {
    if (!storageManager?.persist) return false;
    if (await storageManager.persisted?.()) return true;
    return await storageManager.persist();
  } catch {
    return false;
  }
}
