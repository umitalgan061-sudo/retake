const SHELL_CACHE = "retake-shell-v5";
const AI_CACHE = "retake-ai-assets-v1";
const APP_SHELL = ["./", "./index.html", "./styles.css", "./app.js", "./src/core.js", "./src/reference-catalog.js", "./src/nvidia-pipeline.js", "./src/face-only.js", "./manifest.webmanifest", "./icons/retake.svg"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(SHELL_CACHE).then((cache) => cache.addAll(APP_SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(
      keys.filter((key) => ![SHELL_CACHE, AI_CACHE].includes(key)).map((key) => caches.delete(key))
    )).then(() => self.clients.claim())
  );
});

function isApprovedAIAsset(url) {
  if (url.origin === "https://cdn.jsdelivr.net") return url.pathname.startsWith("/npm/@mediapipe/tasks-vision@1.2.2/");
  return url.origin === "https://storage.googleapis.com" &&
    url.pathname === "/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
}

async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (response.ok && (response.type === "basic" || response.type === "cors")) await cache.put(request, response.clone());
  return response;
}

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);

  if (url.origin === self.location.origin) {
    // Kişisel yüz görselleri Cache Storage'a yazılmasın.
    if (url.pathname.startsWith("/api/images/")) {
      event.respondWith(fetch(request));
      return;
    }
    if (request.mode === "navigate") {
      event.respondWith(
        fetch(request).then((response) => {
          if (response.ok) caches.open(SHELL_CACHE).then((cache) => cache.put("./index.html", response.clone()));
          return response;
        }).catch(async () => (await caches.match("./index.html")) || Response.error())
      );
    } else {
      event.respondWith(cacheFirst(request, SHELL_CACHE).catch(async () => (await caches.match(request)) || Response.error()));
    }
    return;
  }

  if (isApprovedAIAsset(url)) {
    event.respondWith(cacheFirst(request, AI_CACHE).catch(async () => (await caches.match(request)) || Response.error()));
  }
});