const SHELL_CACHE = "retake-shell-v20";
const AI_CACHE = "retake-ai-assets-v2";
// Written by src/reference-cache.js; must survive service-worker updates.
const REFERENCE_CACHE = "retake-reference-photos-v1";
const APP_SHELL = ["./", "./index.html", "./styles.css", "./app.js", "./src/core.js", "./src/reference-catalog.js", "./src/nvidia-pipeline.js", "./src/face-only.js", "./src/identity-profile.js", "./src/profile-storage.js", "./src/head-geometry.js", "./src/stone.js", "./src/engine.js", "./src/reference-cache.js", "./manifest.webmanifest", "./icons/retake.svg", "./icons/retake-180.png", "./icons/retake-192.png", "./icons/retake-512.png", "./icons/retake-maskable-512.png"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(SHELL_CACHE).then((cache) => cache.addAll(APP_SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(
      keys.filter((key) => ![SHELL_CACHE, AI_CACHE, REFERENCE_CACHE].includes(key)).map((key) => caches.delete(key))
    )).then(() => self.clients.claim())
  );
});

function isApprovedAIAsset(url) {
  if (url.origin === "https://cdn.jsdelivr.net") return url.pathname.startsWith("/npm/@mediapipe/tasks-vision@1.2.2/");
  return url.origin === "https://storage.googleapis.com" && [
    "/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    "/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite"
  ].includes(url.pathname);
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
    if (url.pathname.startsWith("/api/images/")) {
      event.respondWith(fetch(request));
      return;
    }
    // Reference photos: the page keeps them in REFERENCE_CACHE; serve any
    // other request for them from there too, and only go to the network on a miss.
    if (url.pathname.startsWith("/reference-faces/") && !url.pathname.endsWith("/manifest.json")) {
      event.respondWith(
        caches.open(REFERENCE_CACHE).then(async (cache) => (await cache.match(request)) || fetch(request))
      );
      return;
    }
    if (url.pathname.startsWith("/reference-faces/")) {
      event.respondWith(fetch(request));
      return;
    }
    if (request.mode === "navigate") {
      event.respondWith(
        fetch(request).then(async (response) => {
          if (response.ok) {
            caches.open(SHELL_CACHE).then((cache) => cache.put("./index.html", response.clone()));
            return response;
          }
          // Hosting errors (e.g. 509 bandwidth quota) fall back to the installed app.
          return (await caches.match("./index.html")) || response;
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