import { onRequest } from "firebase-functions/v2/https";
import { defineSecret } from "firebase-functions/params";
import {
  MAX_INPUT_BYTES,
  decodePngDataUrl,
  imageDataUrlFromNvidiaResponse,
  describeNvidiaFailure
} from "./lib/nvidia.js";

const NVIDIA_API_KEY = defineSecret("NVIDIA_API_KEY");
const REGION = "europe-west1";
const FLUX_URL = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev";
const NVCF_ASSETS_URL = "https://api.nvcf.nvidia.com/v2/nvcf/assets";
const REQUEST_WINDOW_MS = 15 * 60 * 1000;
const REQUEST_LIMIT = 5;
const requestBuckets = new Map();

function allowedOrigins() {
  const configured = (process.env.RETAKE_ALLOWED_ORIGINS || "https://retake06.web.app,https://retake06.firebaseapp.com")
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  return new Set(configured);
}

function setCors(req, res) {
  const origin = req.headers.origin;
  if (origin && allowedOrigins().has(origin)) {
    res.set("Access-Control-Allow-Origin", origin);
    res.set("Vary", "Origin");
    res.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set("Access-Control-Allow-Headers", "Content-Type, Accept");
    res.set("Access-Control-Max-Age", "600");
    return true;
  }
  return !origin;
}

function checkRateLimit(req) {
  const forwarded = req.headers["x-forwarded-for"];
  const ip = (Array.isArray(forwarded) ? forwarded[0] : String(forwarded || "unknown")).split(",")[0].trim();
  const now = Date.now();
  const bucket = requestBuckets.get(ip);
  if (!bucket || now - bucket.startedAt >= REQUEST_WINDOW_MS) {
    requestBuckets.set(ip, { startedAt: now, count: 1 });
    if (requestBuckets.size > 1000) {
      for (const [key, value] of requestBuckets) {
        if (now - value.startedAt >= REQUEST_WINDOW_MS) requestBuckets.delete(key);
      }
    }
    return true;
  }
  if (bucket.count >= REQUEST_LIMIT) return false;
  bucket.count += 1;
  return true;
}

async function safeJson(response) {
  return response.json().catch(() => ({}));
}

async function createAndUploadAsset(apiKey, imageBytes) {
  const createResponse = await fetch(NVCF_ASSETS_URL, {
    method: "POST",
    headers: {
      Authorization: "Bearer " + apiKey,
      "Content-Type": "application/json",
      Accept: "application/json"
    },
    body: JSON.stringify({
      contentType: "image/png",
      description: "Retake temporary face-edit reference board"
    }),
    signal: AbortSignal.timeout(30000)
  });

  if (!createResponse.ok) {
    throw Object.assign(new Error(describeNvidiaFailure(createResponse.status, "asset")), {
      status: createResponse.status
    });
  }

  const asset = await safeJson(createResponse);
  if (!asset.assetId || !asset.uploadUrl) {
    throw new Error("NVIDIA varlık oluşturma yanıtı eksik assetId/uploadUrl içeriyor.");
  }

  const uploadUrl = new URL(asset.uploadUrl);
  if (uploadUrl.protocol !== "https:") {
    throw new Error("NVIDIA geçersiz yükleme adresi döndürdü.");
  }

  try {
    const uploadResponse = await fetch(uploadUrl, {
      method: "PUT",
      headers: {
        "Content-Type": "image/png",
        "x-amz-meta-nvcf-asset-description": "Retake temporary face-edit board"
      },
      body: imageBytes,
      signal: AbortSignal.timeout(60000)
    });

    if (!uploadResponse.ok) {
      throw Object.assign(new Error("NVIDIA geçici görsel varlığına yükleme yapılamadı."), {
        status: uploadResponse.status
      });
    }
    return asset.assetId;
  } catch (error) {
    await deleteAsset(apiKey, asset.assetId);
    throw error;
  }
}

async function deleteAsset(apiKey, assetId) {
  if (!assetId) return;
  try {
    await fetch(NVCF_ASSETS_URL + "/" + encodeURIComponent(assetId), {
      method: "DELETE",
      headers: { Authorization: "Bearer " + apiKey },
      signal: AbortSignal.timeout(15000)
    });
  } catch (error) {
    console.warn("Retake geçici NVIDIA varlığı silinemedi:", error?.message || "unknown error");
  }
}

async function invokeFlux(apiKey, assetId) {
  const response = await fetch(FLUX_URL, {
    method: "POST",
    headers: {
      Authorization: "Bearer " + apiKey,
      "Content-Type": "application/json",
      Accept: "application/json",
      "NVCF-INPUT-ASSET-REFERENCES": assetId
    },
    body: JSON.stringify({
      prompt: [
        "Edit the provided two-panel reference board.",
        "The LEFT panel is one target face crop. The RIGHT panel contains twenty identity reference photos of the same person.",
        "Change only the facial identity of the person in the LEFT panel to resemble the person shown consistently across the RIGHT panel.",
        "Use all twenty references together. Keep the target head pose, expression, gaze, camera angle, lighting and realistic skin texture.",
        "If the target eyes are closed, naturally open them using the open-eye appearance shown in the references while retaining the gaze direction.",
        "Preserve the target hairline, hair, ears, neck, clothing, body and background. Do not change or use the right panel as part of the output subject.",
        "Do not add people, text, graphics or a new scene. Return the same image board and aspect ratio."
      ].join(" "),
      image: "data:image/png;asset_id," + assetId,
      width: 1568,
      height: 1024,
      cfg_scale: 3.5,
      aspect_ratio: "3:2",
      samples: 1,
      seed: 0,
      steps: 30
    }),
    signal: AbortSignal.timeout(500000)
  });

  if (!response.ok) {
    throw Object.assign(new Error(describeNvidiaFailure(response.status)), {
      status: response.status
    });
  }

  const payload = await safeJson(response);
  return imageDataUrlFromNvidiaResponse(payload);
}

export const retakeNvidiaEdit = onRequest({
  region: REGION,
  timeoutSeconds: 540,
  memory: "1GiB",
  maxInstances: 1,
  concurrency: 1,
  secrets: [NVIDIA_API_KEY]
}, async (req, res) => {
  if (!setCors(req, res)) {
    res.status(403).json({ error: "Bu site adresinden gelen istek kabul edilmiyor." });
    return;
  }
  if (req.method === "OPTIONS") {
    res.status(204).send("");
    return;
  }

  const path = (req.path || "/").replace(/\/+$/, "") || "/";
  if (req.method === "GET" && path === "/status") {
    const configured = Boolean(NVIDIA_API_KEY.value());
    res.set("Cache-Control", "no-store");
    res.status(200).json({
      provider: "NVIDIA",
      model: "black-forest-labs/flux.1-kontext-dev",
      configured,
      mode: "experimental-image-edit",
      message: configured ? "" : "NVIDIA_API_KEY Firebase Secret olarak tanımlanmamış."
    });
    return;
  }

  if (req.method !== "POST" || path !== "/edit") {
    res.status(404).json({ error: "API yolu bulunamadı." });
    return;
  }
  if (!checkRateLimit(req)) {
    res.status(429).json({ error: "Kısa sürede çok fazla istek gönderildi. 15 dakika sonra tekrar dene." });
    return;
  }

  const apiKey = NVIDIA_API_KEY.value();
  if (!apiKey) {
    res.status(503).json({ error: "NVIDIA_API_KEY Firebase Secret olarak yapılandırılmamış." });
    return;
  }

  if (req.body?.consent !== true) {
    res.status(400).json({ error: "NVIDIA'ya görsel gönderimi için açık onay gerekli." });
    return;
  }

  let imageBytes;
  try {
    imageBytes = decodePngDataUrl(req.body?.imageDataUrl);
  } catch (error) {
    res.status(400).json({ error: error.message });
    return;
  }
  if (imageBytes.length > MAX_INPUT_BYTES) {
    res.status(413).json({ error: "Referans panosu 10 MB sınırını aşıyor." });
    return;
  }

  let assetId;
  try {
    assetId = await createAndUploadAsset(apiKey, imageBytes);
    const imageDataUrl = await invokeFlux(apiKey, assetId);
    res.set("Cache-Control", "no-store");
    res.status(200).json({
      imageDataUrl,
      model: "black-forest-labs/flux.1-kontext-dev",
      experimental: true,
      outsideFacePixels: "preserved-by-client-mask"
    });
  } catch (error) {
    const status = Number.isInteger(error?.status) ? error.status : 502;
    console.error("NVIDIA FLUX request failed:", error?.message || "unknown error");
    res.status(status >= 400 && status < 600 ? status : 502).json({
      error: error?.message || "NVIDIA görsel düzenleme isteği başarısız oldu."
    });
  } finally {
    await deleteAsset(apiKey, assetId);
  }
});
