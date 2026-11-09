const MAX_INPUT_BYTES = 10 * 1024 * 1024;
const RATE_WINDOW_MS = 15 * 60 * 1000;
const FLUX_URL = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev";
const NVCF_ASSETS_URL = "https://api.nvcf.nvidia.com/v2/nvcf/assets";
const editAttempts = new Map();

function configuredOrigins(env) {
  return new Set(
    String(env.RETAKE_ALLOWED_ORIGINS || "https://retake06.web.app,https://retake06.firebaseapp.com")
      .split(",")
      .map((origin) => origin.trim())
      .filter(Boolean)
  );
}

function corsHeaders(request, env) {
  const origin = request.headers.get("Origin");
  if (!origin) return new Headers();
  if (!configuredOrigins(env).has(origin)) return null;
  return new Headers({
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, X-Retake-Consent",
    "Access-Control-Max-Age": "600",
    "Vary": "Origin"
  });
}

function jsonResponse(body, status, cors) {
  const headers = new Headers(cors || undefined);
  headers.set("Content-Type", "application/json; charset=utf-8");
  headers.set("Cache-Control", "no-store");
  return new Response(JSON.stringify(body), { status, headers });
}

function allowedByRateLimit(key, limit) {
  const now = Date.now();
  const current = editAttempts.get(key);
  if (!current || now - current.startedAt >= RATE_WINDOW_MS) {
    editAttempts.set(key, { startedAt: now, count: 1 });
    if (editAttempts.size > 1200) {
      for (const [entry, value] of editAttempts) {
        if (now - value.startedAt >= RATE_WINDOW_MS) editAttempts.delete(entry);
      }
    }
    return true;
  }
  if (current.count >= limit) return false;
  current.count += 1;
  return true;
}

function clientAddress(request) {
  return request.headers.get("CF-Connecting-IP") || "unknown";
}

function describeNvidiaFailure(status, stage = "inference") {
  if (status === 401) return "NVIDIA API anahtarı geçersiz veya süresi dolmuş.";
  if (status === 403) return "NVIDIA API anahtarında invoke_function yetkisi veya bu görsel uç noktasına erişim yok.";
  if (status === 402) return "NVIDIA tarafında kullanım kotası veya ödeme sınırı aşıldı.";
  if (status === 404) return "FLUX.1-Kontext-dev uç noktası bu NVIDIA hesabında etkin değil. Hesabın desteklediği hosted endpoint erişimini kontrol et.";
  if (status === 413) return "Görsel NVIDIA'nın kabul ettiği boyut sınırını aşıyor.";
  if (status === 422) return "NVIDIA görsel girdisini reddetti. Bu hesap/endpoint özel görsel yüklemeyi desteklemiyor olabilir.";
  if (status === 429) return "NVIDIA istek sınırına ulaşıldı. Biraz sonra tekrar dene.";
  if (status >= 500) return "NVIDIA servisi geçici olarak hata verdi.";
  return stage === "asset" ? "NVIDIA geçici görsel varlığı oluşturulamadı veya yüklenemedi." : "NVIDIA isteği başarısız oldu.";
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
    })
  });

  if (!createResponse.ok) {
    throw Object.assign(new Error(describeNvidiaFailure(createResponse.status, "asset")), {
      status: createResponse.status
    });
  }

  const asset = await createResponse.json().catch(() => ({}));
  if (typeof asset.assetId !== "string" || typeof asset.uploadUrl !== "string") {
    throw new Error("NVIDIA varlık yanıtında assetId veya uploadUrl bulunamadı.");
  }

  let uploadUrl;
  try {
    uploadUrl = new URL(asset.uploadUrl);
  } catch {
    throw new Error("NVIDIA geçersiz bir yükleme adresi döndürdü.");
  }
  if (uploadUrl.protocol !== "https:") {
    throw new Error("NVIDIA güvenli olmayan bir yükleme adresi döndürdü.");
  }

  try {
    const uploadResponse = await fetch(uploadUrl, {
      method: "PUT",
      headers: {
        "Content-Type": "image/png",
        "x-amz-meta-nvcf-asset-description": "Retake temporary face-edit board"
      },
      body: imageBytes
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
      headers: { Authorization: "Bearer " + apiKey }
    });
  } catch {
    // Temporary asset cleanup is best-effort; never log image or credential data.
  }
}

async function invokeFlux(apiKey, assetId) {
  return fetch(FLUX_URL, {
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
    })
  });
}

export default {
  async fetch(request, env) {
    const cors = corsHeaders(request, env);
    if (request.headers.get("Origin") && !cors) {
      return jsonResponse({ error: "Bu site adresinden gelen istek kabul edilmiyor." }, 403);
    }

    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, "") || "/";

    if (request.method === "OPTIONS") {
      if (!cors) return new Response(null, { status: 403 });
      return new Response(null, { status: 204, headers: cors });
    }

    if (path === "/status" && request.method === "GET") {
      return jsonResponse({
        provider: "NVIDIA",
        model: "black-forest-labs/flux.1-kontext-dev",
        configured: Boolean(env.NVIDIA_API_KEY),
        mode: "experimental-image-edit",
        message: env.NVIDIA_API_KEY ? "" : "NVIDIA_API_KEY Worker Secret olarak tanımlanmamış."
      }, 200, cors);
    }

    if (path !== "/edit" || request.method !== "POST") {
      return jsonResponse({ error: "API yolu bulunamadı." }, 404, cors);
    }

    if (!env.NVIDIA_API_KEY) {
      return jsonResponse({ error: "NVIDIA_API_KEY Worker Secret olarak tanımlanmamış." }, 503, cors);
    }
    if (!allowedByRateLimit(clientAddress(request), 5)) {
      return jsonResponse({ error: "Bu IP adresinden 15 dakika içinde en fazla 5 düzenleme isteği gönderilebilir." }, 429, cors);
    }
    if (request.headers.get("X-Retake-Consent") !== "true") {
      return jsonResponse({ error: "NVIDIA'ya görsel gönderimi için açık onay gerekli." }, 400, cors);
    }
    if (!(request.headers.get("Content-Type") || "").toLowerCase().startsWith("image/png")) {
      return jsonResponse({ error: "Yalnızca PNG referans panosu kabul edilir." }, 415, cors);
    }

    const contentLength = Number(request.headers.get("Content-Length") || 0);
    if (contentLength > MAX_INPUT_BYTES) {
      return jsonResponse({ error: "Referans panosu 10 MB sınırını aşıyor." }, 413, cors);
    }

    let imageBytes;
    try {
      imageBytes = new Uint8Array(await request.arrayBuffer());
    } catch {
      return jsonResponse({ error: "Görsel verisi okunamadı." }, 400, cors);
    }
    if (imageBytes.length < 8 || imageBytes.byteLength > MAX_INPUT_BYTES) {
      return jsonResponse({ error: "Referans panosu boş, eksik veya 10 MB sınırını aşıyor." }, 413, cors);
    }
    const pngSignature = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
    if (!pngSignature.every((value, index) => imageBytes[index] === value)) {
      return jsonResponse({ error: "Gönderilen görsel geçerli bir PNG değil." }, 400, cors);
    }

    let assetId = null;
    let nvidiaResponse = null;
    try {
      assetId = await createAndUploadAsset(env.NVIDIA_API_KEY, imageBytes);
      nvidiaResponse = await invokeFlux(env.NVIDIA_API_KEY, assetId);
    } catch (error) {
      await deleteAsset(env.NVIDIA_API_KEY, assetId);
      const status = Number.isInteger(error?.status) ? error.status : 502;
      return jsonResponse({
        error: error?.message || describeNvidiaFailure(status, "asset")
      }, status >= 400 && status < 600 ? status : 502, cors);
    }

    await deleteAsset(env.NVIDIA_API_KEY, assetId);

    if (!nvidiaResponse.ok) {
      const status = nvidiaResponse.status;
      if (nvidiaResponse.body) {
        try {
          await nvidiaResponse.body.cancel();
        } catch {
          // The upstream error body is not needed by the client.
        }
      }
      return jsonResponse({ error: describeNvidiaFailure(status) }, status, cors);
    }

    const responseHeaders = new Headers(cors || undefined);
    responseHeaders.set("Content-Type", nvidiaResponse.headers.get("Content-Type") || "application/json");
    responseHeaders.set("Cache-Control", "no-store");
    return new Response(nvidiaResponse.body, {
      status: nvidiaResponse.status,
      headers: responseHeaders
    });
  }
};
