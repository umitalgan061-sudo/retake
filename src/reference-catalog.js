import { MAX_REFERENCE_FACES } from "./core.js";

export const REFERENCE_MANIFEST_URL = "./reference-faces/manifest.json";
const SUPPORTED_IMAGE_EXTENSION = /\.(?:jpe?g|png|webp)$/i;

export function parseReferenceManifest(payload) {
  const entries = Array.isArray(payload) ? payload : payload?.images;
  if (!Array.isArray(entries) || entries.length !== MAX_REFERENCE_FACES) {
    throw new TypeError(`Referans manifesti tam olarak ${MAX_REFERENCE_FACES} görsel içermeli.`);
  }

  const seen = new Set();
  return entries.map((entry, index) => {
    const file = typeof entry === "string" ? entry : entry?.file;
    const name = entry && typeof entry === "object" && typeof entry.name === "string"
      ? entry.name.trim()
      : `Referans ${String(index + 1).padStart(2, "0")}`;

    if (typeof file !== "string" || !file.trim() ||
        file.includes("/") || file.includes("\\") || !SUPPORTED_IMAGE_EXTENSION.test(file)) {
      throw new TypeError(`Manifestteki ${index + 1}. dosya adı geçersiz.`);
    }

    const key = file.toLocaleLowerCase("en-US");
    if (seen.has(key)) throw new TypeError(`Manifestte yinelenen dosya adı: ${file}`);
    seen.add(key);
    return { file, name: name || `Referans ${String(index + 1).padStart(2, "0")}` };
  });
}

export function getReferenceImageUrl(fileName, baseUrl) {
  if (typeof fileName !== "string" || fileName.includes("/") || fileName.includes("\\") ||
      !SUPPORTED_IMAGE_EXTENSION.test(fileName)) {
    throw new TypeError("Geçersiz yerel referans dosya adı.");
  }
  return new URL(encodeURIComponent(fileName), baseUrl).href;
}
