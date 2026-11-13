export const MAX_REFERENCE_FACES = 25;
export const MAX_IMAGE_BYTES = 25 * 1024 * 1024;
export const MAX_IMAGE_PIXELS = 20_000_000;
export const MAX_IMAGE_EDGE = 12_000;
export const ACCEPTED_IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

export function validateImageFile(file, maxBytes = MAX_IMAGE_BYTES) {
  if (!file || typeof file !== "object") return { valid: false, reason: "Dosya okunamadı." };
  if (!ACCEPTED_IMAGE_TYPES.has(file.type)) return { valid: false, reason: "Yalnızca JPG, PNG veya WebP görselleri destekleniyor." };
  if (!Number.isFinite(file.size) || file.size <= 0) return { valid: false, reason: "Dosya boş veya okunamıyor." };
  if (file.size > maxBytes) return { valid: false, reason: "Dosya 25 MB sınırını aşıyor." };
  return { valid: true, reason: "" };
}

export function validateReferenceFiles(files) {
  if (!Array.isArray(files) || files.length !== MAX_REFERENCE_FACES) {
    return {
      valid: false,
      reason: `Kişisel profil için tam ${MAX_REFERENCE_FACES} fotoğraf seçmelisin (${files?.length || 0} seçildi).`
    };
  }

  const invalidFiles = files.flatMap((file) => {
    const validation = validateImageFile(file);
    return validation.valid ? [] : [`${file?.name || "Adsız dosya"}: ${validation.reason}`];
  });
  return {
    valid: invalidFiles.length === 0,
    reason: invalidFiles.join(" "),
    invalidFiles
  };
}

export function validateImageDimensions(width, height) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    return { valid: false, reason: "Görsel boyutları okunamadı." };
  }
  if (width > MAX_IMAGE_EDGE || height > MAX_IMAGE_EDGE || width * height > MAX_IMAGE_PIXELS) {
    return {
      valid: false,
      reason: `Görsel boyutu çok büyük (${width} × ${height} px). En fazla ${MAX_IMAGE_EDGE} px kenar ve ${MAX_IMAGE_PIXELS / 1_000_000} MP destekleniyor.`
    };
  }
  return { valid: true, reason: "" };
}

export function getLandmarkBounds(landmarks, padding = 0.055) {
  if (!Array.isArray(landmarks) || landmarks.length === 0) return null;
  const points = landmarks.filter((point) => Number.isFinite(point?.x) && Number.isFinite(point?.y));
  if (points.length === 0) return null;
  const minX = Math.min(...points.map((point) => point.x));
  const maxX = Math.max(...points.map((point) => point.x));
  const minY = Math.min(...points.map((point) => point.y));
  const maxY = Math.max(...points.map((point) => point.y));
  const padX = Math.max((maxX - minX) * padding, 0.012);
  const padY = Math.max((maxY - minY) * padding, 0.012);
  const x = clamp(minX - padX, 0, 1);
  const y = clamp(minY - padY, 0, 1);
  const right = clamp(maxX + padX, 0, 1);
  const bottom = clamp(maxY + padY, 0, 1);
  if (right <= x || bottom <= y) return null;
  return { x, y, width: right - x, height: bottom - y };
}

export function getContainedRect(sourceWidth, sourceHeight, containerWidth, containerHeight) {
  const values = [sourceWidth, sourceHeight, containerWidth, containerHeight];
  if (values.some((value) => !Number.isFinite(value) || value <= 0)) return null;
  const scale = Math.min(containerWidth / sourceWidth, containerHeight / sourceHeight);
  const width = sourceWidth * scale;
  const height = sourceHeight * scale;
  return { x: (containerWidth - width) / 2, y: (containerHeight - height) / 2, width, height, scale };
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const amount = bytes / (1024 ** index);
  return new Intl.NumberFormat("tr-TR", { maximumFractionDigits: 1 }).format(amount) + " " + units[index];
}

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
/**
 * Like Promise.allSettled(items.map(task)) but runs at most `limit` tasks at
 * once. Results keep the order of `items`.
 */
export async function settleWithLimit(items, limit, task) {
  const results = new Array(items.length);
  let next = 0;
  async function worker() {
    while (next < items.length) {
      const index = next;
      next += 1;
      try {
        results[index] = { status: "fulfilled", value: await task(items[index], index) };
      } catch (reason) {
        results[index] = { status: "rejected", reason };
      }
    }
  }
  const workers = Math.max(1, Math.min(Math.floor(limit) || 1, items.length));
  await Promise.all(Array.from({ length: workers }, worker));
  return results;
}
