// Diagnostics report (v26): one text the person can paste into a chat so the
// engine can be tuned for their PC. It holds measurements only — never an
// image, a file name, a folder path, a token or a pairing code.

const HEALTH_KEYS = Object.freeze([
  "engine", "version", "model", "ultra_rounds", "ultra_min_seconds", "quality_depths",
  "ultra_default_depth", "fine_features", "identity_scorer", "pass_cache", "nvidia_judge",
  "nim_editors_configured", "pixel_boost", "face_enhancer_enabled", "face_enhancer_model",
  "realism", "personal_model", "cuda", "ultra_learning", "self_test_benchmark",
  "neck_harmonization", "last_job_cleanup"
]);
const SECRET_NAME = /token|secret|key|pairing|auth|cookie|password/i;
const MAX_VALUE = 300;

// Windows/POSIX paths and file names never leave the device.
export function scrub(value) {
  return String(value)
    .replace(/[A-Za-z]:[\\/][^\s,;"']*/g, "[yol]")
    .replace(/(?:^|(?<=[\s,;"'=(]))\/(?:[\w.-]+\/)+[\w.-]*/g, "[yol]")
    .replace(/[\w.-]+\.(?:png|jpe?g|webp|heic|bmp|tiff?|json|onnx|safetensors)\b/gi, "[dosya]")
    .slice(0, MAX_VALUE);
}

export function collectDiagnosticHeaders(headers) {
  const collected = {};
  if (!headers) return collected;
  const add = (name, value) => {
    const key = String(name).toLowerCase();
    if (!key.startsWith("x-retake-") || SECRET_NAME.test(key)) return;
    collected[key] = scrub(value);
  };
  if (typeof headers.forEach === "function") headers.forEach((value, name) => add(name, value));
  else for (const [name, value] of Object.entries(headers)) add(name, value);
  return Object.fromEntries(Object.entries(collected).sort(([a], [b]) => a.localeCompare(b)));
}

export function parseUltraTiming(value) {
  const result = {};
  if (typeof value !== "string") return result;
  for (const part of value.split(";")) {
    const match = /^([a-z]+)=(\d{1,6})(?:x(\d{1,6}))?$/.exec(part.trim());
    if (!match) continue;
    result[match[1]] = match[3] === undefined
      ? { seconds: Number(match[2]) }
      : { count: Number(match[2]), seconds: Number(match[3]) };
  }
  return result;
}

function minutes(seconds) {
  return seconds >= 90 ? `${Math.round(seconds / 60)} dk` : `${seconds} sn`;
}

export function describeUltraTiming(value) {
  const timing = parseUltraTiming(value);
  const names = { search: "tur", refine: "ince ayar", realism: "gerçekçilik", extra: "ek finalist" };
  const parts = [];
  for (const [key, label] of Object.entries(names)) {
    const item = timing[key];
    if (item?.count > 0) parts.push(`${item.count} ${label} × ${minutes(item.seconds)}`);
  }
  if (timing.judge?.seconds > 0) parts.push(`jüri ${minutes(timing.judge.seconds)}`);
  if (timing.total?.seconds > 0) parts.push(`toplam ${minutes(timing.total.seconds)}`);
  return parts.length ? `Süre: ${parts.join(" · ")}` : "";
}

function safeJson(value) {
  try {
    return JSON.stringify(value);
  } catch {
    return "";
  }
}

function healthSummary(health) {
  const summary = {};
  if (!health || typeof health !== "object") return summary;
  for (const key of HEALTH_KEYS) {
    if (!Object.hasOwn(health, key)) continue;
    const value = health[key];
    summary[key] = value !== null && typeof value === "object" ? scrub(safeJson(value)) : scrub(value);
  }
  return summary;
}

export function buildDiagnosticReport({ headers = {}, health = null, depth = "", appVersion = "", createdAt = new Date() } = {}) {
  const lines = [
    "Retake tanı raporu (görsel, dosya adı veya klasör yolu içermez)",
    `Zaman: ${createdAt.toISOString()}`,
  ];
  if (appVersion) lines.push(`Site: ${scrub(appVersion)}`);
  if (depth) lines.push(`Seçilen derinlik: ${scrub(depth)}`);
  const timing = describeUltraTiming(headers["x-retake-ultra-timing"]);
  if (timing) lines.push(timing);
  lines.push("", "Sonuç ölçümleri:");
  for (const [name, value] of Object.entries(headers)) lines.push(`${name}: ${value}`);
  const summary = healthSummary(health);
  if (Object.keys(summary).length) {
    lines.push("", "Motor durumu:");
    for (const [key, value] of Object.entries(summary)) lines.push(`${key}: ${value}`);
  }
  return lines.join("\n");
}
