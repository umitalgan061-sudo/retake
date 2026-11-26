// Quality depth (engine v23+): how long Ultra may work for one photo.
// Every depth keeps the same identity, detail and realism gates; a deeper
// level only measures more candidates (longer variant search, a refinement
// around the winner, more realism finalists).

export const QUALITY_DEPTH_STORAGE_KEY = "retake.qualityDepth";
export const DEFAULT_QUALITY_DEPTH = "deep";
// The engine's own default for clients that send nothing (seven minutes).
export const ENGINE_DEFAULT_DEPTH = "standard";

export const QUALITY_DEPTHS = Object.freeze({
  standard: Object.freeze({
    label: "Standart",
    minutes: [8, 15],
    timeoutSeconds: 45 * 60,
    detail: "7 dakikalık arama, 1 gerçekçilik finalisti"
  }),
  deep: Object.freeze({
    label: "Derin",
    minutes: [25, 40],
    timeoutSeconds: 90 * 60,
    detail: "25 dakikalık arama, kazananın çevresinde ince ayar, 2 gerçekçilik finalisti"
  }),
  max: Object.freeze({
    label: "En yüksek",
    minutes: [60, 95],
    timeoutSeconds: 150 * 60,
    detail: "1 saatlik arama (24 ayarın hepsi), geniş ince ayar, 3 gerçekçilik finalisti"
  })
});

export function normalizeDepth(value, fallback = DEFAULT_QUALITY_DEPTH) {
  return typeof value === "string" && Object.hasOwn(QUALITY_DEPTHS, value) ? value : fallback;
}

// What the browser asks for: an engine without depth support always gets the
// seven-minute search, and a self-test benchmark stays comparable.
export function requestedDepth(preferred, { supported = false, benchmark = false } = {}) {
  if (!supported || benchmark) return ENGINE_DEFAULT_DEPTH;
  return normalizeDepth(preferred);
}

// Engine health may describe its depths; trust only sane values.
function engineDepth(health, depth) {
  const entry = health && typeof health === "object" ? health[depth] : null;
  return entry && typeof entry === "object" ? entry : null;
}

export function depthMinutes(depth, health = null) {
  const engine = engineDepth(health, depth);
  const minutes = engine?.expected_minutes;
  if (Array.isArray(minutes) && minutes.length === 2 && minutes.every((value) => Number.isInteger(value) && value > 0 && value < 600) && minutes[0] <= minutes[1]) {
    return minutes;
  }
  return QUALITY_DEPTHS[normalizeDepth(depth, ENGINE_DEFAULT_DEPTH)].minutes;
}

export function depthTimeoutMs(depth, health = null) {
  const fallback = QUALITY_DEPTHS[normalizeDepth(depth, ENGINE_DEFAULT_DEPTH)].timeoutSeconds;
  const engine = Number(engineDepth(health, depth)?.client_timeout_seconds);
  const seconds = Number.isFinite(engine) && engine >= 10 * 60 && engine <= 6 * 60 * 60 ? Math.max(engine, fallback) : fallback;
  return seconds * 1000;
}

export function depthLabel(depth) {
  return QUALITY_DEPTHS[normalizeDepth(depth, ENGINE_DEFAULT_DEPTH)].label;
}

export function describeDepthDuration(depth, health = null) {
  const [low, high] = depthMinutes(depth, health);
  return `${depthLabel(depth)} · yaklaşık ${low}–${high} dakika`;
}
