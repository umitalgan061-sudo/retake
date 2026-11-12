// Where the Retake engine lives. By default it is the FaceFusion bridge on this
// same computer. A phone or an installed PWA can instead be paired with that
// computer through an HTTPS tunnel and a pairing code.

export const DEFAULT_ENGINE_ORIGIN = "http://127.0.0.1:8765";
export const ENGINE_STORAGE_KEY = "retake-engine";
export const PAIRING_HEADER = "X-Retake-Token";
// Must stay in sync with the connect-src list in index.html.
export const TUNNEL_SUFFIXES = [".trycloudflare.com", ".ts.net"];

const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost"]);
const TOKEN_PATTERN = /^[A-Za-z0-9_-]{16,128}$/;

/** Validate an engine address typed by the user. */
export function normalizeEngineAddress(value) {
  const text = String(value ?? "").trim();
  if (!text) return { ok: true, origin: DEFAULT_ENGINE_ORIGIN, remote: false };
  const loopbackLike = /^(127\.0\.0\.1|localhost)(:|\/|$)/i.test(text);
  const withScheme = /^[a-z][a-z0-9+.-]*:\/\//i.test(text) ? text : (loopbackLike ? "http://" : "https://") + text;
  let url;
  try {
    url = new URL(withScheme);
  } catch {
    return { ok: false, reason: "Adres okunamadı. Örnek: https://ornek.trycloudflare.com" };
  }
  if (url.username || url.password) return { ok: false, reason: "Adres kullanıcı adı veya parola içeremez." };
  const host = url.hostname.toLowerCase();
  if (LOOPBACK_HOSTS.has(host)) {
    if (url.protocol !== "http:") return { ok: false, reason: "Bu cihazdaki motor http:// ile açılır." };
    return { ok: true, origin: url.origin, remote: false };
  }
  if (url.protocol !== "https:") {
    return { ok: false, reason: "Uzak motor adresi https:// ile başlamalı (şifreli tünel)." };
  }
  if (!TUNNEL_SUFFIXES.some((suffix) => host.endsWith(suffix) && host.length > suffix.length)) {
    return {
      ok: false,
      reason: "Yalnızca Cloudflare (…trycloudflare.com) veya Tailscale (…ts.net) tünel adresleri desteklenir."
    };
  }
  return { ok: true, origin: url.origin, remote: true };
}

export function normalizePairingToken(value) {
  const text = String(value ?? "").trim();
  if (!text) return { ok: true, token: "" };
  return TOKEN_PATTERN.test(text)
    ? { ok: true, token: text }
    : { ok: false, reason: "Eşleştirme kodu 16–128 karakter olmalı (harf, rakam, - ve _)." };
}

/** A remote engine is only usable with a pairing code. */
export function resolveEngineSettings(address, token) {
  const parsedAddress = normalizeEngineAddress(address);
  if (!parsedAddress.ok) return parsedAddress;
  const parsedToken = normalizePairingToken(token);
  if (!parsedToken.ok) return parsedToken;
  if (parsedAddress.remote && !parsedToken.token) {
    return { ok: false, reason: "Uzak motor için bilgisayardaki pencerede yazan eşleştirme kodunu gir." };
  }
  return { ok: true, origin: parsedAddress.origin, remote: parsedAddress.remote, token: parsedToken.token };
}

export function loadEngineSettings(storage) {
  const fallback = { ok: true, origin: DEFAULT_ENGINE_ORIGIN, remote: false, token: "" };
  try {
    const saved = JSON.parse(storage?.getItem(ENGINE_STORAGE_KEY) || "null");
    if (!saved || typeof saved !== "object") return fallback;
    const resolved = resolveEngineSettings(saved.origin, saved.token);
    return resolved.ok ? resolved : fallback;
  } catch {
    return fallback;
  }
}

export function saveEngineSettings(storage, settings) {
  try {
    if (!settings || (settings.origin === DEFAULT_ENGINE_ORIGIN && !settings.token)) {
      storage?.removeItem(ENGINE_STORAGE_KEY);
    } else {
      storage?.setItem(ENGINE_STORAGE_KEY, JSON.stringify({ origin: settings.origin, token: settings.token }));
    }
    return true;
  } catch {
    return false;
  }
}

export function engineHeaders(settings, extra = {}) {
  return settings?.token ? { ...extra, [PAIRING_HEADER]: settings.token } : { ...extra };
}

/** mm:ss for the wait counter. */
export function formatElapsed(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  return String(Math.floor(total / 60)).padStart(2, "0") + ":" + String(total % 60).padStart(2, "0");
}
