import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  DEFAULT_ENGINE_ORIGIN,
  engineHeaders,
  formatElapsed,
  loadEngineSettings,
  normalizeEngineAddress,
  normalizePairingToken,
  resolveEngineSettings,
  saveEngineSettings,
  TUNNEL_SUFFIXES
} from "../src/engine.js";

const TOKEN = "abcDEF123_-abcDEF123_-xy";

function memoryStorage() {
  const data = new Map();
  return {
    getItem: (key) => (data.has(key) ? data.get(key) : null),
    setItem: (key, value) => data.set(key, String(value)),
    removeItem: (key) => data.delete(key)
  };
}

test("empty address means the engine on this computer", () => {
  assert.deepEqual(normalizeEngineAddress(""), { ok: true, origin: DEFAULT_ENGINE_ORIGIN, remote: false });
  assert.deepEqual(normalizeEngineAddress("127.0.0.1:8765"), { ok: true, origin: "http://127.0.0.1:8765", remote: false });
});

test("tunnel addresses are accepted with or without the scheme", () => {
  const expected = { ok: true, origin: "https://calm-river.trycloudflare.com", remote: true };
  assert.deepEqual(normalizeEngineAddress("calm-river.trycloudflare.com"), expected);
  assert.deepEqual(normalizeEngineAddress(" https://calm-river.trycloudflare.com/health "), expected);
  assert.equal(normalizeEngineAddress("https://pc.tail1234.ts.net").remote, true);
});

test("plain http, LAN addresses and unknown hosts are refused", () => {
  for (const bad of ["http://calm-river.trycloudflare.com", "https://192.168.1.20:8765", "https://example.com",
    "https://trycloudflare.com.evil.example", "https://user:pw@a.trycloudflare.com", "https://.trycloudflare.com", "::::"]) {
    assert.equal(normalizeEngineAddress(bad).ok, false, bad);
  }
});

test("pairing code format", () => {
  assert.deepEqual(normalizePairingToken(` ${TOKEN} `), { ok: true, token: TOKEN });
  assert.equal(normalizePairingToken("short").ok, false);
  assert.equal(normalizePairingToken("has spaces in the middle of it").ok, false);
  assert.deepEqual(normalizePairingToken(""), { ok: true, token: "" });
});

test("a remote engine requires a pairing code", () => {
  assert.equal(resolveEngineSettings("a.trycloudflare.com", "").ok, false);
  assert.deepEqual(resolveEngineSettings("a.trycloudflare.com", TOKEN),
    { ok: true, origin: "https://a.trycloudflare.com", remote: true, token: TOKEN });
  assert.equal(resolveEngineSettings("", "").remote, false);
});

test("settings survive a save/load round trip and bad data falls back to local", () => {
  const storage = memoryStorage();
  const settings = resolveEngineSettings("a.trycloudflare.com", TOKEN);
  assert.equal(saveEngineSettings(storage, settings), true);
  assert.deepEqual(loadEngineSettings(storage), settings);
  saveEngineSettings(storage, { origin: DEFAULT_ENGINE_ORIGIN, token: "" });
  assert.equal(loadEngineSettings(storage).origin, DEFAULT_ENGINE_ORIGIN);
  storage.setItem("retake-engine", "{not json");
  assert.equal(loadEngineSettings(storage).remote, false);
  storage.setItem("retake-engine", JSON.stringify({ origin: "https://evil.example", token: TOKEN }));
  assert.equal(loadEngineSettings(storage).origin, DEFAULT_ENGINE_ORIGIN);
  assert.equal(loadEngineSettings(null).origin, DEFAULT_ENGINE_ORIGIN);
});

test("the pairing header is only sent when a code is set", () => {
  assert.deepEqual(engineHeaders({ token: "" }, { Accept: "x" }), { Accept: "x" });
  assert.deepEqual(engineHeaders({ token: TOKEN }), { "X-Retake-Token": TOKEN });
});

test("elapsed time formatting", () => {
  assert.equal(formatElapsed(0), "00:00");
  assert.equal(formatElapsed(75.9), "01:15");
  assert.equal(formatElapsed(-3), "00:00");
});

test("every accepted tunnel suffix is allowed by the page's connect-src", () => {
  const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
  const csp = /http-equiv="Content-Security-Policy" content="([^"]+)"/.exec(html)[1];
  const connect = csp.split(";").find((part) => part.trim().startsWith("connect-src"));
  for (const suffix of TUNNEL_SUFFIXES) assert.ok(connect.includes("https://*" + suffix), suffix);
});
