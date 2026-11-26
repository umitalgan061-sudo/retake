import test from "node:test";
import assert from "node:assert/strict";
import {
  buildDiagnosticReport, collectDiagnosticHeaders, describeUltraTiming, parseUltraTiming, scrub
} from "../src/diagnostics.js";

test("paths, file names and secrets never reach the report", () => {
  assert.equal(scrub("C:\\Users\\Ümit\\.retake\\token.txt failed"), "[yol] failed");
  assert.equal(scrub("open /home/u/x/retake-fine-features.png now"), "open [yol] now");
  assert.equal(scrub("retake-fine-features.png"), "[dosya]");
  assert.equal(scrub("search=19x72;total=2210"), "search=19x72;total=2210");
  const headers = new Map([
    ["X-Retake-Similarity", "0.9123"],
    ["X-Retake-Pairing-Token", "abc"],
    ["X-Retake-Api-Key", "secret"],
    ["Content-Type", "image/png"],
    ["x-retake-ultra-timing", "search=3x72;total=300"],
  ]);
  const collected = collectDiagnosticHeaders(headers);
  assert.deepEqual(Object.keys(collected), ["x-retake-similarity", "x-retake-ultra-timing"]);
});

test("timing header is described in plain Turkish", () => {
  assert.deepEqual(parseUltraTiming("search=19x72;refine=4x31;judge=40;total=2210;bad=x"), {
    search: { count: 19, seconds: 72 }, refine: { count: 4, seconds: 31 },
    judge: { seconds: 40 }, total: { seconds: 2210 }
  });
  assert.equal(
    describeUltraTiming("search=19x72;refine=4x31;realism=2x210;judge=40;hold=0;total=2210"),
    "Süre: 19 tur × 72 sn · 4 ince ayar × 31 sn · 2 gerçekçilik × 4 dk · jüri 40 sn · toplam 37 dk"
  );
  assert.equal(describeUltraTiming(""), "");
  assert.equal(describeUltraTiming(null), "");
});

test("the report lists measurements and a whitelisted engine summary", () => {
  const report = buildDiagnosticReport({
    headers: { "x-retake-similarity": "0.9123", "x-retake-ultra-timing": "search=2x60;total=420" },
    health: {
      model: "retake_hybrid_25_reference_ultra_v22",
      identity_scorer: { enabled: true, calls: 12, last_error: "C:\\AI\\facefusion\\x.onnx missing" },
      facefusion_dir: "C:\\AI\\facefusion",
      pairing_token: "never"
    },
    depth: "deep", appVersion: "retake-shell-v36", createdAt: new Date("2026-11-26T04:00:00Z")
  });
  assert.match(report, /^Retake tanı raporu/);
  assert.match(report, /Seçilen derinlik: deep/);
  assert.match(report, /Süre: 2 tur × 60 sn · toplam 7 dk/);
  assert.match(report, /x-retake-similarity: 0\.9123/);
  assert.match(report, /model: retake_hybrid_25_reference_ultra_v22/);
  assert.match(report, /identity_scorer: \{"enabled":true,"calls":12,"last_error":"\[yol\] missing"\}/);
  assert.doesNotMatch(report, /facefusion_dir|never|AI\\\\facefusion/);
});


test("result revision is not silently replaced by a newer connected motor", async () => {
  const {EXPECTED_QUALITY_REVISION,describeQualityCompatibility}=await import("../src/diagnostics.js");
  const old="portrait-quality-20261125-r2";
  assert.equal(describeQualityCompatibility(EXPECTED_QUALITY_REVISION),"");
  assert.equal(describeQualityCompatibility("portrait-quality-20261126-r10"),"");
  assert.match(describeQualityCompatibility(old),/eski kalite paketi/);
  assert.match(describeQualityCompatibility(undefined),/doğrulanamadı/);
  const report=buildDiagnosticReport({headers:{"x-retake-quality-revision":old,"x-retake-token":"private"},
    health:{quality_revision:EXPECTED_QUALITY_REVISION,flux_texture_alternatives:true}});
  assert.match(report,/Sonucu üreten kalite sürümü: portrait-quality-20261125-r2/);
  assert.match(report,/Şu an bağlı motorun kalite sürümü: portrait-quality-20261126-r7/);
  assert.match(report,/güncel motor durumu bu sonucun kanıtı değildir/);
  assert.match(report,/flux_texture_alternatives: true/);
  assert.doesNotMatch(report,/private|x-retake-token/);
  assert.match(buildDiagnosticReport({health:{quality_revision:EXPECTED_QUALITY_REVISION}}),/Sonucu üreten kalite sürümü: doğrulanamadı/);
});


test("diagnostic quality gaps use only verified numerical headers", async () => {
  const {summarizeQualityEvidence}=await import("../src/diagnostics.js");
  const report=summarizeQualityEvidence({"x-retake-similarity":".832","x-retake-ultra-peak-identity":".835","x-retake-identity-aspiration":".87","x-retake-detail-gate":"passed"});
  assert.match(report,/arama zirvesinden fark 0,30 puan/);
  assert.match(report,/arama hedefinden fark -3,80 puan/);
  assert.equal(summarizeQualityEvidence({"x-retake-similarity":"","x-retake-ultra-peak-identity":"nan"}),"");
});
