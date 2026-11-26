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
  assert.match(report,/Şu an bağlı motorun kalite sürümü: portrait-quality-20261126-r9/);
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


test("local comparison distinguishes same input from different preparation and settings", async () => {
  const {createQualityBaseline,compareQualityBaseline,readQualityBaseline}=await import("../src/diagnostics.js");
  const context={input:"a".repeat(64),prepared:"b".repeat(64)};
  const headers={"x-retake-detail-gate":"passed","x-retake-similarity":".832",
    "x-retake-detail-score":".80","x-retake-ultra-timing":"total=1560",
    "x-retake-quality-revision":"portrait-quality-20261126-r9"};
  const baseline=createQualityBaseline(context,headers,1000);
  const current=createQualityBaseline({...context,prepared:"c".repeat(64)},{...headers,"x-retake-similarity":".85",
    "x-retake-detail-score":".82","x-retake-ultra-timing":"total=1200"},2000);
  const report=compareQualityBaseline(baseline,current);
  assert.match(report,/Hazırlanan motor girdileri farklı/);
  assert.match(report,/83,20 → 85,00 \(\+1,80\)/);
  assert.match(report,/süre 1560 → 1200 sn/);
  assert.doesNotMatch(report,/[a-f0-9]{64}/);
  assert.match(compareQualityBaseline(baseline,{...current,input:"d".repeat(64)}),/Kıyas yapılamadı/);
  assert.equal(createQualityBaseline(context,{...headers,"x-retake-detail-gate":"failed"}),null);
  assert.equal(createQualityBaseline(context,{...headers,"x-retake-similarity":""}),null);
  assert.deepEqual(readQualityBaseline({getItem:()=>JSON.stringify(baseline)}),baseline);
  assert.equal(readQualityBaseline({getItem:()=>"{invalid"}),null);
});

test("comparison context depends on bytes, source face and settings, not filenames", async () => {
  const {comparisonContext}=await import("../src/diagnostics.js");
  const {webcrypto}=await import("node:crypto");
  if(!globalThis.crypto) Object.defineProperty(globalThis,"crypto",{value:webcrypto,configurable:true});
  const args={target:new Blob(["same photo"]),referenceKey:"f".repeat(64),faceIndex:0,
    depth:"max",judge:true,full:false,benchmark:false,prepared:[new Blob(["crop"])],metadata:{mode:"both"}};
  const a=await comparisonContext(args);
  assert.ok(a);
  assert.deepEqual(await comparisonContext({...args,target:new Blob(["same photo"])}),a);
  assert.notEqual((await comparisonContext({...args,depth:"deep"})).input,a.input);
  assert.notEqual((await comparisonContext({...args,faceIndex:1})).input,a.input);
  const other=await comparisonContext({...args,prepared:[new Blob(["other crop"])]});
  assert.equal(other.input,a.input);
  assert.notEqual(other.prepared,a.prepared);
  assert.equal(await comparisonContext({...args,benchmark:true}),null);
});


test("identity pixel scope is distinguished from browser export", async () => {
  const {summarizeQualityEvidence}=await import("../src/diagnostics.js");
  assert.match(summarizeQualityEvidence({"x-retake-identity-pixel-scope":"visible-engine-crop"}), /tarayıcı ölçeklemesi öncesi/);
  assert.equal(summarizeQualityEvidence({"x-retake-identity-pixel-scope":"unverified"}),"");
});
