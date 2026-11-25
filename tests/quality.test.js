import test from "node:test";
import assert from "node:assert/strict";
import {
  bestResultIndex,
  describeProfileHealth,
  describeSimilarity,
  parseSimilarity,
  similarityScore,
  similarityVerdict
} from "../src/quality.js";

test("engine scores are parsed defensively", () => {
  assert.equal(parseSimilarity("0.7412"), 0.7412);
  for (const bad of ["", null, undefined, "abc", "1.5", "-2"]) assert.equal(parseSimilarity(bad), null);
});

test("scores and verdicts", () => {
  assert.equal(similarityScore(0.736), 74);
  assert.equal(similarityScore(-0.2), 0);
  assert.equal(similarityScore(null), null);
  assert.deepEqual([60, 45, 31, 12].map(similarityVerdict), ["Çok güçlü", "Güçlü", "Orta", "Zayıf"]);
  assert.equal(describeSimilarity(0.74, 0.21), "Kimlik benzerliği 74,0 · Çok güçlü · orijinal 21");
  assert.equal(describeSimilarity(0.44, null), "Kimlik benzerliği 44,0 · Güçlü");
  assert.match(describeSimilarity(0.8334, 0.0), /83,3 · Çok güçlü/);
  assert.match(describeSimilarity(0.8279, 0.0), /82,8 · Çok güçlü/);
  assert.equal(describeSimilarity(null, 0.2), "");
});

test("the most similar look is picked; unscored looks are skipped", () => {
  assert.equal(bestResultIndex([{ similarity: 0.6 }, {}, { similarity: 0.71 }, { similarity: 0.5 }]), 2);
  assert.equal(bestResultIndex([{}, {}]), -1);
  assert.equal(bestResultIndex(null), -1);
});

test("profile health names the references that do not match", () => {
  const health = describeProfileHealth("0.765743", "2,17");
  assert.equal(health.text, "Referans uyumu 77 · Çok güçlü · Diğerlerine benzemeyen: Referans 02, Referans 17 (etkisi azaltıldı)");
  assert.equal(health.warn, true);
  assert.equal(describeProfileHealth("0.7", "").text, "Referans uyumu 70 · Çok güçlü · Aykırı fotoğraf yok");
  assert.equal(describeProfileHealth("", ""), null);
});

test("the result is compared with the person's own photos", async () => {
  const { describeSimilarity, referenceComparison } = await import("../src/quality.js");
  assert.equal(referenceComparison(0.80, 0.766), "fotoğraflarınla aynı düzeyde (77)");
  assert.equal(referenceComparison(0.76, 0.766), "fotoğraflarınla aynı düzeyde (77)");
  assert.equal(referenceComparison(0.62, 0.766), "fotoğraflarının düzeyi 77");
  assert.equal(referenceComparison(0.62, null), "");
  assert.equal(describeSimilarity(0.80, 0.05, 0.766), "Kimlik benzerliği 80,0 · Çok güçlü · fotoğraflarınla aynı düzeyde (77) · orijinal 5");
  assert.equal(describeSimilarity(0.80, null), "Kimlik benzerliği 80,0 · Çok güçlü");
});

test("original cosine below zero is not silently displayed as zero", () => {
  assert.match(describeSimilarity(.87, -.12), /orijinal -12,00 \(kosinüs\)/);
  assert.match(describeSimilarity(.87, .001), /orijinal 0,10 \(kosinüs\)/);
});
test("FLUX configuration is not evidence of a selected candidate", async () => {
  const {describeFluxStatus} = await import("../src/quality.js");
  assert.match(describeFluxStatus("accepted"), /kullanıldı/);
  assert.match(describeFluxStatus("worker-error"), /hatası/);
  assert.match(describeFluxStatus("quality-rejected"), /elendi/);
  assert.equal(describeFluxStatus("unknown"), "");
});
test("NVIDIA jury badge reports the real outcome and never invents numbers", async () => {
  const {describeNvidiaJudge} = await import("../src/quality.js");
  assert.equal(describeNvidiaJudge("switched", 6, 0.833), "NVIDIA jürisi daha doğal görünen finalisti seçti (6 oy, %83 uzlaşı)");
  assert.equal(describeNvidiaJudge("kept", null, null), "NVIDIA jürisi yerel seçimi onayladı");
  assert.match(describeNvidiaJudge("abstain", 6, 0.5), /yerel seçim korundu/);
  assert.match(describeNvidiaJudge("error"), /ulaşılamadı/);
  assert.equal(describeNvidiaJudge("off"), "");
  assert.equal(describeNvidiaJudge("not-consented"), "");
  assert.equal(describeNvidiaJudge("switched", 6, 7), "NVIDIA jürisi daha doğal görünen finalisti seçti");
});


test("adaptive result and diagnostic labels do not imply missing data or certainty", async () => {
  const {describeAdaptiveQuality,describeFluxMetrics,describeJudgeError} = await import("../src/quality.js");
  assert.match(describeAdaptiveQuality(.796,.06,.835,2), /hedefi karşılanamadı/);
  assert.match(describeAdaptiveQuality(.85,.06,.835,2), /hedefi karşılandı/);
  assert.equal(describeAdaptiveQuality(.8,.1,null,1),"");
  assert.equal(describeFluxMetrics(null,"",""),"");
  assert.match(describeFluxMetrics(-.01,.03,.2),/kimlik farkı -1,0/);
  assert.match(describeJudgeError("http-401"),/yetkilendirmesi/);
  assert.equal(describeJudgeError("token=secret"),"");
});
test("identity sufficiency badge only speaks when the engine measured it", async () => {
  const {describeIdentitySufficiency} = await import("../src/quality.js");
  assert.match(describeIdentitySufficiency("yes"), /en doğal/);
  assert.equal(describeIdentitySufficiency("no"), "");
  assert.equal(describeIdentitySufficiency("unknown"), "");
  assert.equal(describeIdentitySufficiency(undefined), "");
});
test("self-test fidelity is shown only for a well-formed engine measurement", async () => {
  const {describeSelfFidelity} = await import("../src/quality.js");
  assert.equal(describeSelfFidelity("ssim:0.6929,deltaE:7.152"), "Öz-test (fotoğraf referanslarında da var): orijinale yapısal benzerlik 0,69 · renk farkı ΔE 7,2 (aynı fotoğrafın JPEG yeniden kaydı ≈ 0,97 · 1,3)");
  assert.equal(describeSelfFidelity(""), "");
  assert.equal(describeSelfFidelity("ssim:2,deltaE:1"), "");
  assert.equal(describeSelfFidelity("ssim:0.9,deltaE:-1"), "");
  assert.equal(describeSelfFidelity("<script>"), "");
});
