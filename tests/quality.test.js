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
  assert.equal(describeSimilarity(0.74, 0.21), "Kimlik benzerliği 74 · Çok güçlü · orijinal 21");
  assert.equal(describeSimilarity(0.44, null), "Kimlik benzerliği 44 · Güçlü");
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
  assert.equal(describeSimilarity(0.80, 0.05, 0.766), "Kimlik benzerliği 80 · Çok güçlü · fotoğraflarınla aynı düzeyde (77) · orijinal 5");
  assert.equal(describeSimilarity(0.80, null), "Kimlik benzerliği 80 · Çok güçlü");
});
