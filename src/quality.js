// Reading the engine's ArcFace scores for display. ArcFace cosine similarity is
// shown as a 0–100 score; it is a resemblance measure, not proof of identity.

export function parseSimilarity(value) {
  if (value === null || value === undefined || String(value).trim() === "") return null;
  const number = Number(value);
  return Number.isFinite(number) && number >= -1 && number <= 1 ? number : null;
}

export function similarityScore(cosine) {
  return cosine === null || cosine === undefined ? null : Math.max(0, Math.min(100, Math.round(cosine * 100)));
}

// Thresholds follow common ArcFace (w600k) practice: same-person photos
// usually score above ~0.4; above ~0.55 is a strong match.
export function similarityVerdict(score) {
  if (score === null || score === undefined) return "";
  if (score >= 55) return "Çok güçlü";
  if (score >= 42) return "Güçlü";
  if (score >= 30) return "Orta";
  return "Zayıf";
}

export function describeSimilarity(after, before) {
  const score = similarityScore(after);
  if (score === null) return "";
  const previous = similarityScore(before);
  return `Kimlik benzerliği ${score} · ${similarityVerdict(score)}` + (previous === null ? "" : ` · orijinal ${previous}`);
}

/** Index of the most similar result, or -1 when none has a score. */
export function bestResultIndex(results) {
  let best = -1;
  (Array.isArray(results) ? results : []).forEach((item, index) => {
    if (typeof item?.similarity !== "number") return;
    if (best === -1 || item.similarity > results[best].similarity) best = index;
  });
  return best;
}

/** Profile health from the engine's ArcFace screen of the 25 references. */
export function describeProfileHealth(medianValue, outlierValue) {
  const median = parseSimilarity(medianValue);
  const outliers = String(outlierValue || "")
    .split(",")
    .map((value) => Number(value.trim()))
    .filter((value) => Number.isInteger(value) && value > 0);
  if (median === null && outliers.length === 0) return null;
  const parts = [];
  if (median !== null) {
    const score = similarityScore(median);
    parts.push(`Referans uyumu ${score} · ${similarityVerdict(score)}`);
  }
  parts.push(outliers.length
    ? `Diğerlerine benzemeyen: ${outliers.map((n) => "Referans " + String(n).padStart(2, "0")).join(", ")} (etkisi azaltıldı)`
    : "Aykırı fotoğraf yok");
  return { median, outliers, text: parts.join(" · "), warn: outliers.length > 0 || (median !== null && median < 0.3) };
}
