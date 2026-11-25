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

// How the result compares with the person's own photos: their typical
// similarity to each other is the realistic ceiling for "looks like them".
export function referenceComparison(after, referenceMedian) {
  const score = similarityScore(after);
  const level = similarityScore(parseSimilarity(referenceMedian));
  if (score === null || level === null || level <= 0) return "";
  if (score >= level - 2) return `fotoğraflarınla aynı düzeyde (${level})`;
  return `fotoğraflarının düzeyi ${level}`;
}

export function describeSimilarity(after, before, referenceMedian = null) {
  const score = similarityScore(after);
  if (score === null) return "";
  const previous = similarityScore(before);
  const comparison = referenceComparison(after, referenceMedian);
  // Scores shown as integers hide genuine small changes: 0.825–0.834
  // all display as 83. Show one decimal, never claim a calibrated percent.
  const decimalScore = (Number(after) * 100).toFixed(1).replace(".", ",");
  return `Kimlik benzerliği ${decimalScore} · ${similarityVerdict(score)}` +
    (comparison ? ` · ${comparison}` : "") +
    (previous === null ? "" : ` · orijinal ${previous}`);
}

const NATURALNESS_WORDS = { border: "kenar", tone: "ten uyumu", grain: "gren", detail: "detay", light: "ışık" };

/** Engine's photographic naturalness (0–100) and its checks ("border=89,tone=100,…"). */
export function parseNaturalness(value, checksValue = "") {
  if (value === null || value === undefined || String(value).trim() === "") return null;
  const score = Number(value);
  if (!Number.isFinite(score) || score < 0 || score > 100) return null;
  const checks = {};
  String(checksValue || "").split(",").forEach((part) => {
    const [name, raw] = part.split("=");
    const number = Number(raw);
    if (name && Number.isFinite(number)) checks[name.trim()] = number;
  });
  return { score, checks };
}

/** "Gerçeklik 88/100" plus the weakest check when one clearly lags. */
export function describeNaturalness(naturalness) {
  if (!naturalness || !Number.isFinite(naturalness.score)) return "";
  const text = `Gerçeklik ${Math.round(naturalness.score)}/100`;
  const entries = Object.entries(naturalness.checks || {});
  if (!entries.length) return text;
  const [name, value] = entries.reduce((low, item) => (item[1] < low[1] ? item : low));
  return value < 70 ? `${text} · zayıf: ${NATURALNESS_WORDS[name] || name}` : text;
}

const JUDGE_WORDS = {
  seam: "kenar dikişi", plastic_skin: "plastik cilt", tone: "ten rengi", lighting: "ışık",
  eyes: "gözler", mouth: "ağız/diş", blur: "bulanıklık", oversharp: "aşırı keskinlik",
  warp: "şekil bozulması", hair: "saç çizgisi", noise: "gren/gürültü"
};

/** NVIDIA vision judge verdict from the engine headers (0–10 realism). */
export function parseJudge(value, issuesValue = "", switched = "") {
  if (value === null || value === undefined || String(value).trim() === "") return null;
  const score = Number(value);
  if (!Number.isFinite(score) || score < 0 || score > 10) return null;
  const issues = String(issuesValue || "").split(",").map((part) => part.trim()).filter((part) => part in JUDGE_WORDS);
  return { score, issues, switched: switched === "yes" };
}

/** "NVIDIA hakem 8,5/10 · gözler" — a second opinion, not a guarantee. */
export function describeJudge(judge) {
  if (!judge || !Number.isFinite(judge.score)) return "";
  const text = `NVIDIA hakem ${judge.score.toFixed(1).replace(".", ",")}/10`;
  const issues = judge.issues.slice(0, 2).map((issue) => JUDGE_WORDS[issue]);
  return text + (issues.length ? ` · ${issues.join(", ")}` : "") + (judge.switched ? " · hakem seçti" : "");
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


// Native-resolution face-interior change diagnostic. Not a beauty or identity
// score. Ignore perimeter changes: a seam alone cannot certify face editing.
export function measureFaceEdit(source,output,width,height,bounds){
  if(source.length!==width*height*4||output.length!==source.length)throw new RangeError("Görüntü ölçüsü geçersiz.");
  const {x,y,width:w,height:h}=bounds;
  if(![x,y,w,h].every(Number.isFinite)||w<=0||h<=0)throw new RangeError("Yüz alanı geçersiz.");
  let count=0,changed=0,sum=0;
  for(let py=Math.max(0,Math.floor(y+h*.11));py<Math.min(height,Math.ceil(y+h*.89));py++){
    for(let px=Math.max(0,Math.floor(x+w*.14));px<Math.min(width,Math.ceil(x+w*.86));px++){
      if(((px+.5-x-w*.5)/(w*.36))**2+((py+.5-y-h*.5)/(h*.39))**2>1)continue;
      const i=(py*width+px)*4;
      const delta=Math.max(Math.abs(source[i]-output[i]),Math.abs(source[i+1]-output[i+1]),Math.abs(source[i+2]-output[i+2]));
      count++;sum+=delta;if(delta>=8)changed++;
    }
  }
  const changedRatio=count?changed/count:0,meanDelta=count?sum/count:0;
  return {changedRatio,meanDelta,meaningful:changedRatio>=.03&&meanDelta>=1.5};
}
