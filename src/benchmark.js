// Self-test benchmark ("Öz-test kıyaslaması").
//
// The person's own reference photos are the only targets with a known right
// answer: the original face. Running Ultra on a few of them (each strictly
// leave-one-out on the engine) measures how natural Retake really is for this
// person, and every round also teaches the engine's learned variant order.
// Only numbers are kept here; no image leaves the browser ↔ local engine path.

export const BENCHMARK_SIZES = Object.freeze([3, 5, 10, 25]);
export const DELTA_E_WEIGHT = 0.02; // same as the engine's fidelity score

const SELF_FIDELITY = /^ssim:(-?\d+(?:\.\d+)?),deltaE:(\d+(?:\.\d+)?)$/;

export function parseSelfFidelity(value) {
  const match = SELF_FIDELITY.exec(String(value ?? "").trim());
  if (!match) return null;
  const ssim = Number(match[1]);
  const deltaE = Number(match[2]);
  if (!Number.isFinite(ssim) || ssim < -1 || ssim > 1 || !Number.isFinite(deltaE) || deltaE < 0 || deltaE > 200) return null;
  return { ssim, deltaE };
}

export function fidelityScore(ssim, deltaE) {
  if (!Number.isFinite(ssim) || !Number.isFinite(deltaE)) return null;
  return Math.max(0, Math.min(1, ssim - DELTA_E_WEIGHT * deltaE));
}

// Evenly spread over the bank, so different poses and light are covered.
export function pickBenchmarkReferences(references, count) {
  const ready = (Array.isArray(references) ? references : [])
    .map((item, index) => ({ item, index }))
    .filter(({ item }) => item && item.status === "ready" && item.blob && typeof item.blob.size === "number" && item.blob.size > 0);
  const size = Math.max(0, Math.min(ready.length, Number.isInteger(count) ? count : 0));
  if (!size) return [];
  if (size === ready.length) return ready;
  const step = ready.length / size;
  const picked = [];
  for (let slot = 0; slot < size; slot += 1) {
    picked.push(ready[Math.min(ready.length - 1, Math.floor(slot * step + step / 2))]);
  }
  return picked;
}

function mean(values) {
  return values.length ? values.reduce((total, value) => total + value, 0) / values.length : null;
}

function round(value, digits) {
  return value === null ? null : Number(value.toFixed(digits));
}

export function summarizeBenchmark(rows) {
  const list = Array.isArray(rows) ? rows : [];
  const measured = list.filter((row) => row && !row.error && Number.isFinite(row.ssim) && Number.isFinite(row.deltaE));
  const scored = measured.map((row) => ({ ...row, fidelity: fidelityScore(row.ssim, row.deltaE) }));
  const ordered = [...scored].sort((a, b) => b.fidelity - a.fidelity);
  return {
    total: list.length,
    measured: measured.length,
    failed: list.filter((row) => row && row.error).length,
    meanSsim: round(mean(measured.map((row) => row.ssim)), 4),
    meanDeltaE: round(mean(measured.map((row) => row.deltaE)), 3),
    meanFidelity: round(mean(scored.map((row) => row.fidelity)), 4),
    best: ordered[0] ? { file: ordered[0].file, fidelity: round(ordered[0].fidelity, 4) } : null,
    worst: ordered.length > 1 ? { file: ordered.at(-1).file, fidelity: round(ordered.at(-1).fidelity, 4) } : null
  };
}

export function buildBenchmarkReport(rows, meta = {}) {
  const allowedRow = ({ index, file, ssim, deltaE, similarity, before, sufficient, schedule, winner, error }) => ({
    index, file, ssim: ssim ?? null, deltaE: deltaE ?? null,
    fidelity: Number.isFinite(ssim) && Number.isFinite(deltaE) ? round(fidelityScore(ssim, deltaE), 4) : null,
    similarity: similarity ?? null, before: before ?? null, sufficient: sufficient || "",
    schedule: schedule || "", winner: winner ?? null, error: error || ""
  });
  return {
    kind: "retake-self-test-benchmark",
    version: 1,
    createdAt: meta.createdAt || new Date().toISOString(),
    engineModel: meta.engineModel || "",
    note: "Yalnız sayılar; görsel içermez. SSIM 1 ve ΔE 0 orijinalle aynı demektir. Aynı fotoğrafın JPEG yeniden kaydı ≈ 0,97 / 1,3.",
    summary: summarizeBenchmark(rows),
    rows: (Array.isArray(rows) ? rows : []).map(allowedRow)
  };
}

// Sequential controller; the UI passes in how to run one self-test.
export async function runBenchmark({ items, runOne, onProgress = () => {}, shouldStop = () => false }) {
  const rows = [];
  for (let position = 0; position < items.length; position += 1) {
    if (shouldStop()) break;
    const { item, index } = items[position];
    onProgress({ position, total: items.length, file: item.file, rows: [...rows] });
    try {
      const result = await runOne(item, index);
      const fidelity = parseSelfFidelity(result?.selfFidelity);
      rows.push({
        index, file: item.file,
        ssim: fidelity?.ssim ?? null, deltaE: fidelity?.deltaE ?? null,
        similarity: Number.isFinite(result?.similarity) ? result.similarity : null,
        before: Number.isFinite(result?.before) ? result.before : null,
        sufficient: result?.identitySufficient || "",
        schedule: result?.ultraSchedule || "",
        winner: Number.isInteger(result?.winnerRound) ? result.winnerRound : null,
        error: fidelity ? "" : "Motor öz-test ölçümü döndürmedi"
      });
    } catch (error) {
      rows.push({ index, file: item.file, error: String(error?.message || error || "Bilinmeyen hata").slice(0, 200) });
    }
  }
  onProgress({ position: rows.length, total: items.length, file: "", rows: [...rows], done: true });
  return rows;
}
