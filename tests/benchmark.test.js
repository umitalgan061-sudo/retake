import test from "node:test";
import assert from "node:assert/strict";
import {
  parseSelfFidelity, fidelityScore, pickBenchmarkReferences, summarizeBenchmark,
  buildBenchmarkReport, runBenchmark, BENCHMARK_SIZES
} from "../src/benchmark.js";

const refs = (count, broken = []) => Array.from({ length: count }, (_, index) => ({
  status: broken.includes(index) ? "error" : "ready",
  file: `ref-${index}.jpg`,
  blob: { size: 100 + index }
}));

test("parses the engine's self-test header strictly", () => {
  assert.deepEqual(parseSelfFidelity("ssim:0.6929,deltaE:7.152"), { ssim: 0.6929, deltaE: 7.152 });
  for (const bad of ["", "ssim:2,deltaE:1", "ssim:0.9,deltaE:-1", "ssim:0.9", "<b>", null]) {
    assert.equal(parseSelfFidelity(bad), null);
  }
  assert.ok(Math.abs(fidelityScore(0.97, 1.3) - 0.944) < 1e-9);
  assert.equal(fidelityScore(0.1, 50), 0);
  assert.equal(fidelityScore(NaN, 1), null);
});

test("spreads picks across the ready bank and skips broken photos", () => {
  const picked = pickBenchmarkReferences(refs(25), 5);
  assert.deepEqual(picked.map(({ index }) => index), [2, 7, 12, 17, 22]);
  assert.equal(pickBenchmarkReferences(refs(25, [2]), 25).length, 24);
  assert.ok(pickBenchmarkReferences(refs(25, [2]), 5).every(({ item }) => item.status === "ready"));
  assert.deepEqual(pickBenchmarkReferences(refs(3), 0), []);
  assert.deepEqual(pickBenchmarkReferences(null, 3), []);
  assert.deepEqual(BENCHMARK_SIZES, [3, 5, 10, 25]);
});

test("summary and report keep numbers only", () => {
  const rows = [
    { index: 2, file: "a.jpg", ssim: 0.90, deltaE: 2.0, similarity: 0.86 },
    { index: 7, file: "b.jpg", ssim: 0.70, deltaE: 7.0 },
    { index: 9, file: "c.jpg", error: "x" }
  ];
  const summary = summarizeBenchmark(rows);
  assert.equal(summary.measured, 2);
  assert.equal(summary.failed, 1);
  assert.equal(summary.meanSsim, 0.8);
  assert.equal(summary.meanDeltaE, 4.5);
  assert.equal(summary.best.file, "a.jpg");
  assert.equal(summary.worst.file, "b.jpg");
  const report = buildBenchmarkReport([...rows, { index: 1, file: "d.jpg", ssim: 0.8, deltaE: 3, image: "data:image/png;base64,AAAA" }], { engineModel: "v22" });
  assert.equal(report.kind, "retake-self-test-benchmark");
  assert.equal(report.rows[0].fidelity, 0.86);
  assert.ok(!JSON.stringify(report).includes("base64"));
});

test("controller runs sequentially, isolates failures and honours stop", async () => {
  const order = [];
  let stop = false;
  const items = pickBenchmarkReferences(refs(25), 3);
  const progress = [];
  const rows = await runBenchmark({
    items,
    runOne: async (item, index) => {
      order.push(index);
      if (index === 12) throw new Error("Yüz bulunamadı");
      if (order.length === 3) stop = true;
      return { selfFidelity: "ssim:0.9000,deltaE:2.000", similarity: 0.85, ultraSchedule: "learned-fidelity", winnerRound: 4 };
    },
    onProgress: (state) => progress.push(state),
    shouldStop: () => stop
  });
  assert.deepEqual(order, [4, 12, 20]);
  assert.equal(rows[1].error, "Yüz bulunamadı");
  assert.equal(rows[0].ssim, 0.9);
  assert.equal(rows[0].schedule, "learned-fidelity");
  assert.equal(progress.at(-1).done, true);

  stop = false;
  const missing = await runBenchmark({ items: items.slice(0, 1), runOne: async () => ({}) });
  assert.match(missing[0].error, /ölçüm/);
  const stopped = await runBenchmark({ items, runOne: async () => ({}), shouldStop: () => true });
  assert.deepEqual(stopped, []);
});
