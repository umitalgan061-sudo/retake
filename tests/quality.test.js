import test from "node:test";
import assert from "node:assert/strict";
import {
  bestResultIndex,
  describeProfileHealth,
  describeSimilarity,
  parseSimilarity,
  similarityScore,
  similarityVerdict,
  describeUltraDepth,
  describeFineFeatures,
  parseFineDetail
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

test("the engine's naturalness score reads as a short verdict", async () => {
  const { describeNaturalness, parseNaturalness } = await import("../src/quality.js");
  const good = parseNaturalness("88.4", "border=89,tone=100,grain=86,detail=83,light=97");
  assert.equal(good.score, 88.4);
  assert.equal(good.checks.tone, 100);
  assert.equal(describeNaturalness(good), "Gerçeklik 88/100");
  const weak = parseNaturalness("61", "border=28,tone=73,grain=100,detail=90,light=73");
  assert.equal(describeNaturalness(weak), "Gerçeklik 61/100 · zayıf: kenar");
  assert.equal(parseNaturalness(""), null);
  assert.equal(parseNaturalness("140"), null);
  assert.equal(describeNaturalness(null), "");
  assert.equal(describeNaturalness(parseNaturalness("90")), "Gerçeklik 90/100");
});

test("original cosine below zero is not silently displayed as zero", () => {
  assert.match(describeSimilarity(.87, -.12), /orijinal -0,1200 \(kosinüs\)/);
  assert.match(describeSimilarity(.87, .001), /orijinal 0,0010 \(kosinüs\)/);
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
test("learned schedule badge names its evidence", async () => {
  const {describeUltraSchedule} = await import("../src/quality.js");
  assert.match(describeUltraSchedule("learned-fidelity", 9), /öz-testlerde.*\(9 ölçüm\)/);
  assert.match(describeUltraSchedule("learned-proxy", 12), /\(12 tur\)/);
  assert.equal(describeUltraSchedule("learned-proxy", null).includes("("), false);
  assert.equal(describeUltraSchedule("default", 5), "");
  assert.equal(describeUltraSchedule(undefined), "");
});

test("describes what a deeper Ultra search did", () => {
  assert.equal(describeUltraDepth("standard", "0/0", 1), "");
  assert.equal(describeUltraDepth("", "", null), "");
  assert.equal(
    describeUltraDepth("deep", "1/3", 2),
    "Derin arama · ince ayar sonucu 1 adımda iyileştirdi (3 denendi) · 2 gerçekçilik finalisti karşılaştırıldı"
  );
  assert.equal(describeUltraDepth("max", "0/6", 1), "En yüksek kalite araması · 6 ince ayar denendi, en iyi ızgara ayarı korundu");
  assert.equal(describeUltraDepth("deep", "garbage", "2"), "Derin arama");
});

test("fine feature headers are parsed defensively", () => {
  assert.deepEqual(parseFineDetail("eye_r:0.610>0.970,lips:1.020>1.010"), [
    { name: "eye_r", label: "sağ göz", before: 0.61, after: 0.97 },
    { name: "lips", label: "dudak", before: 1.02, after: 1.01 }
  ]);
  for (const bad of ["", null, "ear:0.5>0.9", "eye_r:abc>1", "eye_r:0.5>99", "<script>"]) {
    assert.deepEqual(parseFineDetail(bad), []);
  }
});

test("describes the finest-feature finishing step", () => {
  assert.equal(
    describeFineFeatures("applied", "eye_r:0.610>0.970,eye_l:0.640>0.950,lips:1.020>1.010", "1/2"),
    "İnce bölgeler iyileştirildi (fotoğrafa göre ayrıntı: sağ göz 0,61→0,97, sol göz 0,64→0,95) · göz ışığı 1/2 gözde geri getirildi"
  );
  assert.equal(describeFineFeatures("applied", "", "0/2"), "İnce bölgeler iyileştirildi");
  assert.equal(
    describeFineFeatures("applied", "", "0/2", "2/2"),
    "İnce bölgeler iyileştirildi · göz akı tonu 2/2 gözde fotoğrafın ışığına uyarlandı"
  );
  assert.equal(describeFineFeatures("identity-loss"), "ince bölge iyileştirmesi kimliği korumadığı için uygulanmadı");
  assert.equal(describeFineFeatures("unchanged"), "göz, kaş ve dudak ayrıntısı fotoğrafla uyumlu");
  assert.equal(describeFineFeatures("no-mesh"), "");
  assert.equal(describeFineFeatures(""), "");
});

test("the jury badge says when close-ups were used", async () => {
  const {describeNvidiaJudge} = await import("../src/quality.js");
  assert.equal(
    describeNvidiaJudge("switched", 8, 0.75, "review-sheet"),
    "NVIDIA jürisi daha doğal görünen finalisti seçti (8 oy, %75 uzlaşı, göz ve ağız yakın planlarıyla)"
  );
  assert.equal(describeNvidiaJudge("kept", null, null, "review-sheet"), "NVIDIA jürisi yerel seçimi onayladı (göz ve ağız yakın planlarıyla)");
  assert.equal(describeNvidiaJudge("abstain", 6, 0.5, "review-sheet"), "NVIDIA jürisi kararsız kaldı; yerel seçim korundu");
  assert.equal(describeNvidiaJudge("kept", 6, 1, "face"), "NVIDIA jürisi yerel seçimi onayladı (6 oy, %100 uzlaşı)");
});


test("lower-face measurements are distinguished from identity and malformed evidence is omitted", async () => {
  const {describePortraitEvidence,describeFluxMetrics}=await import("../src/quality.js");
  assert.match(describePortraitEvidence("chin:0.00>3.10/0.000>0.120","0.8"), /çene 0,00→3,10/);
  assert.match(describePortraitEvidence("chin:0.00>3.10/0.000>0.120","0.8"), /doku\/kontrast/);
  assert.equal(describePortraitEvidence("chin:0.00>3.10/0.000>2.0",""),"");
  assert.equal(describePortraitEvidence("",""),"");
  assert.match(describeFluxMetrics(-.005,.009,.025), /karışım %2,5/);
});


test("portrait geometry requires a measured mesh and respects pixel aspect ratio", async () => {
  const {portraitGeometry,describePortraitGeometry}=await import("../src/quality.js");
  const points=Array.from({length:478},()=>({x:.5,y:.5}));
  const positions={234:[.25,.5],454:[.75,.5],10:[.5,.2],152:[.5,.8],33:[.3,.4],133:[.4,.4],362:[.6,.4],263:[.7,.4],159:[.35,.39],145:[.35,.41],386:[.65,.39],374:[.65,.41],61:[.4,.65],291:[.6,.65],13:[.5,.64],14:[.5,.66],98:[.45,.55],327:[.55,.55],172:[.3,.7],397:[.7,.7]};
  for(const [i,[x,y]] of Object.entries(positions))points[i]={x,y};
  const source=portraitGeometry(points,400,200);
  assert.equal(source.width,200);
  assert.equal(source.height,120);
  assert.ok(Math.abs(source.values.rightEye-.1)<1e-9);
  const edited=points.map(p=>({...p}));edited[159].y-=.01;
  const delivered=portraitGeometry(edited,400,200);
  assert.ok(delivered.values.rightEye>source.values.rightEye);
  assert.match(describePortraitGeometry({source,delivered}),/sağ göz açıklığı 0,10→0,15/);
  assert.match(describePortraitGeometry({source,delivered:null}),/ölçülemedi/);
  assert.equal(portraitGeometry(points.slice(0,100),400,200),null);
  assert.equal(portraitGeometry(points,20,20),null);
});


test("native input evidence does not mistake a larger smooth image for texture", async () => {
  const {sourcePortraitQuality,describeSourcePortraitQuality}=await import("../src/quality.js");
  const size=64, flat=new Uint8ClampedArray(size*size*4).fill(120);
  const bounds={x:0,y:0,width:64,height:64};
  const a=sourcePortraitQuality(flat,size,size,bounds);
  assert.ok(a.edgeEnergy<1e-9);
  assert.equal(a.clippedFraction,0);
  assert.match(describeSourcePortraitQuality(a),/gerçek giriş ayrıntısı sınırlı/);
  const striped=flat.slice();
  for(let y=0;y<size;y++)for(let x=0;x<size;x++)if(x%2===0)striped.fill(80,(y*size+x)*4,(y*size+x)*4+3);
  assert.ok(sourcePortraitQuality(striped,size,size,bounds).edgeEnergy>a.edgeEnergy);
  assert.equal(sourcePortraitQuality(flat,size,size,{...bounds,width:3}),null);
});
