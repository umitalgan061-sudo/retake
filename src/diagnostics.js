// Diagnostics report (v26): one text the person can paste into a chat so the
// engine can be tuned for their PC. It holds measurements only — never an
// image, a file name, a folder path, a token or a pairing code.

const HEALTH_KEYS = Object.freeze([
  "engine", "version", "model", "ultra_rounds", "ultra_min_seconds", "quality_depths",
  "ultra_default_depth", "fine_features", "identity_scorer", "pass_cache", "nvidia_judge",
  "nim_editors_configured", "pixel_boost", "face_enhancer_enabled", "face_enhancer_model",
  "realism", "personal_model", "cuda", "ultra_learning", "self_test_benchmark",
  "neck_harmonization", "last_job_cleanup", "quality_revision", "regional_edit_evidence",
  "lower_face_texture_evidence", "native_input_recovery", "reference_appearance_tiebreak",
  "finishing_identity_guard", "flux_texture_alternatives", "distinct_extra_search", "jury_local_quality_guard", "visible_identity_verification"
]);
const SECRET_NAME = /token|secret|key|pairing|auth|cookie|password/i;
const MAX_VALUE = 300;
export const EXPECTED_QUALITY_REVISION = "portrait-quality-20261126-r8";
export const WEB_QUALITY_REVISION = EXPECTED_QUALITY_REVISION;

// Windows/POSIX paths and file names never leave the device.
export function scrub(value) {
  return String(value)
    .replace(/[A-Za-z]:[\\/][^\s,;"']*/g, "[yol]")
    .replace(/(?:^|(?<=[\s,;"'=(]))\/(?:[\w.-]+\/)+[\w.-]*/g, "[yol]")
    .replace(/[\w.-]+\.(?:png|jpe?g|webp|heic|bmp|tiff?|json|onnx|safetensors)\b/gi, "[dosya]")
    .slice(0, MAX_VALUE);
}

export function collectDiagnosticHeaders(headers) {
  const collected = {};
  if (!headers) return collected;
  const add = (name, value) => {
    const key = String(name).toLowerCase();
    if (!key.startsWith("x-retake-") || SECRET_NAME.test(key)) return;
    collected[key] = scrub(value);
  };
  if (typeof headers.forEach === "function") headers.forEach((value, name) => add(name, value));
  else for (const [name, value] of Object.entries(headers)) add(name, value);
  return Object.fromEntries(Object.entries(collected).sort(([a], [b]) => a.localeCompare(b)));
}

export function parseUltraTiming(value) {
  const result = {};
  if (typeof value !== "string") return result;
  for (const part of value.split(";")) {
    const match = /^([a-z]+)=(\d{1,6})(?:x(\d{1,6}))?$/.exec(part.trim());
    if (!match) continue;
    result[match[1]] = match[3] === undefined
      ? { seconds: Number(match[2]) }
      : { count: Number(match[2]), seconds: Number(match[3]) };
  }
  return result;
}

function minutes(seconds) {
  return seconds >= 90 ? `${Math.round(seconds / 60)} dk` : `${seconds} sn`;
}

export function describeUltraTiming(value) {
  const timing = parseUltraTiming(value);
  const names = { search: "tur", refine: "ince ayar", realism: "gerçekçilik", extra: "ek finalist" };
  const parts = [];
  for (const [key, label] of Object.entries(names)) {
    const item = timing[key];
    if (item?.count > 0) parts.push(`${item.count} ${label} × ${minutes(item.seconds)}`);
  }
  if (timing.judge?.seconds > 0) parts.push(`jüri ${minutes(timing.judge.seconds)}`);
  if (timing.total?.seconds > 0) parts.push(`toplam ${minutes(timing.total.seconds)}`);
  return parts.length ? `Süre: ${parts.join(" · ")}` : "";
}

function safeJson(value) {
  try {
    return JSON.stringify(value);
  } catch {
    return "";
  }
}

function healthSummary(health) {
  const summary = {};
  if (!health || typeof health !== "object") return summary;
  for (const key of HEALTH_KEYS) {
    if (!Object.hasOwn(health, key)) continue;
    const value = health[key];
    summary[key] = value !== null && typeof value === "object" ? scrub(safeJson(value)) : scrub(value);
  }
  return summary;
}

export function buildDiagnosticReport({ headers = {}, health = null, depth = "", appVersion = "", createdAt = new Date() } = {}) {
  const lines = [
    "Retake tanı raporu (görsel, dosya adı veya klasör yolu içermez)",
    `Zaman: ${createdAt.toISOString()}`,
  ];
  if (appVersion) lines.push(`Site: ${scrub(appVersion)}`);
  if (depth) lines.push(`Seçilen derinlik: ${scrub(depth)}`);
  const timing = describeUltraTiming(headers["x-retake-ultra-timing"]);
  if (timing) lines.push(timing);
  const evidence = summarizeQualityEvidence(headers);
  if (evidence) lines.push(evidence);
  const resultRevision = validQualityRevision(headers["x-retake-quality-revision"]);
  const currentRevision = validQualityRevision(health?.quality_revision);
  lines.push(`Sonucu üreten kalite sürümü: ${resultRevision || "doğrulanamadı (eski motor veya sonuç)"}`);
  if (currentRevision) lines.push(`Şu an bağlı motorun kalite sürümü: ${currentRevision}`);
  if (resultRevision && currentRevision && resultRevision !== currentRevision) lines.push("Not: Sonuç farklı bir motor sürümünde üretildi; güncel motor durumu bu sonucun kanıtı değildir.");
  lines.push("", "Sonuç ölçümleri:");
  for (const [name, value] of Object.entries(headers)) {
    if (SECRET_NAME.test(name)) continue;
    lines.push(`${scrub(name)}: ${scrub(value)}`);
  }
  const summary = healthSummary(health);
  if (Object.keys(summary).length) {
    lines.push("", "Motor durumu:");
    for (const [key, value] of Object.entries(summary)) lines.push(`${key}: ${value}`);
  }
  return lines.join("\n");
}


export function validQualityRevision(value) {
  return typeof value === "string" && /^portrait-quality-\d{8}-r\d{1,4}$/.test(value) ? value : null;
}

export function describeQualityCompatibility(value) {
  const revision=validQualityRevision(value);
  if (revision === EXPECTED_QUALITY_REVISION) return "";
  if (!revision) return "Motorun yeni kalite paketi doğrulanamadı; yerel motoru güncelle ve yeniden başlat.";
  const parse=v=>v.match(/(\d{8})-r(\d+)$/).slice(1).map(Number);
  const [date,number]=parse(revision), [expectedDate,expectedNumber]=parse(EXPECTED_QUALITY_REVISION);
  if (date>expectedDate || (date===expectedDate && number>=expectedNumber)) return "";
  return "Yerel motor eski kalite paketini kullanıyor ("+revision+"); güncelle ve yeniden başlat.";
}


export function summarizeQualityEvidence(headers) {
  const read=(key,min,max)=>{
    const raw=headers[key];
    if(raw===undefined||raw===null||String(raw).trim()==="")return null;
    const n=Number(raw);
    return Number.isFinite(n)&&n>=min&&n<=max?n:null;
  };
  const final=read("x-retake-similarity",-1,1);
  const peak=read("x-retake-ultra-peak-identity",-1,1);
  const goal=read("x-retake-identity-aspiration",-1,1);
  const points=value=>(100*value).toFixed(2).replace(".",",");
  const rows=[];
  if(final!==null&&peak!==null&&peak>=final-.0001)
    rows.push("arama zirvesinden fark "+points(Math.max(0,peak-final))+" puan");
  if(final!==null&&goal!==null)
    rows.push("arama hedefinden fark "+points(final-goal)+" puan");
  if(headers["x-retake-detail-gate"]==="passed")rows.push("motor ayrıntı kontrolü geçti");
  if(headers["x-retake-identity-pixel-scope"]==="visible-engine-crop")rows.push("kimlik doğrulaması görünür motor kırpımında yapıldı (tarayıcı ölçeklemesi öncesi)");
  if(headers["x-retake-nim-editor"]==="retake-local/flux2-klein-4b"&&headers["x-retake-flux-status"]==="accepted")
    rows.push("seçilen adayda FLUX kullanıldı");
  return rows.length?"Kalite özeti (kosinüs ölçeği, benzerlik yüzdesi değil): "+rows.join(" · "):"";
}


// Explicit, single local comparison baseline. No image, filename, embedding,
// token or fingerprint is copied to diagnostics or sent to the engine.
export const QUALITY_BASELINE_KEY = "retake-quality-baseline-v1";
const HEX=/^[a-f0-9]{64}$/;
const COMPARISON_FIELDS = Object.freeze({
  similarity:["x-retake-similarity",-1,1,100,"kimlik kosinüs puanı"],
  peak:["x-retake-ultra-peak-identity",-1,1,100,"arama zirvesi"],
  detail:["x-retake-detail-score",0,1,100,"ayrıntı uyumu"],
  fine:["x-retake-fine-quality",0,1,100,"ince ayrıntı uyumu"],
  appearance:["x-retake-reference-appearance",0,1,100,"alt yüz doku uyumu"],
  rim:["x-retake-rim-fraction",0,1,100,"kontur farkı"],
});
const toHex=bytes=>Array.from(new Uint8Array(bytes),v=>v.toString(16).padStart(2,"0")).join("");
async function hashBuffer(buffer) { return toHex(await globalThis.crypto.subtle.digest("SHA-256",buffer)); }

export async function comparisonContext({target,referenceKey,faceIndex,depth,judge,full,benchmark,prepared,metadata}) {
  if(!globalThis.crypto?.subtle || !target?.arrayBuffer || !HEX.test(referenceKey||"") ||
     !Number.isInteger(faceIndex)||faceIndex<0 || benchmark) return null;
  try {
    const targetHash=await hashBuffer(await target.arrayBuffer());
    const input=await hashBuffer(new TextEncoder().encode(JSON.stringify([targetHash,referenceKey,faceIndex,depth,Boolean(judge),Boolean(full)])));
    const digests=[];
    for(const blob of prepared||[]) {
      if(!blob?.arrayBuffer) return null;
      digests.push(await hashBuffer(await blob.arrayBuffer()));
    }
    const preparedHash=await hashBuffer(new TextEncoder().encode(JSON.stringify([digests,metadata])));
    return {input,prepared:preparedHash};
  } catch { return null; }
}

function measuredNumber(value,min,max) {
  if(value===null||value===undefined||String(value).trim()==="")return null;
  const n=Number(value);
  return Number.isFinite(n)&&n>=min&&n<=max?n:null;
}

export function createQualityBaseline(context,headers,now=Date.now()) {
  if(!HEX.test(context?.input||"")||!HEX.test(context?.prepared||"")||headers?.["x-retake-detail-gate"]!=="passed")return null;
  const metrics={};
  for(const [name,[key,min,max]] of Object.entries(COMPARISON_FIELDS)) {
    const value=measuredNumber(headers[key],min,max);
    if(value!==null)metrics[name]=value;
  }
  if(metrics.similarity===undefined)return null;
  const seconds=parseUltraTiming(headers["x-retake-ultra-timing"]).total?.seconds;
  if(Number.isFinite(seconds)&&seconds>=0&&seconds<=86400)metrics.seconds=seconds;
  return {version:1,input:context.input,prepared:context.prepared,
    revision:validQualityRevision(headers["x-retake-quality-revision"]),metrics,
    createdAt:Number.isFinite(now)?now:Date.now()};
}

export function readQualityBaseline(storage) {
  try {
    const raw=storage?.getItem(QUALITY_BASELINE_KEY);
    if(!raw||raw.length>4096)return null;
    const record=JSON.parse(raw);
    if(record?.version!==1||!HEX.test(record.input||"")||!HEX.test(record.prepared||"")||!Number.isFinite(record.createdAt))return null;
    const metrics={};
    for(const [name,[,min,max]] of Object.entries(COMPARISON_FIELDS)){
      const n=record.metrics?.[name];
      if(typeof n==="number"&&measuredNumber(n,min,max)!==null)metrics[name]=n;
    }
    if(metrics.similarity===undefined)return null;
    const seconds=record.metrics?.seconds;
    if(typeof seconds==="number"&&measuredNumber(seconds,0,86400)!==null)metrics.seconds=seconds;
    return {version:1,input:record.input,prepared:record.prepared,revision:validQualityRevision(record.revision),
      metrics,createdAt:record.createdAt};
  } catch { return null; }
}

export function compareQualityBaseline(baseline,current) {
  if(!baseline||!current)return "";
  if(baseline.input!==current.input)return "Kıyas yapılamadı: giriş, referans seti, seçilen yüz veya Ultra ayarı farklı.";
  const lines=[];
  if(baseline.prepared!==current.prepared)lines.push("Hazırlanan motor girdileri farklı; değişiklik tüm işlem hattına ait olabilir.");
  if(baseline.revision!==current.revision)lines.push("Kalite sürümü değişti; ölçümler aynı ölçekteyse yorumlanmalı.");
  if(!baseline.revision||!current.revision)lines.push("En az bir kalite sürümü doğrulanamadı.");
  for(const [name,[,,,scale,label]] of Object.entries(COMPARISON_FIELDS)) {
    const a=baseline.metrics[name],b=current.metrics[name];
    if(!Number.isFinite(a)||!Number.isFinite(b))continue;
    const delta=(b-a)*scale;
    lines.push(label+" "+(a*scale).toFixed(2).replace(".",",")+" → "+(b*scale).toFixed(2).replace(".",",")+
      " ("+(delta>0?"+":"")+delta.toFixed(2).replace(".",",")+")");
  }
  if(Number.isFinite(baseline.metrics.seconds)&&Number.isFinite(current.metrics.seconds))
    lines.push("süre "+baseline.metrics.seconds+" → "+current.metrics.seconds+" sn");
  return "Yerel ölçüm kıyası (görsel kalite veya kimlik garantisi değil): "+lines.join(" · ");
}
