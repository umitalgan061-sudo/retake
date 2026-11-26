import { FaceDetector, FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/vision_bundle.mjs";
import {
  formatBytes,
  findExactReferenceFile,
  getLandmarkBounds,
  settleWithLimit,
  transparentFraction,
  TRANSPARENT_TARGET_FRACTION,
  MAX_IMAGE_BYTES,
  MAX_REFERENCE_FACES,
  validateImageDimensions,
  validateImageFile,
  validateReferenceFiles
} from "./src/core.js";
import {
  getReferenceImageUrl,
  parseReferenceManifest,
  REFERENCE_MANIFEST_URL
} from "./src/reference-catalog.js";
import {
  getFaceCropRect,
  createFaceAlphaMask
} from "./src/nvidia-pipeline.js";
import { alphaMaskFromRGBA, binaryExtentFromRGBA, compositeMaskedRGBA, compositePreblendedRGBA, inspectOutsideMaskIntegrity } from "./src/face-only.js";
import { calculateReferenceWeight, getSimilarityTransform, weightedAverageRGBA } from "./src/identity-profile.js";
import {
  canonicalJaw,
  estimateHeadFit,
  jawPointsFromMesh,
  meshFrontalness,
  planJawWarp,
  summarizeReferenceJaw,
  getTargetRenderSize,
  mapLandmarksFromRegion,
  measureFaceShapeRatio,
  rectIoU,
  orderDetectedFaces,
  rectOverlap,
  summarizeReferenceShape
} from "./src/head-geometry.js";
import { mountSculpture, placeWideSculpture } from "./src/forma-sculpture.js";
import { buildBenchmarkReport, pickBenchmarkReferences, runBenchmark, summarizeBenchmark, fidelityScore } from "./src/benchmark.js";
import { descriptorDistance, meshToPixels, poseDescriptor, rankRealReferences, referenceCrop } from "./src/real-face.js";
import {
  DEFAULT_QUALITY_DEPTH, QUALITY_DEPTHS, QUALITY_DEPTH_STORAGE_KEY,
  depthLabel, depthMinutes, depthTimeoutMs, describeDepthDuration, normalizeDepth, requestedDepth
} from "./src/depth.js";
import { buildDiagnosticReport, collectDiagnosticHeaders, describeUltraTiming, describeQualityCompatibility, WEB_QUALITY_REVISION, validQualityRevision, comparisonContext, createQualityBaseline, readQualityBaseline, compareQualityBaseline, QUALITY_BASELINE_KEY } from "./src/diagnostics.js";
import { MIN_UPSCALE, describeNativeFinish, finishUpscaledFace } from "./src/grain.js";
import {
  bestResultIndex,
  measureFaceEdit,
  describeProfileHealth,
  describeFluxStatus,
  describeAdaptiveQuality,
  describeFluxMetrics,
  describeJudgeError,
  describeNvidiaJudge,
  describeIdentitySufficiency,
  describeSelfFidelity,
  describeUltraSchedule,
  describeUltraDepth,
  describeFineFeatures,
  describeNeckEvidence,
  describePortraitEvidence,
  portraitGeometry,
  describePortraitGeometry,
  sourcePortraitQuality,
  describeSourcePortraitQuality,
  describeSimilarity,
  describeNaturalness,
  parseNaturalness,
  parseSimilarity,
  similarityScore
} from "./src/quality.js";
import {
  fetchCachedReference,
  fetchReferenceManifest,
  pruneReferenceCache,
  requestPersistentStorage
} from "./src/reference-cache.js";
import {
  DEFAULT_ENGINE_ORIGIN,
  engineHeaders,
  formatElapsed,
  formatRemaining,
  loadEngineSettings,
  resolveEngineSettings,
  saveEngineSettings
} from "./src/engine.js";
import {
  createReferenceSetKey,
  deleteCachedProfile,
  getCachedProfile,
  saveCachedProfile
} from "./src/profile-storage.js";

const MAX_PREVIEW_EDGE = 2048;
const MAX_REFERENCE_EDGE = 2048;
const FACE_VARIATION_LABELS = [
  "Otomatik oturtma",
  "Fotoğraftaki kafa oranı",
  "Güçlü oturtma",
  "Hafif oturtma"
];
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const FACE_DETECTOR_MODEL_URLS = {
  short: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
  full: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite"
};

const $ = (selector) => document.querySelector(selector);
const elements = {
  targetInput: $("#targetInput"),
  targetDropzone: $("#targetDropzone"),
  targetEmpty: $("#targetEmpty"),
  targetPreview: $("#targetPreview"),
  targetCanvas: $("#targetCanvas"),
  targetMeta: $("#targetMeta"),
  clearTarget: $("#clearTargetButton"),
  faceSelection: $("#faceSelection"),
  faceCount: $("#faceCount"),
  faceChips: $("#faceChips"),
  referencesInput: $("#referencesInput"),
  chooseReferences: $("#chooseReferencesButton"),
  reloadReferences: $("#reloadReferencesButton"),
  clearReferences: $("#clearReferencesButton"),
  referenceProgress: $("#referenceProgress"),
  swapButton: $("#swapButton"),
  realFaceToggle: $("#realFaceToggle"),
  nvidiaJudgeRow: $("#nvidiaJudgeRow"),
  qualityDepthRow: $("#qualityDepthRow"),
  qualityDepthNote: $("#qualityDepthNote"),
  benchmarkCount: $("#benchmarkCount"),
  benchmarkStart: $("#benchmarkStart"),
  benchmarkStop: $("#benchmarkStop"),
  benchmarkStatus: $("#benchmarkStatus"),
  benchmarkResults: $("#benchmarkResults"),
  benchmarkRows: $("#benchmarkRows"),
  benchmarkSummary: $("#benchmarkSummary"),
  benchmarkDownload: $("#benchmarkDownload"),
  benchmarkSummaryNote: $("#benchmarkSummaryNote"),
  nvidiaJudgeToggle: $("#nvidiaJudgeToggle"),
  downloadButton: $("#downloadButton"),
  diagnosticsButton: $("#diagnosticsButton"),
  baselineSave: $("#baselineSave"),
  baselineClear: $("#baselineClear"),
  baselineStatus: $("#baselineStatus"),
  resultImage: $("#resultImage"),
  resultView: $("#resultView"),
  originalImage: $("#originalImage"),
  resultTools: $("#resultTools"),
  qualityBadge: $("#qualityBadge"),
  profileHealth: $("#profileHealth"),
  compareRange: $("#compareRange"),
  strengthRange: $("#strengthRange"),
  strengthValue: $("#strengthValue"),
  historyStrip: $("#historyStrip"),
  historyCount: $("#historyCount"),
  exportFormat: $("#exportFormat"),
  shareButton: $("#shareButton"),
  waitProgress: $("#waitProgress"),
  waitStep: $("#waitStep"),
  engineDetails: $("#engineDetails"),
  engineSummary: $("#engineSummary"),
  engineForm: $("#engineForm"),
  engineAddress: $("#engineAddress"),
  engineToken: $("#engineToken"),
  engineReset: $("#engineResetButton"),
  engineFormStatus: $("#engineFormStatus"),
  outputPlaceholder: $("#outputPlaceholder"),
  outputMessage: $("#outputMessage"),
  outputStage: $(".output-stage"),
  waitStage: $("#waitStage"),
  waitTitle: $("#waitTitle"),
  waitTimer: $("#waitTimer"),
  outputSymbol: $(".output-symbol"),
  outputTitle: $("#outputTitle"),
  outputStatus: $("#outputStatus"),
  engineTitle: $("#engineTitle"),
  engineDescription: $("#engineDescription"),
  disabledNote: $("#disabledNote"),
  referencesEmpty: $("#referencesEmpty"),
  referencesGrid: $("#referencesGrid"),
  referenceCount: $("#referenceCount"),
  runtimeStatus: $("#runtimeStatus"),
  engineTitle: $("#engineTitle"),
  engineDescription: $("#engineDescription"),
  themeButton: $("#themeButton"),
  installButton: $("#installButton"),
  toast: $("#toast"),
  appDialog: $("#appDialog"),
  dialogTitle: $("#dialogTitle"),
  dialogMessage: $("#dialogMessage"),
  dialogCancel: $("#dialogCancel"),
  dialogConfirm: $("#dialogConfirm")
};

const state = {
  detector: null,
  vision: null,
  referenceDetector: null,
  referenceDetectorPromise: null,
  referenceDetectorFull: null,
  referenceDetectorFullPromise: null,
  target: null,
  references: [],
  identityProfile: null,
  referenceSetKey: null,
  profileIsBuilding: false,
  referencesLoading: false,
  referenceLoadProgress: null,
  profilePreparationProgress: null,
  referenceFaceCropCount: null,
  selectedReferenceCount: 0,
  referenceLoadError: null,
  referenceLoadToken: 0,
  targetLoadToken: 0,
  targetLoading: false,
  selectedFaceIndex: null,
  faceFusionStatus: { checked: false, configured: false },
  isSwapping: false,
  nextVariation: 0,
  result: null,
  results: [],
  engine: { origin: DEFAULT_ENGINE_ORIGIN, remote: false, token: "" },
  toastTimer: null,
  dialogResolve: null,
  installPrompt: null
};

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  window.clearTimeout(state.toastTimer);
  state.toastTimer = window.setTimeout(() => elements.toast.classList.remove("show"), 3200);
}

function showAppDialog({ title, message, confirmLabel, cancelLabel = "Vazgeç", danger = false }) {
  elements.dialogTitle.textContent = title;
  elements.dialogMessage.textContent = message;
  elements.dialogConfirm.textContent = confirmLabel;
  elements.dialogCancel.textContent = cancelLabel;
  elements.dialogConfirm.classList.toggle("danger-action", danger);
  elements.appDialog.returnValue = "";
  return new Promise((resolve) => {
    state.dialogResolve = resolve;
    elements.appDialog.showModal();
  });
}

function setRuntimeStatus(message) {
  elements.runtimeStatus.textContent = message;
}

function updateProfileIndicator({ title, detail, progress, status = "loading" } = {}) {
  const titleElement = $("#profileStatusText");
  const detailElement = $("#profileStatusDetail");
  const progressElement = $("#profileProgressBar");
  const dotElement = $("#profileStateDot");
  if (titleElement && title) titleElement.textContent = title;
  if (detailElement && detail) detailElement.textContent = detail;
  if (progressElement && Number.isFinite(progress)) {
    const roundedProgress = Math.round(Math.max(0, Math.min(100, progress)) / 10) * 10;
    for (let value = 0; value <= 100; value += 10) progressElement.classList.toggle("progress-" + value, value === roundedProgress);
  }
  if (dotElement) {
    dotElement.classList.toggle("is-ready", status === "ready");
    dotElement.classList.toggle("is-error", status === "error");
  }
}

function localFaceFusionBase() {
  return state.engine.origin;
}

function engineFetch(path, options = {}) {
  return fetch(state.engine.origin + path, {
    cache: "no-store",
    ...options,
    headers: engineHeaders(state.engine, options.headers || {})
  });
}

function updateSwapAvailability() {
  const referenceCount = state.references.filter((item) => item.status === "ready").length;
  const hasTarget = Boolean(state.target && Number.isInteger(state.selectedFaceIndex) &&
    state.target.faces[state.selectedFaceIndex]);
  const enabled = state.faceFusionStatus.configured && state.detector &&
    referenceCount === MAX_REFERENCE_FACES && hasTarget &&
    !state.targetLoading && !state.referencesLoading && !state.isSwapping && !state.benchmark?.running;
  elements.swapButton.disabled = !enabled;
  updateBenchmarkAvailability(referenceCount);
  elements.downloadButton.disabled = !state.result || state.isSwapping;
  elements.shareButton.disabled = !state.result || state.isSwapping;
  if (elements.diagnosticsButton) {
    const hasReport = Boolean(state.result?.diagnostics && Object.keys(state.result.diagnostics).length);
    elements.diagnosticsButton.hidden = !hasReport;
    elements.diagnosticsButton.disabled = !hasReport || state.isSwapping;
  }
  updateQualityBaseline();
  const copy = elements.swapButton.querySelector(".button-copy");
  if (copy) copy.textContent = state.result ? "Yeniden oluştur" : "Yüzümü uygula";
  const tag = elements.swapButton.querySelector(".button-state");
  if (tag) tag.textContent = state.isSwapping ? "İşleniyor" :
    (enabled ? "HAZIR" : !state.faceFusionStatus.configured ? "MOTOR BEKLENİYOR" :
      referenceCount !== MAX_REFERENCE_FACES ? "PROFİL GEREKLİ" : "FOTOĞRAF EKLE");
  if (!state.isSwapping) {
    elements.disabledNote.textContent = state.referencesLoading
      ? "Yüz referansları hazırlanıyor…"
      : referenceCount !== MAX_REFERENCE_FACES
        ? "Varsayılan profil yüklenemedi. Yüz profili ayarlarından 25 fotoğraf seçebilir veya havuzu yeniden yükleyebilirsin."
        : !state.faceFusionStatus.configured
          ? (isLikelyMobile()
            ? "Telefondan kullanmak için Motor bağlantısı bölümünden bilgisayarını eşleştir."
            : "Düzenleme için bilgisayarında FaceFusion motorunu başlat.")
          : !hasTarget
            ? "Başlamak için bir fotoğraf yükle ve değiştirilecek yüzü seç."
            : "Yeniden oluştur, farklı bir kafa oranı ve sakal dengesi dener.";
  }
  qualityDepthInputs().forEach((input) => { input.disabled = state.isSwapping || Boolean(state.benchmark?.running); });
  elements.chooseReferences.disabled = state.referencesLoading || state.isSwapping;
  elements.reloadReferences.disabled = state.referencesLoading || state.isSwapping;
  elements.clearReferences.disabled = state.referencesLoading || state.isSwapping;
  elements.targetInput.disabled = state.targetLoading || state.isSwapping;
  elements.clearTarget.disabled = state.targetLoading || state.isSwapping;
  elements.targetDropzone.setAttribute("aria-disabled", String(state.targetLoading || state.isSwapping));
}

async function checkFaceFusionStatus() {
  const base = localFaceFusionBase();
  if (!base) {
    state.faceFusionStatus = { checked: true, configured: false, message: "Yerel FaceFusion adresi geçersiz." };
  } else {
    try {
      const response = await engineFetch("/health", { headers: { Accept: "application/json" } });
      const health = await response.json().catch(() => ({}));
      if (response.status === 401) {
        throw new Error("Eşleştirme kodu gerekli veya hatalı. Motor bağlantısı bölümünden kodu gir.");
      }
      if (!response.ok || !health.ok) throw new Error(health.error || "Yerel FaceFusion servisi hazır değil.");
      if (!health.cuda) throw new Error("FaceFusion çalışıyor ancak CUDAExecutionProvider bulunamadı. CUDA kurulumunu kontrol et.");
      if (!health.ultra_mode || health.ultra_min_seconds < 420 ||
          health.model !== "retake_hybrid_25_reference_ultra_v22" ||
          health.lossless_preblended_transport !== true) {
        throw new Error(
          "Kayıpsız kontur düzeltmeli Retake motoru gerekli: GitHub’dan güncelleyip FaceFusion servisini yeniden başlat."
        );
      }
      state.faceFusionStatus = {
        checked: true,
        configured: true,
        model: health.model || "uniface_256",
        qualityRevision: validQualityRevision(health.quality_revision),
        pixelBoost: health.pixel_boost || "768x768",
        realism: Boolean(health.realism),
        nimEditors: Array.isArray(health.nim_editors_configured) ? health.nim_editors_configured : [],
        // Only an engine with a Hafize project token or a local NVIDIA key may offer the jury.
        nvidiaJudge: health.nvidia_judge === "configured",
        nvidiaJudgeDirect: health.nvidia_judge_transport === "direct",
        // Strict leave-one-out self-tests and learned variant order (v22.3+).
        selfTestBenchmark: health.self_test_benchmark === true,
        ultraLearning: health.ultra_learning === true,
        // Selectable quality depth and job-sized pass cache (v23+).
        qualityDepths: health.quality_depths === true,
        ultraDepths: health.ultra_depths && typeof health.ultra_depths === "object" ? health.ultra_depths : null,
        finalFaceEditProof: health.final_face_edit_proof === true,
        personalModel: Boolean(health.personal_model),
        message: describeQualityCompatibility(health.quality_revision) || (health.final_face_edit_proof !== true
          ? 'Eski motor: son yüz değişimi denetimi için yerel Retake motorunu güncelle. Tarayıcı son görüntüyü yine kontrol eder.'
          : health.personal_model
          ? "Ultra-420 v22 · tam referans yüzü, sakal/çene ve kişisel model adayları hazır."
          : health.realism
            ? "Ultra-420 v22 · tam yüz ve sakal/çene adayları, finalde gerçekçilik açık."
            : "Ultra-420 v22 · 24 aday, tam yüz ve sakal/çene aktarımı. Gerçekçilik: tools\\install_realism.ps1")
      };
    } catch (error) {
      const detail = error?.message || "Yerel FaceFusion bağlantısı kurulamadı.";
      state.faceFusionStatus = {
        checked: true,
        configured: false,
        message: (detail.includes("Failed to fetch") || detail.includes("NetworkError") || detail.includes("Load failed")) && state.engine.remote
          ? "Eşleştirilen motora ulaşılamadı. Bilgisayardaki motorun ve tünelin açık olduğunu, adresin güncel olduğunu kontrol et."
          : (detail.includes("Failed to fetch") || detail.includes("NetworkError") || detail.includes("Load failed")) && isLikelyMobile()
          ? "Bu cihazda motor yok. Telefondan kullanmak için aşağıdaki Motor bağlantısı bölümünden bilgisayarındaki motoru eşleştir."
          : detail.includes("Failed to fetch") || detail.includes("NetworkError")
          ? "127.0.0.1:8765 bağlantısı kurulamadı. tools\\start_facefusion_retake.ps1 dosyasını çalıştır; tarayıcı yerel ağ izni sorarsa izin ver."
          : detail
      };
    }
  }

  const connectionLabel = $("#topConnectionText");
  if (connectionLabel) {
    connectionLabel.textContent = state.faceFusionStatus.configured ? "Motor hazır" : "Motor çevrimdışı";
    connectionLabel.closest(".connection-chip")?.classList.toggle("is-ready", Boolean(state.faceFusionStatus.configured));
  }
  elements.engineSummary.textContent = state.engine.remote
    ? (state.faceFusionStatus.configured ? "Eşleşti" : "Bağlanamadı")
    : "Bu bilgisayar";
  if (elements.nvidiaJudgeRow) {
    const judgeReady = Boolean(state.faceFusionStatus.configured && state.faceFusionStatus.nvidiaJudge);
    elements.nvidiaJudgeRow.hidden = !judgeReady;
    const judgeTitle = elements.nvidiaJudgeRow.querySelector("strong");
    if (judgeTitle) {
      judgeTitle.textContent = "NVIDIA görsel jürisi (" +
        (state.faceFusionStatus.nvidiaJudgeDirect ? "doğrudan NVIDIA" : "Hafize üzerinden") + ")";
    }
    if (elements.nvidiaJudgeToggle) elements.nvidiaJudgeToggle.disabled = !judgeReady;
  }
  updateQualityDepthAvailability();
  if (state.faceFusionStatus.configured) {
    elements.engineTitle.textContent = "Yerel düzenleme hazır";
    elements.engineDescription.textContent = state.engine.remote
      ? "Yüz dönüşümü, şifreli tünelle eşleştirdiğin bilgisayarındaki FaceFusion motorunda çalışır."
      : "Yüz dönüşümü bu cihazdaki FaceFusion motorunda çalışır.";
  } else {
    elements.engineTitle.textContent = "Yerel düzenleme motoru çevrimdışı";
    elements.engineDescription.textContent = state.faceFusionStatus.message;
  }
  updateSwapAvailability();
}
function createObjectURL(blob) {
  return URL.createObjectURL(blob);
}

function loadImage(fileOrBlob) {
  return new Promise((resolve, reject) => {
    const url = createObjectURL(fileOrBlob);
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve({ image, url });
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Görsel açılamadı. Başka bir JPEG, PNG veya WebP dosyası deneyin."));
    };
    image.src = url;
  });
}

async function loadReferenceImage(blob) {
  const loaded = await loadImage(blob);
  const width = loaded.image.naturalWidth || loaded.image.width;
  const height = loaded.image.naturalHeight || loaded.image.height;
  const validation = validateImageDimensions(width, height);
  if (!validation.valid) {
    URL.revokeObjectURL(loaded.url);
    throw new Error(validation.reason);
  }
  if (Math.max(width, height) <= MAX_REFERENCE_EDGE) return loaded;

  const scale = MAX_REFERENCE_EDGE / Math.max(width, height);
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  const context = canvas.getContext("2d");
  if (!context) {
    URL.revokeObjectURL(loaded.url);
    throw new Error("Referans fotoğrafını güvenli boyuta küçültmek için görsel alanı oluşturulamadı.");
  }
  try {
    context.drawImage(loaded.image, 0, 0, canvas.width, canvas.height);
    const resizedBlob = await canvasToBlob(canvas, blob.type);
    return await loadImage(resizedBlob);
  } finally {
    URL.revokeObjectURL(loaded.url);
  }
}

function faceFromLandmarks(landmarks, sourceWidth, sourceHeight) {
  const bounds = getLandmarkBounds(landmarks);
  if (!bounds) return null;
  return {
    ...bounds,
    shapeRatio: measureFaceShapeRatio(landmarks, sourceWidth, sourceHeight),
    // Jaw contour and fixed points in full-image pixels, for the jawline warp.
    jaw: jawPointsFromMesh(landmarks, sourceWidth, sourceHeight),
    // Full mesh (normalized) and pose signature, for the real-photo face.
    mesh: landmarks.map((point) => ({ x: point.x, y: point.y })),
    pose: poseDescriptor(landmarks, sourceWidth, sourceHeight)
  };
}

// FaceLandmarker's built-in detector is tuned for faces that fill the frame.
// For a face the first pass missed, zoom into the region BlazeFace found and
// run the 478-point mesh there, then map the mesh back to the full photo.
function refineFaceInRegion(image, box, sourceWidth, sourceHeight) {
  const centerX = (box.x + box.width / 2) * sourceWidth;
  const centerY = (box.y + box.height / 2) * sourceHeight;
  const side = Math.max(box.width * sourceWidth, box.height * sourceHeight) * 2.2;
  const left = clamp(Math.round(centerX - side / 2), 0, sourceWidth - 1);
  const top = clamp(Math.round(centerY - side / 2), 0, sourceHeight - 1);
  const regionWidth = Math.min(sourceWidth - left, Math.round(side));
  const regionHeight = Math.min(sourceHeight - top, Math.round(side));
  if (regionWidth < 8 || regionHeight < 8) return null;

  const zoom = clamp(640 / Math.max(regionWidth, regionHeight), 1, 4);
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(regionWidth * zoom);
  canvas.height = Math.round(regionHeight * zoom);
  const context = canvas.getContext("2d");
  if (!context) return null;
  context.imageSmoothingQuality = "high";
  context.drawImage(image, left, top, regionWidth, regionHeight, 0, 0, canvas.width, canvas.height);

  const meshes = state.detector.detect(canvas).faceLandmarks || [];
  const region = {
    x: left / sourceWidth,
    y: top / sourceHeight,
    width: regionWidth / sourceWidth,
    height: regionHeight / sourceHeight
  };
  return meshes
    .map((landmarks) => faceFromLandmarks(mapLandmarksFromRegion(landmarks, region), sourceWidth, sourceHeight))
    // BlazeFace boxes can be much larger or smaller than the face itself, so
    // match by containment rather than IoU.
    .filter((face) => face && rectOverlap(face, box) > 0.4)
    .sort((a, b) => rectIoU(b, box) - rectIoU(a, box))[0] || null;
}

// Two detections of one face: they overlap well, or one sits inside the other
// (BlazeFace's box around a distant face can be three times the face).
function isSameFace(a, b) {
  return rectIoU(a, b) > 0.25 || rectOverlap(a, b) > 0.5;
}

async function detectFaces(image) {
  if (!state.detector) throw new Error("Yüz algılama motoru henüz hazır değil.");
  const sourceWidth = image.naturalWidth || image.width;
  const sourceHeight = image.naturalHeight || image.height;
  const faces = (state.detector.detect(image).faceLandmarks || [])
    .map((landmarks) => faceFromLandmarks(landmarks, sourceWidth, sourceHeight))
    .filter(Boolean);

  // Second pass: small, distant, turned or partly covered faces. It also
  // confirms the mesh faces: the mesh model sometimes "sees" a face in rust,
  // foliage or a pattern, which a second detector rarely agrees with.
  let confirmationRan = false;
  for (const range of ["full", "short"]) {
    try {
      const detector = await getReferenceFaceDetector(range);
      const candidates = (detector.detect(image).detections || [])
        .map((item) => normalizeFallbackFace(item, sourceWidth, sourceHeight))
        .filter((item) => item && item.confidence >= 0.5);
      confirmationRan = true;
      for (const candidate of candidates) {
        const known = faces.filter((face) => isSameFace(face, candidate.bounds));
        known.forEach((face) => { face.confirmed = true; });
        if (known.length) continue;
        const refined = refineFaceInRegion(image, candidate.bounds, sourceWidth, sourceHeight);
        const same = refined ? faces.filter((face) => isSameFace(face, refined)) : [];
        same.forEach((face) => { face.confirmed = true; });
        if (same.length) continue;
        faces.push({
          ...(refined || { ...candidate.bounds, shapeRatio: null, jaw: null, mesh: null, pose: null }),
          confirmed: true
        });
      }
    } catch (error) {
      console.info(`Ek yüz taraması (${range}) kullanılamadı.`, error);
    }
  }
  if (!confirmationRan) faces.forEach((face) => { face.confirmed = true; });
  // Refine even previously detected small faces once at a tighter crop.
  // Retain the original mesh unless the new detection overlaps strongly.
  for (let i=0;i<faces.length;i++) {
    const face=faces[i];
    if (!face.confirmed || Math.min(face.width*sourceWidth,face.height*sourceHeight)>=160) continue;
    try {
      const improved=refineFaceInRegion(image,face,sourceWidth,sourceHeight);
      if (improved?.mesh?.length===478 && rectIoU(face,improved)>.55 && portraitGeometry(improved.mesh,sourceWidth,sourceHeight)) {
        faces[i]={...improved,confirmed:true};
      }
    } catch (error) { console.info("Küçük yüz için ikinci mesh taraması atlandı.",error); }
  }
  return orderDetectedFaces(faces);
}

function drawTargetPreview() {
  if (!state.target) return;
  const image = state.target.image;
  const scale = Math.min(1, MAX_PREVIEW_EDGE / Math.max(image.naturalWidth, image.naturalHeight));
  const width = Math.max(1, Math.round(image.naturalWidth * scale));
  const height = Math.max(1, Math.round(image.naturalHeight * scale));
  const canvas = elements.targetCanvas;
  const context = canvas.getContext("2d", { alpha: false });
  canvas.width = width;
  canvas.height = height;
  context.drawImage(image, 0, 0, width, height);

  state.target.faces.forEach((face, index) => {
    const x = face.x * width;
    const y = face.y * height;
    const boxWidth = face.width * width;
    const boxHeight = face.height * height;
    const selected = index === state.selectedFaceIndex;
    context.save();
    context.lineWidth = Math.max(3, width / 750);
    context.strokeStyle = selected ? "#79e9c5" : "rgba(246, 250, 255, .92)";
    context.fillStyle = selected ? "rgba(65, 211, 164, .12)" : "rgba(255,255,255,.025)";
    context.fillRect(x, y, boxWidth, boxHeight);
    context.strokeRect(x, y, boxWidth, boxHeight);
    const label = "YÜZ " + (index + 1);
    const fontSize = Math.max(12, Math.min(24, width / 65));
    context.font = "700 " + fontSize + "px system-ui";
    const labelWidth = context.measureText(label).width + 18;
    const labelHeight = fontSize + 12;
    const labelY = Math.max(0, y - labelHeight - 5);
    context.fillStyle = selected ? "#79e9c5" : "rgba(245,248,252,.95)";
    context.fillRect(x, labelY, labelWidth, labelHeight);
    context.fillStyle = "#0b1713";
    context.fillText(label, x + 9, labelY + fontSize + 2);
    context.restore();
  });
}

function renderFaceChips() {
  elements.faceChips.replaceChildren();
  const faces = state.target?.faces || [];
  elements.faceSelection.hidden = !state.target || faces.length === 0;
  elements.faceCount.textContent = faces.length + " yüz";

  faces.forEach((face, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "face-chip";
    button.textContent = "Yüz " + (index + 1);
    button.setAttribute("aria-pressed", String(index === state.selectedFaceIndex));
    button.addEventListener("click", () => selectTargetFace(index));
    elements.faceChips.append(button);
  });
}

function clearResult() {
  state.results.forEach((item) => URL.revokeObjectURL(item.url));
  state.results = [];
  state.result = null;
  elements.resultView.hidden = true;
  elements.resultTools.hidden = true;
  elements.resultImage.removeAttribute("src");
  elements.originalImage.removeAttribute("src");
  elements.historyStrip.replaceChildren();
  elements.outputPlaceholder.hidden = false;
  elements.outputStage.classList.remove("has-result");
  elements.outputSymbol.hidden = false;
  elements.outputTitle.textContent = "Önizleme";
  elements.outputStatus.textContent = "Beklemede";
  elements.outputMessage.textContent = "Fotoğrafını yükle, ardından yüz düzenlemeyi başlat.";
  updateSwapAvailability();
}

const MAX_RESULT_HISTORY = 6;

function showProfileHealth(median, outliers) {
  const health = describeProfileHealth(median, outliers);
  if (!health) return;
  elements.profileHealth.textContent = health.text;
  elements.profileHealth.classList.toggle("is-warning", health.warn);
  elements.profileHealth.hidden = false;
}

// Press and hold the result to see the original, release to see the result.
function initializeHoldToCompare() {
  const view = elements.resultView;
  const hold = (on) => view.classList.toggle("is-holding", on);
  view.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 && event.pointerType === "mouse") return;
    hold(true);
  });
  for (const type of ["pointerup", "pointerleave", "pointercancel"]) view.addEventListener(type, () => hold(false));
  view.addEventListener("contextmenu", (event) => event.preventDefault());
}

function resultStrength() {
  return Number(elements.strengthRange.value) / 100;
}

function updateResultView() {
  const split = Number(elements.compareRange.value);
  elements.resultView.style.setProperty("--split", split + "%");
  elements.resultView.style.setProperty("--strength", String(resultStrength()));
  elements.resultView.classList.toggle("is-comparing", split > 0);
  elements.strengthValue.textContent = "%" + elements.strengthRange.value;
}

function renderResultHistory() {
  elements.historyStrip.replaceChildren();
  elements.historyCount.textContent = state.results.length + " / " + MAX_RESULT_HISTORY;
  const best = bestResultIndex(state.results);
  state.results.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "history-item";
    button.setAttribute("aria-pressed", String(item === state.result));
    button.title = item.label;
    const image = document.createElement("img");
    image.src = item.url;
    image.alt = "";
    image.loading = "lazy";
    const caption = document.createElement("span");
    const score = similarityScore(item.similarity);
    caption.textContent = score === null ? String(index + 1) : String(score);
    if (index === best && state.results.length > 1) {
      button.classList.add("is-best");
      button.title = item.label + " · en benzer görünüm";
    }
    button.append(image, caption);
    button.addEventListener("click", () => selectResult(index));
    elements.historyStrip.append(button);
  });
}

function selectResult(index) {
  const item = state.results[index];
  if (!item || !state.target) return;
  state.result = item;
  elements.originalImage.src = state.target.url;
  elements.resultImage.src = item.url;
  elements.resultView.hidden = false;
  elements.resultTools.hidden = false;
  elements.outputPlaceholder.hidden = true;
  elements.outputStage.classList.add("has-result");
  elements.outputSymbol.hidden = true;
  elements.outputTitle.textContent = "Görünüm " + (index + 1);
  elements.outputStatus.textContent = item.label;
  const quality = [
    describeSimilarity(item.similarity, item.before, item.referenceMedian),
    describeNaturalness(item.naturalness),
    item.identityModel ? "kimlik modeli: " + ({ hyperswap_1b_256: "HyperSwap 1b", alphaface_256: "AlphaFace" }[item.identityModel] || item.identityModel) : "",
    item.faceEdit ? `yüz içi değişim %${(item.faceEdit.changedRatio*100).toFixed(1).replace('.',',')} · ortalama en büyük RGB kanal farkı ${item.faceEdit.meanDelta.toFixed(1).replace('.',',')}/255` : "",
    describeAdaptiveQuality(item.similarity, item.before, item.identityAspiration, item.finalistAttempts),
    describeFluxStatus(item.fluxStatus),
    describeFluxMetrics(item.fluxIdentityDelta,item.fluxDetailDelta,item.fluxStrength),
    item.fluxBlendMode === 'texture' ? 'FLUX ölçülen aday: sınırlı alın/yanak dokusu' : '',
    item.fluxBlendMode === 'texture' && Number.isFinite(item.fluxSkinGain) ? 'girişe göre cilt doku uyumu farkı '+(100*item.fluxSkinGain).toFixed(1).replace('.',',') : '',
    describeNvidiaJudge(item.nvidiaJudge, item.nvidiaJudgeVotes, item.nvidiaJudgeAgreement, item.nvidiaJudgeLayout),
    describeIdentitySufficiency(item.identitySufficient),
    describeSelfFidelity(item.selfFidelity),
    describeUltraSchedule(item.ultraSchedule, item.calibrationSamples),
    describeUltraDepth(item.ultraDepth, item.ultraRefinements, item.realismFinalists),
    describeFineFeatures(item.fineFeatures, item.fineDetail, item.catchlights, item.sclera),
    describeUltraTiming(item.ultraTiming),
    describeNativeFinish(item.nativeFinish),
    item.nvidiaJudge === 'error' ? describeJudgeError(item.nvidiaJudgeError) : '',
    item.nvidiaJudgeReason ? ({NOT_ENOUGH_VOTES:'geçerli jüri oyu yetersiz',SPLIT_VOTE:'jüri oyları ayrıştı',NARROW_MARGIN:'jüri puan farkı yetersiz',CONSENSUS:'jüri uzlaşısı'})[item.nvidiaJudgeReason] || item.nvidiaJudgeReason : '',
    item.nvidiaJudgeModels ? 'Jüri modelleri: '+item.nvidiaJudgeModels : '',
    item.nvidiaJudgeFinalists > 0 ? 'jüri finalisti: '+item.nvidiaJudgeFinalists : '',
    item.nimEditor && item.nimEditor !== "retake-local/flux2-klein-4b" ? 'NVIDIA yerel model: '+item.nimEditor : '',
    item.detailGate === "passed" ? "bölgesel ayrıntı kontrolü geçti" : "",
    item.detailGate.startsWith("failed:") ? "uyarı: bölgesel ayrıntı zayıf (" + ({eyes:"gözler",nose:"burun",mouth:"ağız",lower_face:"çene"}[item.detailGate.slice(7).split("-")[0]] || "yüz") + ")" : "",
    item.detailRegions ? "Orijinale göre doku enerjisi: " + Object.entries({eyes:"göz",nose:"burun",mouth:"ağız",lower_face:"alt yüz"}).map(([region,name]) => { const part=item.detailRegions.split(",").find(value=>value.startsWith(region+":")); const ratio=part ? Number(part.split(":")[1]) : NaN; return name+" "+(Number.isFinite(ratio) && ratio>=0 ? ratio.toFixed(2)+"×" : "ölçülemedi"); }).join(", ") : "",
    item.regionalEdit ? "Bölgesel RGB farkı (/255): " + item.regionalEdit.split(",").map(part => { const [region,value]=part.split(":"); const name=({eyes:"göz",nose:"burun",mouth:"ağız",lower_face:"alt yüz",moustache:"bıyık",chin:"çene",forehead:"alın",brows:"kaşlar",left_cheek:"sol yanak",right_cheek:"sağ yanak",left_jaw:"sol çene",right_jaw:"sağ çene"})[region]; const delta=Number(value); return name && Number.isFinite(delta) && delta>=0 && delta<=255 ? name+" "+delta.toFixed(1) : ""; }).filter(Boolean).join(", ") : "",
    describePortraitEvidence(item.lowerTexture,item.referenceAppearance),
    describePortraitGeometry(item.portraitGeometry),
    validQualityRevision(item.diagnostics?.["x-retake-quality-revision"]) ? "kalite sürümü "+item.diagnostics["x-retake-quality-revision"] : "",
    describeSourcePortraitQuality(item.sourcePortraitQuality),
    item.inputPreparation === 'native-contrast' ? 'küçük yüz için yerel kontrast hazırlığı kullanıldı' : '',
    item.neckStatus === "applied" ? "boyun/boğaz ten uyumu: "+item.neckPixels+" piksel" : item.neckStatus && item.neckStatus !== "off" ? "boyun korundu: "+(({ "no-mesh":"yüz işaretleri yetersiz","pose-uncertain":"poz belirsiz","skin-uncertain":"ten alanı belirsiz","neck-not-visible":"boyun görünmüyor","already-balanced":"ek düzeltme gerekli değil","quality-rejected":"kalite kontrolü","error":"işlem hatası","unreadable":"görüntü okunamadı"})[item.neckStatus] || "uygun aday yok") : "",
    describeNeckEvidence(item.neckEvidence),
    item.photoBalance ? "orijinal fotoğraf dokusu dengelendi" : "",
    item.before === null ? "orijinal kimlik ölçülemedi" : "",
    item.realism ? "fotoğraf gerçekçiliği" : "",
    item.texture && item.texture !== "average" ? "gerçek referans cilt dokusu" : "",
    item.realFace ? "gerçek fotoğraf dokusu" : "",
    item.lowerFace ? "referans sakal/çene dokusu" : "",
    item.completeIdentity ? "tam yüz referans ayrıntısı" : "",
    item.jawFocused ? "çene geometrisi uyarlaması" : "",
    item.rimCleanup ? "yüz kenarı yumuşatıldı" : "",
    item.identitySelection === "identity-only" ? "referans sakal/çene aktarımı finalde seçilemedi" : ""
  ].filter(Boolean).join(" · ");
  elements.qualityBadge.textContent = quality;
  elements.qualityBadge.hidden = !quality;
  updateResultView();
  renderResultHistory();
  updateSwapAvailability();
}

function addResult(blob, label, scores = {}) {
  const item = {
    blob,
    url: createObjectURL(blob),
    label,
    faceEdit: scores.faceEdit ?? null,
    nimEditor: scores.nimEditor || '',
    nvidiaJudge: scores.nvidiaJudge || "",
    identitySufficient: scores.identitySufficient || "",
    selfFidelity: scores.selfFidelity || "",
    ultraSchedule: scores.ultraSchedule || "",
    calibrationSamples: Number.isInteger(scores.calibrationSamples) ? scores.calibrationSamples : null,
    ultraDepth: scores.ultraDepth || "",
    fineFeatures: scores.fineFeatures || "",
    ultraTiming: scores.ultraTiming || "",
    nativeFinish: scores.nativeFinish && typeof scores.nativeFinish === "object" ? scores.nativeFinish : null,
    portraitGeometry: scores.portraitGeometry ?? null,
    sourcePortraitQuality: scores.sourcePortraitQuality ?? null,
    diagnostics: scores.diagnostics && typeof scores.diagnostics === "object" ? scores.diagnostics : null,
    qualityComparison: scores.qualityComparison || null,
    fineDetail: scores.fineDetail || "",
    catchlights: scores.catchlights || "",
    sclera: scores.sclera || "",
    ultraRefinements: scores.ultraRefinements || "",
    realismFinalists: Number.isInteger(scores.realismFinalists) ? scores.realismFinalists : null,
    nvidiaJudgeVotes: Number.isInteger(scores.nvidiaJudgeVotes) ? scores.nvidiaJudgeVotes : null,
    nvidiaJudgeAgreement: Number.isFinite(scores.nvidiaJudgeAgreement) ? scores.nvidiaJudgeAgreement : null,
    fluxStatus: scores.fluxStatus || "",
    fluxIdentityDelta: scores.fluxIdentityDelta ?? null,
    fluxDetailDelta: scores.fluxDetailDelta ?? null,
    fluxStrength: scores.fluxStrength ?? null,
    fluxBlendMode: scores.fluxBlendMode || '',
    fluxSkinGain: scores.fluxSkinGain ?? null,
    identityAspiration: scores.identityAspiration ?? null,
    finalistAttempts: scores.finalistAttempts ?? null,
    nvidiaJudgeError: scores.nvidiaJudgeError || "",
    nvidiaJudgeReason: scores.nvidiaJudgeReason || "",
    nvidiaJudgeModels: scores.nvidiaJudgeModels || "",
    nvidiaJudgeLayout: scores.nvidiaJudgeLayout || "",
    nvidiaJudgeFinalists: Number.isInteger(scores.nvidiaJudgeFinalists) ? scores.nvidiaJudgeFinalists : null,
    detailGate: scores.detailGate || "",
    detailRegions: scores.detailRegions || "",
    regionalEdit: scores.regionalEdit || "",
    inputPreparation: scores.inputPreparation || "",
    lowerTexture: scores.lowerTexture || "",
    referenceAppearance: scores.referenceAppearance ?? null,
    neckStatus: scores.neckStatus || "",
    neckPixels: scores.neckPixels ?? null,
    neckEvidence: scores.neckEvidence || '',
    photoBalance: scores.photoBalance === true,
    similarity: scores.similarity ?? null,
    before: scores.before ?? null,
    realFace: scores.realFace === true,
    lowerFace: scores.lowerFace === true,
    completeIdentity: scores.completeIdentity === true,
    jawFocused: scores.jawFocused === true,
    rimCleanup: scores.rimCleanup === true,
    identitySelection: scores.identitySelection || "",
    realism: scores.realism === true,
    texture: scores.texture || "average",
    referenceMedian: scores.referenceMedian ?? null,
    naturalness: scores.naturalness ?? null,
    identityModel: scores.identityModel || ""
  };
  state.results.push(item);
  while (state.results.length > MAX_RESULT_HISTORY) {
    URL.revokeObjectURL(state.results.shift().url);
  }
  elements.compareRange.value = "0";
  elements.strengthRange.value = "100";
  selectResult(state.results.length - 1);
}

// The file that is saved or shared: the selected look at the chosen strength
// and format. At full strength as PNG it is the untouched lossless result.
function measureTransparency(image) {
  try {
    const width = image.naturalWidth || image.width;
    const height = image.naturalHeight || image.height;
    const scale = Math.min(1, 256 / Math.max(width, height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    const context = canvas.getContext("2d", { willReadFrequently: true });
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    return transparentFraction(context.getImageData(0, 0, canvas.width, canvas.height).data);
  } catch {
    return 0;
  }
}

async function buildExportFile() {
  let format = elements.exportFormat.value === "jpeg" ? "jpeg" : "png";
  if (format === "jpeg" && state.target?.transparent) {
    // JPG has no transparency: a removed background would be saved as black.
    format = "png";
    showToast("Fotoğrafın arka planı saydam; JPG'de siyah olacağı için PNG olarak kaydedildi.");
  }
  const strength = resultStrength();
  let blob = state.result.blob;
  if (format === "jpeg" || strength < 1) {
    const [original, result] = await Promise.all([loadImage(state.target.file), loadImage(state.result.blob)]);
    try {
      const canvas = document.createElement("canvas");
      canvas.width = original.image.naturalWidth;
      canvas.height = original.image.naturalHeight;
      // PNG keeps a removed (transparent) background instead of turning it black.
      const context = canvas.getContext("2d", { alpha: format === "png" });
      if (!context) throw new Error("Dışa aktarma alanı oluşturulamadı.");
      context.drawImage(original.image, 0, 0);
      context.globalAlpha = strength;
      context.drawImage(result.image, 0, 0, canvas.width, canvas.height);
      blob = await new Promise((resolve, reject) => canvas.toBlob(
        (value) => (value ? resolve(value) : reject(new Error("Görsel dışa aktarılamadı."))),
        "image/" + format,
        0.95
      ));
    } finally {
      URL.revokeObjectURL(original.url);
      URL.revokeObjectURL(result.url);
    }
  }
  const extension = format === "jpeg" ? "jpg" : "png";
  return new File([blob], "retake-" + Date.now() + "." + extension, { type: "image/" + format });
}

function isLikelyMobile() {
  return /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) ||
    (navigator.maxTouchPoints > 1 && /Macintosh/.test(navigator.userAgent));
}

function selectTargetFace(index) {
  if (state.isSwapping || state.targetLoading) return;
  if (!state.target || !Number.isInteger(index) || !state.target.faces[index]) return;
  state.selectedFaceIndex = index;
  state.nextVariation = 0;
  clearResult();
  drawTargetPreview();
  renderFaceChips();
  updateSwapAvailability();
}

async function setTargetPhoto(file) {
  if (state.isSwapping || state.targetLoading) return;
  const validation = validateImageFile(file);
  if (!validation.valid) {
    showToast(validation.reason);
    return;
  }
  const token = ++state.targetLoadToken;
  state.targetLoading = true;
  elements.targetMeta.textContent = "Fotoğraf yükleniyor…";
  updateSwapAvailability();
  try {
    const loaded = await loadImage(file);
    if (token !== state.targetLoadToken) {
      URL.revokeObjectURL(loaded.url);
      return;
    }
    const dimensions = validateImageDimensions(loaded.image.naturalWidth, loaded.image.naturalHeight);
    if (!dimensions.valid) {
      URL.revokeObjectURL(loaded.url);
      state.targetLoading = false;
      elements.targetMeta.textContent = dimensions.reason;
      updateSwapAvailability();
      showToast(dimensions.reason);
      return;
    }
    let faces = [];
    let detectionMessage = "Yüz algılama yapılamadı.";
    if (state.detector) {
      try {
        faces = await detectFaces(loaded.image);
        detectionMessage = faces.length === 0
          ? "Fotoğrafta yüz bulunamadı."
          : faces.length + " yüz algılandı.";
      } catch (error) {
        detectionMessage = error.message;
      }
    } else {
      detectionMessage = "Yüz algılama motoru hazırlanıyor; fotoğraf şimdilik önizleme için açıldı.";
    }

    if (token !== state.targetLoadToken) {
      URL.revokeObjectURL(loaded.url);
      return;
    }
    if (state.target?.url) URL.revokeObjectURL(state.target.url);
    clearResult();
    const transparent = measureTransparency(loaded.image) > TRANSPARENT_TARGET_FRACTION;
    state.target = { file, image: loaded.image, url: loaded.url, faces, transparent };
    state.nextVariation = 0;
    state.targetLoading = false;
    state.selectedFaceIndex = faces.length > 0 ? 0 : null;
    elements.targetEmpty.hidden = true;
    elements.targetPreview.hidden = false;
    elements.clearTarget.hidden = false;
    elements.targetMeta.textContent =
      loaded.image.naturalWidth + " × " + loaded.image.naturalHeight + " px · " + formatBytes(file.size) + " · " + detectionMessage;
    drawTargetPreview();
    renderFaceChips();
    updateSwapAvailability();
    // The background warning must not be overwritten by the detection toast.
    showToast(transparent
      ? detectionMessage + " Bu fotoğrafın arka planı silinmiş (saydam): Retake yalnız yüzü değiştirir, arka planı geri getiremez. Arka planın görünmesi için orijinal fotoğrafı seç."
      : detectionMessage);
  } catch (error) {
    if (token === state.targetLoadToken) {
      state.targetLoading = false;
      elements.targetMeta.textContent = error.message || "Fotoğraf yüklenemedi.";
      updateSwapAvailability();
    }
    showToast(error.message || "Fotoğraf yüklenemedi.");
  }
}

function clearTargetPhoto() {
  if (state.isSwapping || state.targetLoading) return;
  state.targetLoadToken += 1;
  if (state.target?.url) URL.revokeObjectURL(state.target.url);
  clearResult();
  state.target = null;
  state.nextVariation = 0;
  state.selectedFaceIndex = null;
  elements.targetCanvas.width = 1;
  elements.targetCanvas.height = 1;
  elements.targetPreview.hidden = true;
  elements.targetEmpty.hidden = false;
  elements.clearTarget.hidden = true;
  elements.faceSelection.hidden = true;
  elements.faceChips.replaceChildren();
  elements.targetMeta.textContent = "Henüz fotoğraf seçilmedi";
  elements.targetInput.value = "";
  updateSwapAvailability();
}

function renderReferences() {
  elements.referencesGrid.replaceChildren();
  const readyCount = state.references.filter((item) => item.status === "ready").length;
  elements.referenceCount.textContent = readyCount + " / " + MAX_REFERENCE_FACES;
  elements.referencesEmpty.hidden = readyCount > 0;
  if (state.profileIsBuilding) {
    updateProfileIndicator({
      title: "Yüz profili oluşturuluyor",
      detail: "Referanslar hizalanıyor, kalite ağırlıkları ve kimlik profili hesaplanıyor.",
      progress: 72
    });
  } else if (state.identityProfile?.validCount === MAX_REFERENCE_FACES) {
    updateProfileIndicator({
      title: "Kişisel yüz profili hazır",
      detail: "Kimlik ve görünüm profili bu tarayıcıda önbelleğe alındı.",
      progress: 100,
      status: "ready"
    });
  } else if (state.referencesLoading) {
    updateProfileIndicator({
      title: "Profil kaynakları hazırlanıyor",
      detail: readyCount > 0 ? "Kaynak görseller kontrol ediliyor." : "Yerel profil kaynakları okunuyor.",
      progress: 8 + readyCount / MAX_REFERENCE_FACES * 34
    });
  } else if (readyCount === MAX_REFERENCE_FACES) {
    updateProfileIndicator({
      title: "Kaynak seti hazır",
      detail: "İlk düzenlemede yüz hizalama ve ArcFace kontrolü otomatik tamamlanacak.",
      progress: 46
    });
  } else {
    updateProfileIndicator({
      title: "Profil kaynakları eksik",
      detail: state.referenceLoadError || "Yerel referans dosyaları kontrol edilmeli.",
      progress: readyCount / MAX_REFERENCE_FACES * 36,
      status: "error"
    });
  }

  if (readyCount === 0) {
    const title = elements.referencesEmpty.querySelector("strong");
    const description = elements.referencesEmpty.querySelector("p");
    title.textContent = state.referencesLoading
      ? "Referans fotoğrafları yükleniyor"
      : "Henüz referans fotoğrafı yok";
    description.textContent = state.referencesLoading
      ? `${MAX_REFERENCE_FACES} fotoğraf hazırlanıyor.`
      : (state.referenceLoadError || "Kendi 25 fotoğrafını seç veya varsayılan havuza dön.");
  }

  state.references.forEach((reference) => {
    const card = document.createElement("article");
    card.className = "reference-card";

    if (reference.status === "ready" && reference.url) {
      const image = document.createElement("img");
      image.className = "reference-image";
      image.src = reference.url;
      image.alt = reference.name;
      image.loading = "lazy";
      image.decoding = "async";
      card.append(image);
    } else {
      const placeholder = document.createElement("div");
      placeholder.className = "reference-error";
      placeholder.textContent = reference.status === "loading" ? "Yükleniyor…" : "Görsel açılamadı";
      card.append(placeholder);
    }

    const name = document.createElement("span");
    name.className = "reference-name";
    name.textContent = reference.name;

    const label = document.createElement("span");
    label.className = "reference-state";
    label.textContent = reference.status !== "ready"
      ? (reference.status === "loading" ? "YÜKLENİYOR" : "AÇILAMADI")
      : reference.profileStatus === "included"
        ? (reference.profileMethod?.startsWith("FaceDetector") ? "PROFİLDE · YEDEK" : "PROFİLDE")
        : reference.profileStatus === "excluded"
          ? "PROFİLE ALINAMADI"
          : "YERELDE";
    if (reference.error || reference.profileError) label.title = reference.error || reference.profileError;

    card.append(name, label);
    elements.referencesGrid.append(card);
  });

  elements.reloadReferences.disabled = state.referencesLoading;
  elements.reloadReferences.textContent = state.referencesLoading ? "Yükleniyor…" : "Varsayılan havuza dön";
  elements.referenceProgress.textContent = state.referencesLoading && state.referenceLoadProgress
    ? `Fotoğraflar açılıyor · ${state.referenceLoadProgress.completed}/${state.referenceLoadProgress.total}`
    : state.profilePreparationProgress
      ? `Yüzler hizalanıyor · ${state.profilePreparationProgress.completed}/${state.profilePreparationProgress.total}`
    : "Fotoğraflar tarayıcıda işlenir; kişisel profil önbelleğini istediğin zaman temizleyebilirsin.";
  updateSwapAvailability();
}

async function loadLocalReferences({ force = false } = {}) {
  if (state.isSwapping) return 0;
  const token = ++state.referenceLoadToken;
  clearResult();
  state.identityProfile = null;
  state.referenceSetKey = null;
  state.referencesLoading = true;
  state.referenceLoadProgress = { completed: 0, total: MAX_REFERENCE_FACES };
  state.referenceLoadError = null;
  state.references.forEach((reference) => {
    if (reference.url) URL.revokeObjectURL(reference.url);
  });
  state.references = [];
  state.nextVariation = 0;
  renderReferences();

  try {
    const manifestUrl = new URL(REFERENCE_MANIFEST_URL, window.location.href).href;
    const { response: manifestResponse } = await fetchReferenceManifest(manifestUrl, { force });
    if (!manifestResponse.ok) {
      throw new Error("reference-faces/manifest.json bulunamadı.");
    }

    const references = parseReferenceManifest(await manifestResponse.json());
    if (token !== state.referenceLoadToken) return 0;
    state.references = references.map((reference) => ({ ...reference, status: "loading" }));
    renderReferences();

    const imageBaseUrl = new URL("./reference-faces/", window.location.href).href;
    const photoUrls = references.map((reference) => getReferenceImageUrl(reference.file, imageBaseUrl));
    // Photos are downloaded once and then read from this device's cache.
    requestPersistentStorage();
    let downloaded = 0;
    // Two at a time: 25 multi-megapixel photos decoded together stall the page
    // (and can exhaust a phone's memory).
    const results = await settleWithLimit(references, 2, async (reference, index) => {
      const endpoint = photoUrls[index];
      const { response, fromCache } = await fetchCachedReference(endpoint, {
        force,
        init: { headers: { Accept: "image/avif,image/webp,image/png,image/jpeg" } }
      });
      if (!fromCache && response.ok) downloaded += 1;
      if (!response.ok) throw new Error(`${reference.file}: HTTP ${response.status}`);

      const blob = await response.blob();
      if (!blob.size || blob.size > MAX_IMAGE_BYTES) {
        throw new Error(`${reference.file}: görsel boş veya 25 MB sınırını aşıyor.`);
      }

      const loaded = await loadReferenceImage(blob);
      return { ...reference, status: "ready", url: loaded.url, image: loaded.image, blob, size: blob.size };
    });

    if (token !== state.referenceLoadToken) {
      results.forEach((result) => {
        if (result.status === "fulfilled") URL.revokeObjectURL(result.value.url);
      });
      return 0;
    }

    if (results.every((result) => result.status === "fulfilled")) {
      pruneReferenceCache(photoUrls, manifestUrl).catch(() => {});
    }
    console.info(`Referans fotoğrafları: ${MAX_REFERENCE_FACES - downloaded} önbellekten, ${downloaded} indirildi.`);
    state.references = results.map((result, index) => result.status === "fulfilled"
      ? result.value
      : ({ ...references[index], status: "error", error: result.reason?.message || "Görsel açılamadı." }));
    const readyReferences = state.references.filter((reference) => reference.status === "ready" && reference.blob);
    const loadedCount = readyReferences.length;
    const failedCount = MAX_REFERENCE_FACES - loadedCount;
    if (loadedCount === MAX_REFERENCE_FACES) {
      try {
        state.referenceSetKey = await createReferenceSetKey(readyReferences);
      } catch (error) {
        console.warn("Referans profil anahtarı oluşturulamadı; bu oturumda profil bellekte tutulacak.", error);
      }
    }
    state.referencesLoading = false;
    state.referenceLoadProgress = null;
    state.profilePreparationProgress = null;
    state.referenceLoadError = failedCount
      ? `${loadedCount}/${MAX_REFERENCE_FACES} görsel yüklendi; ${failedCount} dosyayı manifest ve dosya adıyla karşılaştır.`
      : null;

    renderReferences();
    return loadedCount;
  } catch (error) {
    if (token !== state.referenceLoadToken) return 0;
    state.references = [];
    state.referencesLoading = false;
    state.referenceLoadProgress = null;
    state.profilePreparationProgress = null;
    state.referenceLoadError = error.message || "Yerel referans dosyaları okunamadı.";
    renderReferences();
    showToast(state.referenceLoadError);
    return 0;
  }
}

async function loadUserReferences(files) {
  if (state.isSwapping) return;
  const validation = validateReferenceFiles(files);
  if (!validation.valid) {
    showToast(validation.reason);
    return;
  }

  const token = ++state.referenceLoadToken;
  state.nextVariation = 0;
  clearResult();
  state.identityProfile = null;
  state.referenceSetKey = null;
  state.referenceFaceCropCount = null;
  state.selectedReferenceCount = 0;
  state.referencesLoading = true;
  state.referenceLoadProgress = { completed: 0, total: files.length };
  state.profilePreparationProgress = null;
  state.referenceLoadError = null;
  state.references.forEach((reference) => {
    if (reference.url) URL.revokeObjectURL(reference.url);
  });
  state.references = files.map((file, index) => ({
    file: file.name,
    name: file.name || `Kişisel referans ${index + 1}`,
    status: "loading"
  }));
  renderReferences();

  const results = [];
  for (let index = 0; index < files.length; index += 1) {
    const file = files[index];
    const name = file.name || `Kişisel referans ${index + 1}`;
    try {
      const loaded = await loadReferenceImage(file);
      results.push({
        status: "fulfilled",
        value: {
          file: file.name,
          name,
          status: "ready",
          url: loaded.url,
          image: loaded.image,
          blob: file,
          size: file.size
        }
      });
    } catch (error) {
      results.push({
        status: "rejected",
        reason: error,
        value: {
          file: file.name,
          name,
          status: "error",
          error: error?.message || "Görsel açılamadı."
        }
      });
    }

    if (token !== state.referenceLoadToken) {
      results.forEach((result) => {
        if (result.value?.url) URL.revokeObjectURL(result.value.url);
      });
      return;
    }
    state.references[index] = results[index].value;
    state.referenceLoadProgress.completed = index + 1;
    renderReferences();
    await new Promise((resolve) => window.setTimeout(resolve, 0));
  }

  state.references = results.map((result) => result.value);

  const readyReferences = state.references.filter((reference) => reference.status === "ready");
  if (readyReferences.length === MAX_REFERENCE_FACES) {
    try {
      state.referenceSetKey = await createReferenceSetKey(readyReferences);
    } catch (error) {
      console.warn("Kişisel referans profil anahtarı oluşturulamadı.", error);
    }
  }

  if (token !== state.referenceLoadToken) {
    state.references.forEach((reference) => {
      if (reference.url) URL.revokeObjectURL(reference.url);
    });
    return;
  }

  state.referencesLoading = false;
  const failedCount = MAX_REFERENCE_FACES - readyReferences.length;
  state.referenceLoadError = failedCount
    ? `${readyReferences.length}/${MAX_REFERENCE_FACES} fotoğraf açıldı; başarısız dosyaları yeniden seç.`
    : null;
  renderReferences();
  showToast(`${readyReferences.length}/${MAX_REFERENCE_FACES} kişisel referans yüklendi.`);
}

async function canvasToBlob(canvas, type = "image/png") {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => blob ? resolve(blob) : reject(new Error("Görsel hazırlanamadı.")),
      type,
      type === "image/jpeg" ? 0.92 : undefined
    );
  });
}

async function clearReferenceData() {
  if (state.referencesLoading || state.isSwapping) return;
  const confirmed = await showAppDialog({
    title: "Profil verilerini temizle?",
    message: "Seçili referanslar ve bu tarayıcıdaki kırpılmış profil önbelleği silinir. Kaynak fotoğraflar dosyalarından kaldırılmaz.",
    confirmLabel: "Temizle",
    danger: true
  });
  if (!confirmed) return;

  state.referenceLoadToken += 1;
  state.identityProfile = null;
  state.referenceSetKey = null;
  state.referenceFaceCropCount = null;
  state.selectedReferenceCount = 0;
  state.referencesLoading = false;
  state.referenceLoadProgress = null;
  state.profilePreparationProgress = null;
  state.referenceLoadError = null;
  state.references.forEach((reference) => {
    if (reference.url) URL.revokeObjectURL(reference.url);
  });
  state.references = [];
  clearResult();
  renderReferences();

  const removed = await deleteCachedProfile();
  showToast(removed
    ? "Seçili fotoğraflar ve yerel profil önbelleği temizlendi."
    : "Fotoğraflar kaldırıldı ancak yerel profil önbelleği silinemedi; tekrar deneyin.");
}

const PROFILE_PIPELINE_VERSION = "retake-25-reference-realface-v7";
const FACEFUSION_REFERENCE_WIDTH = 512;
const FACEFUSION_REFERENCE_HEIGHT = 682;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function measureAlignedReference(canvas, landmarks, face, sourceWidth, sourceHeight) {
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const { width, height } = canvas;
  const imageData = context.getImageData(0, 0, width, height);
  const data = imageData.data;
  const leftEye = landmarks[33];
  const rightEye = landmarks[263];
  const nose = landmarks[1];
  const eyeMidX = (leftEye.x + rightEye.x) / 2;
  const eyeDistance = Math.hypot(
    (rightEye.x - leftEye.x) * sourceWidth,
    (rightEye.y - leftEye.y) * sourceHeight
  );
  const noseOffset = Math.abs(nose.x - eyeMidX) * sourceWidth;
  const frontalness = clamp(1 - (noseOffset / Math.max(eyeDistance, 1)) / 0.34, 0.12, 1);
  const faceAreaRatio = face.width * face.height;

  let sumLuma = 0;
  let sumLumaSquared = 0;
  let laplacianSum = 0;
  let laplacianSquared = 0;
  let samples = 0;
  const xMin = Math.floor(width * 0.16);
  const xMax = Math.floor(width * 0.84);
  const yMin = Math.floor(height * 0.16);
  const yMax = Math.floor(height * 0.88);
  const lumaAt = (x, y) => {
    const offset = (y * width + x) * 4;
    return data[offset] * 0.2126 + data[offset + 1] * 0.7152 + data[offset + 2] * 0.0722;
  };

  for (let y = yMin + 1; y < yMax - 1; y += 2) {
    for (let x = xMin + 1; x < xMax - 1; x += 2) {
      const luma = lumaAt(x, y);
      sumLuma += luma;
      sumLumaSquared += luma * luma;
      const laplacian = 4 * luma - lumaAt(x - 1, y) - lumaAt(x + 1, y) -
        lumaAt(x, y - 1) - lumaAt(x, y + 1);
      laplacianSum += laplacian;
      laplacianSquared += laplacian * laplacian;
      samples += 1;
    }
  }

  const meanLuma = samples ? sumLuma / samples : 128;
  const exposure = clamp(1 - Math.abs(meanLuma - 128) / 160, 0.15, 1);
  const laplacianMean = samples ? laplacianSum / samples : 0;
  const laplacianVariance = samples
    ? Math.max(0, laplacianSquared / samples - laplacianMean * laplacianMean)
    : 0;
  const sharpness = clamp(Math.log1p(laplacianVariance) / Math.log1p(700), 0.08, 1);

  // Normalize moderate exposure differences before averaging all reference faces.
  const gain = clamp(126 / Math.max(meanLuma, 1), 0.78, 1.28);
  if (Math.abs(gain - 1) > 0.01) {
    for (let index = 0; index < data.length; index += 4) {
      data[index] = Math.round(clamp(data[index] * gain, 0, 255));
      data[index + 1] = Math.round(clamp(data[index + 1] * gain, 0, 255));
      data[index + 2] = Math.round(clamp(data[index + 2] * gain, 0, 255));
    }
    context.putImageData(imageData, 0, 0);
  }

  const weight = calculateReferenceWeight({ faceAreaRatio, frontalness, sharpness, exposure });
  return { weight, frontalness, sharpness, exposure, faceAreaRatio, imageData };
}

function createAlignedReferenceCanvas(image, landmarks) {
  const leftEye = landmarks[33];
  const rightEye = landmarks[263];
  if (!leftEye || !rightEye) return null;

  const sourceWidth = image.naturalWidth || image.width;
  const sourceHeight = image.naturalHeight || image.height;
  const transform = getSimilarityTransform(
    { x: leftEye.x * sourceWidth, y: leftEye.y * sourceHeight },
    { x: rightEye.x * sourceWidth, y: rightEye.y * sourceHeight },
    { x: 168, y: 238 },
    { x: 344, y: 238 }
  );
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 682;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Yüz profili için hizalama alanı oluşturulamadı.");
  context.fillStyle = "#777777";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.setTransform(transform.a, transform.b, transform.c, transform.d, transform.e, transform.f);
  context.drawImage(image, 0, 0);
  context.setTransform(1, 0, 0, 1, 0, 0);
  return canvas;
}

async function getReferenceFaceDetector(range = "short") {
  const normalizedRange = range === "full" ? "full" : "short";
  const detectorKey = normalizedRange === "full" ? "referenceDetectorFull" : "referenceDetector";
  const promiseKey = normalizedRange === "full" ? "referenceDetectorFullPromise" : "referenceDetectorPromise";
  if (state[detectorKey]) return state[detectorKey];
  if (!state[promiseKey]) {
    state[promiseKey] = (async () => {
      const vision = state.vision || await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/wasm"
      );
      const options = {
        baseOptions: { modelAssetPath: FACE_DETECTOR_MODEL_URLS[normalizedRange], delegate: "GPU" },
        runningMode: "IMAGE",
        minDetectionConfidence: normalizedRange === "full" ? 0.15 : 0.20,
        minSuppressionThreshold: 0.30
      };
      try {
        return await FaceDetector.createFromOptions(vision, options);
      } catch (gpuError) {
        console.info(`Referans ${normalizedRange}-range yüz algılama GPU'da başlatılamadı; CPU deneniyor.`, gpuError);
        options.baseOptions.delegate = "CPU";
        return FaceDetector.createFromOptions(vision, options);
      }
    })();
  }
  try {
    state[detectorKey] = await state[promiseKey];
    return state[detectorKey];
  } catch (error) {
    state[promiseKey] = null;
    throw error;
  }
}

function normalizeFallbackFace(detection, sourceWidth, sourceHeight) {
  const box = detection?.boundingBox;
  const keypoints = detection?.keypoints || [];
  if (!box || keypoints.length < 3 || !Number.isFinite(box.originX) ||
      !Number.isFinite(box.originY) || !Number.isFinite(box.width) ||
      !Number.isFinite(box.height) || box.width <= 0 || box.height <= 0) return null;

  const bounds = {
    x: clamp(box.originX / sourceWidth, 0, 1),
    y: clamp(box.originY / sourceHeight, 0, 1),
    width: clamp(box.width / sourceWidth, 0, 1),
    height: clamp(box.height / sourceHeight, 0, 1)
  };
  bounds.width = Math.min(bounds.width, 1 - bounds.x);
  bounds.height = Math.min(bounds.height, 1 - bounds.y);
  if (bounds.width <= 0 || bounds.height <= 0) return null;

  // BlazeFace returns six normalized points: right eye, left eye, nose,
  // mouth center and two ear tragions. The eye/nose indices mirror the
  // FaceLandmarker indices used by Retake's canonical alignment code.
  const landmarks = new Array(478).fill(null);
  const centerX = bounds.x + bounds.width / 2;
  const centerY = bounds.y + bounds.height / 2;
  landmarks[33] = keypoints[0];
  landmarks[263] = keypoints[1];
  landmarks[1] = keypoints[2];
  landmarks[10] = { x: centerX, y: bounds.y };
  landmarks[152] = { x: centerX, y: bounds.y + bounds.height };
  landmarks[234] = { x: bounds.x, y: centerY };
  landmarks[454] = { x: bounds.x + bounds.width, y: centerY };

  const confidence = Number(detection.categories?.[0]?.score);
  return {
    landmarks,
    bounds,
    confidence: Number.isFinite(confidence) ? confidence : 0
  };
}

function buildReferenceFaceCandidate(reference, detected, sourceWidth, sourceHeight, detectionMethod) {
  const image = reference.image;
  const face = detected.bounds;
  const crop = getFaceCropRect(
    sourceWidth,
    sourceHeight,
    face,
    FACEFUSION_REFERENCE_WIDTH / FACEFUSION_REFERENCE_HEIGHT
  );
  const canvas = document.createElement("canvas");
  canvas.width = FACEFUSION_REFERENCE_WIDTH;
  canvas.height = FACEFUSION_REFERENCE_HEIGHT;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Referans yüzü için görsel alanı oluşturulamadı.");
  context.drawImage(
    image,
    crop.x, crop.y, crop.width, crop.height,
    0, 0, canvas.width, canvas.height
  );

  const alignedCanvas = createAlignedReferenceCanvas(image, detected.landmarks);
  if (!alignedCanvas) throw new Error("Göz noktaları hizalama için kullanılamadı.");
  const quality = measureAlignedReference(
    alignedCanvas,
    detected.landmarks,
    face,
    sourceWidth,
    sourceHeight
  );
  // BlazeFace's six keypoints are a reliable fallback, but less detailed than
  // the 478-point mesh. Keep its contribution conservative in both profiles.
  const weight = detectionMethod.startsWith("FaceDetector")
    ? Math.min(2, quality.weight)
    : quality.weight;

  return {
    canvas,
    alignedCanvas,
    // Only the full mesh carries real forehead/chin/cheek points.
    shapeRatio: detectionMethod === "FaceLandmarker"
      ? measureFaceShapeRatio(detected.landmarks, sourceWidth, sourceHeight)
      : null,
    jaw: detectionMethod === "FaceLandmarker" ? canonicalJaw(detected.landmarks, sourceWidth, sourceHeight) : null,
    jawFrontalness: detectionMethod === "FaceLandmarker" ? meshFrontalness(detected.landmarks, sourceWidth, sourceHeight) : 0,
    pose: detectionMethod === "FaceLandmarker" ? poseDescriptor(detected.landmarks, sourceWidth, sourceHeight) : null,
    profilePixels: quality.imageData.data,
    weight,
    quality,
    detectionMethod,
    detectionConfidence: detected.confidence ?? 1,
    name: reference.name,
    file: reference.file
  };
}

async function createReferenceFaceInput(reference) {
  const image = reference.image;
  const sourceWidth = image.naturalWidth || image.width;
  const sourceHeight = image.naturalHeight || image.height;
  let primaryError = "FaceLandmarker yüz bulamadı.";

  try {
    const detection = state.detector.detect(image);
    const faces = (detection.faceLandmarks || [])
      .map((landmarks) => ({ landmarks, bounds: getLandmarkBounds(landmarks), confidence: 1 }))
      .filter((item) => item.bounds)
      .sort((a, b) =>
        (b.bounds.width * b.bounds.height) - (a.bounds.width * a.bounds.height)
      );

    for (const detected of faces) {
      try {
        return buildReferenceFaceCandidate(
          reference, detected, sourceWidth, sourceHeight, "FaceLandmarker"
        );
      } catch (error) {
        primaryError = error?.message || "FaceLandmarker hizalaması başarısız.";
      }
    }
  } catch (error) {
    primaryError = error?.message || "FaceLandmarker çalıştırılamadı.";
  }

  const fallbackErrors = [];
  for (const range of ["short", "full"]) {
    const detectorName = range === "short" ? "BlazeFace-short" : "BlazeFace-full";
    try {
      const fallbackDetector = await getReferenceFaceDetector(range);
      const detection = fallbackDetector.detect(image);
      const faces = (detection.detections || [])
        .map((item) => normalizeFallbackFace(item, sourceWidth, sourceHeight))
        .filter(Boolean)
        .sort((a, b) =>
          (b.bounds.width * b.bounds.height) - (a.bounds.width * a.bounds.height)
        );

      for (const detected of faces) {
        try {
          return buildReferenceFaceCandidate(
            reference, detected, sourceWidth, sourceHeight, `FaceDetector-${range}`
          );
        } catch (error) {
          fallbackErrors.push(`${detectorName}: ${error?.message || "hizalama başarısız"}`);
        }
      }
      if (faces.length === 0) fallbackErrors.push(`${detectorName}: yüz bulunamadı`);
    } catch (error) {
      fallbackErrors.push(`${detectorName}: ${error?.message || "algılayıcı başlatılamadı"}`);
    }
  }

  throw new Error(
    `FaceLandmarker: ${primaryError}; ${fallbackErrors.join("; ")}`
  );
}

function createAppearanceProfile(candidates) {
  const weightedPixels = weightedAverageRGBA(
    candidates.map((candidate) => ({
      data: candidate.profilePixels,
      weight: candidate.weight
    }))
  );
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 682;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Kişisel görünüm profili oluşturulamadı.");
  context.putImageData(new ImageData(weightedPixels, canvas.width, canvas.height), 0, 0);
  return {
    canvas,
    weights: candidates.map((candidate) => candidate.weight),
    validCount: candidates.length,
    totalWeight: candidates.reduce((sum, candidate) => sum + candidate.weight, 0)
  };
}

async function buildLocalFaceFusionInputs(exactReference=null) {
  state.profileIsBuilding = true;
  updateProfileIndicator({
    title: "Yüz profili oluşturuluyor",
    detail: "Referanslar hizalanıyor, kalite ağırlıkları hesaplanıyor ve önbellek hazırlanıyor.",
    progress: 58
  });
  const readyReferences = state.references.filter(
    (reference) => reference.status === "ready" && reference.image && reference.blob
  );
  if (readyReferences.length !== MAX_REFERENCE_FACES) {
    throw new Error(`İşlem için ${MAX_REFERENCE_FACES} yerel referansın tamamı yüklenmiş olmalı.`);
  }
  const face = state.target?.faces?.[state.selectedFaceIndex];
  if (!face) throw new Error("Önce hedef fotoğraftaki değiştirilecek yüzü seç.");

  const crop = getFaceCropRect(
    state.target.image.naturalWidth,
    state.target.image.naturalHeight,
    face
  );

  // Typical crops are sent pixel-for-pixel so the photo around the face is not
  // softened by a resize round trip.
  const targetSize = getTargetRenderSize(crop.width, crop.height);
  const targetCanvas = document.createElement("canvas");
  targetCanvas.width = targetSize.width;
  targetCanvas.height = targetSize.height;
  const targetContext = targetCanvas.getContext("2d", { willReadFrequently: true });
  if (!targetContext) throw new Error("Hedef yüz görseli oluşturulamadı.");
  targetContext.fillStyle = "#10151d";
  targetContext.fillRect(0, 0, targetCanvas.width, targetCanvas.height);
  targetContext.imageSmoothingQuality = "high";
  targetContext.drawImage(
    state.target.image,
    crop.x, crop.y, crop.width, crop.height,
    0, 0, targetCanvas.width, targetCanvas.height
  );

  const profileKey = `${PROFILE_PIPELINE_VERSION}:${state.referenceSetKey || await createReferenceSetKey(readyReferences)}`;
  if (!state.identityProfile || state.identityProfile.key !== profileKey) {
    const cached = await getCachedProfile(profileKey);
    const cachedIsValid = cached &&
      Array.isArray(cached.referenceBlobs) &&
      cached.referenceBlobs.length === MAX_REFERENCE_FACES &&
      cached.validCount === MAX_REFERENCE_FACES &&
      cached.referenceBlobs.every((blob) => blob instanceof Blob && blob.size > 0) &&
      cached.appearanceProfileBlob instanceof Blob &&
      cached.appearanceProfileBlob.size > 0 &&
      Array.isArray(cached.weights) &&
      cached.weights.length === cached.referenceBlobs.length;

    if (cachedIsValid) {
      readyReferences.forEach((reference) => {
        reference.profileStatus = "included";
        reference.profileMethod = "cache";
        reference.profileError = null;
      });
      renderReferences();
      state.identityProfile = {
        key: profileKey,
        token: state.referenceLoadToken,
        referenceBlobs: cached.referenceBlobs,
        weights: cached.weights,
        appearanceProfileBlob: cached.appearanceProfileBlob,
        validCount: cached.validCount,
        totalWeight: cached.totalWeight,
        shapeRatio: Number.isFinite(cached.shapeRatio) ? cached.shapeRatio : null,
        jawContour: Array.isArray(cached.jawContour) ? cached.jawContour : null,
        poseDescriptors: Array.isArray(cached.poseDescriptors) ? cached.poseDescriptors : null
      };
      state.profileIsBuilding = false;
      updateProfileIndicator({
        title: "Kişisel yüz profili hazır",
        detail: "Daha önce oluşturulan profil bu cihazın yerel önbelleğinden açıldı.",
        progress: 100,
        status: "ready"
      });
    } else {
      const candidates = [];
      const referenceDiagnostics = [];
      state.profilePreparationProgress = { completed: 0, total: readyReferences.length };
      renderReferences();
      for (let index = 0; index < readyReferences.length; index += 1) {
        const reference = readyReferences[index];
        try {
          const candidate = await createReferenceFaceInput(reference);
          candidates.push(candidate);
          reference.profileStatus = "included";
          reference.profileMethod = candidate.detectionMethod;
          reference.profileError = null;
          referenceDiagnostics.push({
            file: reference.file,
            status: "included",
            method: candidate.detectionMethod,
            weight: candidate.weight,
            confidence: candidate.detectionConfidence
          });
        } catch (error) {
          const reason = error?.message || "Yüz tespiti veya hizalama başarısız.";
          reference.profileStatus = "excluded";
          reference.profileMethod = null;
          reference.profileError = reason;
          referenceDiagnostics.push({
            file: reference.file,
            status: "excluded",
            reason
          });
          console.warn("Referans yüz profili hazırlanamadı:", reference.name, reason);
        }
        state.profilePreparationProgress.completed = index + 1;
        renderReferences();
        await new Promise((resolve) => window.setTimeout(resolve, 0));
      }
      state.profilePreparationProgress = null;
      renderReferences();

      if (candidates.length !== MAX_REFERENCE_FACES) {
        const missing = referenceDiagnostics
          .filter((item) => item.status !== "included")
          .map((item) => `${item.file}: ${item.reason}`)
          .join(" | ");
        throw new Error(
          `25 fotoğrafın tamamı profile alınamadı (${candidates.length}/${MAX_REFERENCE_FACES}). ` +
          "Eksik referanslar: " + missing + ". İşlem 25 fotoğrafın tamamı olmadan durduruldu."
        );
      }

      const appearance = createAppearanceProfile(candidates);
      const [referenceBlobs, appearanceProfileBlob] = await Promise.all([
        Promise.all(candidates.map((candidate) => canvasToBlob(candidate.canvas))),
        canvasToBlob(appearance.canvas)
      ]);
      const profile = {
        referenceBlobs,
        weights: appearance.weights,
        appearanceProfileBlob,
        validCount: appearance.validCount,
        totalWeight: appearance.totalWeight,
        shapeRatio: summarizeReferenceShape(candidates.map((candidate) => ({
          value: candidate.shapeRatio,
          // Near-frontal, sharp references describe the face shape best.
          weight: candidate.quality.frontalness * candidate.weight
        }))),
        // The person's typical jaw/chin contour, from frontal, sharp references:
        // a turned head foreshortens one side of the jaw.
        jawContour: summarizeReferenceJaw(candidates
          .filter((candidate) => candidate.jaw && candidate.jawFrontalness >= 0.6)
          .map((candidate) => ({ points: candidate.jaw, weight: candidate.jawFrontalness * candidate.weight }))),
        // Pose/expression signature of each reference, in reference order, to
        // pick the photo whose real face can be warped onto the target.
        poseDescriptors: candidates.map((candidate) => candidate.pose),
        referenceDiagnostics
      };
      state.identityProfile = {
        key: profileKey,
        token: state.referenceLoadToken,
        ...profile
      };
      const persisted = await saveCachedProfile(profileKey, profile);
      state.profileIsBuilding = false;
      updateProfileIndicator({
        title: "Kişisel yüz profili hazır",
        detail: persisted
          ? "Profil bu cihazda güvenli tarayıcı alanına kaydedildi."
          : "Profil bu oturum için hazır; kalıcı önbellek kullanılamadı.",
        progress: 100,
        status: "ready"
      });
      state.referenceFaceCropCount = appearance.validCount;
      state.selectedReferenceCount = appearance.validCount;
      renderReferences();
    }
  }

  const profile = state.identityProfile;
  state.referenceFaceCropCount = profile.validCount;
  state.selectedReferenceCount = profile.validCount;
  const scaleX = targetCanvas.width / crop.width;
  const scaleY = targetCanvas.height / crop.height;
  const faceX = clamp(crop.face.x * scaleX, 0, targetCanvas.width - 1);
  const faceY = clamp(crop.face.y * scaleY, 0, targetCanvas.height - 1);
  const faceBounds = {
    x: faceX,
    y: faceY,
    width: Math.min(crop.face.width * scaleX, targetCanvas.width - faceX),
    height: Math.min(crop.face.height * scaleY, targetCanvas.height - faceY)
  };
  // Lengthen/shorten/widen/narrow the head so the reference person's face
  // proportions sit naturally on it.
  const headFit = estimateHeadFit(profile.shapeRatio, face.shapeRatio);
  // Move the photo's jaw/chin contour towards the reference person's.
  const toCanvas = (point) => ({ x: (point.x - crop.x) * scaleX, y: (point.y - crop.y) * scaleY });
  const jawWarp = face.jaw && profile.jawContour
    ? planJawWarp({
      jaw: face.jaw.jaw.map(toCanvas),
      anchors: face.jaw.anchors.map(toCanvas),
      eyes: face.jaw.eyes.map(toCanvas),
      frontalness: face.jaw.frontalness
    }, profile.jawContour, { strength: 1.0, maxShift: 0.10 })
    : null;
  // Always prepare pose-matched real-reference crops for the lower-face-only
  // candidate. This does NOT activate the full-face transplant switch.
  // If the mesh/pose is unreliable, the FaceFusion identity path continues.
  const realFace = await prepareRealFace(face, profile, readyReferences, toCanvas, exactReference);
  // The face's mesh in crop pixels: the engine finds the beard and cheeks
  // with it to give the person's own facial-hair tone.
  const targetLandmarks = face.mesh
    ? meshToPixels(face.mesh, state.target.image.naturalWidth, state.target.image.naturalHeight, toCanvas)
    : null;

  return {
    targetCanvas,
    referenceBlobs: profile.referenceBlobs,
    exactReferenceIndex: exactReference ? readyReferences.indexOf(exactReference) : -1,
    referenceWeights: profile.weights,
    appearanceProfileBlob: profile.appearanceProfileBlob,
    // Only scalar pose distances leave the device: each reference's mesh
    // signature stays in the local profile; the server can try a smaller
    // pose-matched identity pool as well as the full 25-photo mean.
    poseDistances: Array.isArray(profile.poseDescriptors) && face.pose
      ? profile.poseDescriptors.map((descriptor) => {
          const distance = descriptorDistance(descriptor, face.pose);
          return Number.isFinite(distance) ? Math.round(distance * 10000) / 10000 : null;
        })
      : null,
    profileCount: profile.validCount,
    totalProfileWeight: profile.totalWeight,
    faceBounds,
    targetSize: { width: targetCanvas.width, height: targetCanvas.height },
    headFit,
    jawWarp,
    realFace,
    targetLandmarks,
    crop
  };
}

// Full-face paste remains opt-in. Retake v18 tries *only* a conservative
// beard/chin/moustache reference candidate automatically in Ultra mode.
const REAL_FACE_STORAGE_KEY = "retake-real-face-v2";

function isRealFaceEnabled() {
  return elements.realFaceToggle ? elements.realFaceToggle.checked : false;
}

function restoreRealFacePreference() {
  if (!elements.realFaceToggle) return;
  try {
    elements.realFaceToggle.checked = window.localStorage.getItem(REAL_FACE_STORAGE_KEY) === "on";
  } catch { /* storage unavailable: keep the default (off) */ }
  elements.realFaceToggle.addEventListener("change", () => {
    try {
      window.localStorage.setItem(REAL_FACE_STORAGE_KEY, elements.realFaceToggle.checked ? "on" : "off");
    } catch { /* ignore */ }
  });
}

// NVIDIA visual jury (via Hafize): opt-in, off by default, and still confirmed
// per job in the consent dialog because face crops leave this computer.
const NVIDIA_JUDGE_STORAGE_KEY = "retake-nvidia-judge-v1";

function isNvidiaJudgeEnabled() {
  return Boolean(
    state.faceFusionStatus?.nvidiaJudge && elements.nvidiaJudgeToggle &&
    !elements.nvidiaJudgeToggle.disabled && elements.nvidiaJudgeToggle.checked
  );
}

function restoreNvidiaJudgePreference() {
  if (!elements.nvidiaJudgeToggle) return;
  try {
    elements.nvidiaJudgeToggle.checked = window.localStorage.getItem(NVIDIA_JUDGE_STORAGE_KEY) === "on";
  } catch { /* storage unavailable: keep the default (off) */ }
  elements.nvidiaJudgeToggle.addEventListener("change", () => {
    try {
      window.localStorage.setItem(NVIDIA_JUDGE_STORAGE_KEY, elements.nvidiaJudgeToggle.checked ? "on" : "off");
    } catch { /* ignore */ }
  });
}

// Quality depth (engine v23+): the person may trade waiting time for more
// measured candidates. Remembered on this device; "deep" by default.
function qualityDepthInputs() {
  return elements.qualityDepthRow ? [...elements.qualityDepthRow.querySelectorAll('input[name="qualityDepth"]')] : [];
}

function supportsQualityDepth() {
  return Boolean(state.faceFusionStatus?.configured && state.faceFusionStatus.qualityDepths);
}

function selectedQualityDepth() {
  const checked = qualityDepthInputs().find((input) => input.checked);
  return requestedDepth(checked?.value, { supported: supportsQualityDepth() });
}

function updateQualityDepthNote() {
  if (!elements.qualityDepthNote) return;
  const depth = normalizeDepth(qualityDepthInputs().find((input) => input.checked)?.value);
  elements.qualityDepthNote.textContent = `${depthLabel(depth)}: ${QUALITY_DEPTHS[depth].detail}. ` +
    (depth === "standard" ? "" : "Bu süre boyunca sayfayı açık tut; ekran kilitlenmesin diye uyku engellenir.");
}

function updateQualityDepthAvailability() {
  if (!elements.qualityDepthRow) return;
  const supported = supportsQualityDepth();
  elements.qualityDepthRow.hidden = !supported;
  for (const depth of Object.keys(QUALITY_DEPTHS)) {
    const label = document.getElementById(`qualityDepth${depth[0].toUpperCase()}${depth.slice(1)}Time`);
    if (!label) continue;
    const [low, high] = depthMinutes(depth, state.faceFusionStatus?.ultraDepths);
    label.textContent = `≈${low}–${high} dk` + (depth === DEFAULT_QUALITY_DEPTH ? " · önerilen" : "");
  }
  updateQualityDepthNote();
}

function restoreQualityDepthPreference() {
  let saved = DEFAULT_QUALITY_DEPTH;
  try {
    saved = normalizeDepth(window.localStorage.getItem(QUALITY_DEPTH_STORAGE_KEY));
  } catch { /* storage unavailable: keep the default */ }
  for (const input of qualityDepthInputs()) {
    input.checked = input.value === saved;
    input.addEventListener("change", () => {
      if (!input.checked) return;
      try {
        window.localStorage.setItem(QUALITY_DEPTH_STORAGE_KEY, normalizeDepth(input.value));
      } catch { /* ignore */ }
      updateQualityDepthNote();
    });
  }
  updateQualityDepthNote();
}

// Long jobs: keep the screen awake (the page must stay open to collect the
// result) and warn before closing the tab. Both are best-effort.
const longJob = { wakeLock: null, active: false };

async function acquireWakeLock() {
  if (!longJob.active || longJob.wakeLock || document.visibilityState !== "visible") return;
  try {
    longJob.wakeLock = await navigator.wakeLock?.request("screen") ?? null;
    longJob.wakeLock?.addEventListener?.("release", () => { longJob.wakeLock = null; });
  } catch { longJob.wakeLock = null; }
}

function warnBeforeLeaving(event) {
  event.preventDefault();
  event.returnValue = "";
}

function startLongJob() {
  longJob.active = true;
  window.addEventListener("beforeunload", warnBeforeLeaving);
  acquireWakeLock();
}

function endLongJob() {
  longJob.active = false;
  window.removeEventListener("beforeunload", warnBeforeLeaving);
  const lock = longJob.wakeLock;
  longJob.wakeLock = null;
  lock?.release?.().catch?.(() => {});
}

document.addEventListener("visibilitychange", () => { if (longJob.active) acquireWakeLock(); });

// Real-photo face: the person's own reference photos closest in pose and
// expression (up to three), with their 478-point meshes and the target's. The
// engine warps each onto the target face and keeps the one that fits best, so
// real skin, nose and lips replace the generated ones. Returns null when no
// reference is close enough; the AI swap alone is then used.
async function prepareRealFace(face, profile, readyReferences, toCanvas, exactReference=null) {
  if (!face.mesh || !face.pose || !Array.isArray(profile.poseDescriptors)) return null;
  const candidates = [];
  for (const index of rankRealReferences(profile.poseDescriptors, profile.weights, face.pose, undefined, readyReferences.length)) {
    if(readyReferences[index]===exactReference)continue;
    if(candidates.length>=3)break;
    try {
      const candidate = await cropRealReference(readyReferences[index]);
      if (candidate) candidates.push(candidate);
    } catch (error) {
      console.info("Gerçek fotoğraf adayı hazırlanamadı:", readyReferences[index]?.name, error);
    }
  }
  if (!candidates.length) return null;
  return {
    candidates,
    targetLandmarks: meshToPixels(face.mesh, state.target.image.naturalWidth, state.target.image.naturalHeight, toCanvas)
  };
}

// The reference photo cropped around its face (at most 1024 px) with its
// 478-point mesh in the crop's pixels.
async function cropRealReference(reference) {
  const image = reference?.image;
  if (!image) return null;
  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;
  const landmarks = (state.detector.detect(image).faceLandmarks || [])
    .map((mesh) => ({ mesh, bounds: getLandmarkBounds(mesh) }))
    .filter((item) => item.bounds)
    .sort((a, b) => (b.bounds.width * b.bounds.height) - (a.bounds.width * a.bounds.height))[0]?.mesh;
  if (!landmarks || landmarks.length < 478) return null;
  const area = referenceCrop(landmarks, width, height);
  const canvas = document.createElement("canvas");
  canvas.width = area.outWidth;
  canvas.height = area.outHeight;
  const context = canvas.getContext("2d");
  if (!context) return null;
  context.imageSmoothingQuality = "high";
  context.drawImage(image, area.x, area.y, area.width, area.height, 0, 0, canvas.width, canvas.height);
  return {
    blob: await canvasToBlob(canvas),
    name: reference.name,
    landmarks: meshToPixels(landmarks, width, height,
      (point) => ({ x: (point.x - area.x) * area.scale, y: (point.y - area.y) * area.scale }))
  };
}

function applyGeneratedFacePatch(generatedTargetImage, crop, preblended = false, binaryTransport = false, renderSize = crop) {
  if (!state.target?.image) throw new Error("Hedef fotoğraf bellekte bulunamadı.");
  const sourceImage = state.target.image;
  const imageWidth = sourceImage.naturalWidth;
  const imageHeight = sourceImage.naturalHeight;
  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = crop.width;
  cropCanvas.height = crop.height;
  const cropContext = cropCanvas.getContext("2d", { willReadFrequently: true });
  cropContext.drawImage(sourceImage, crop.x, crop.y, crop.width, crop.height, 0, 0, crop.width, crop.height);
  const sourceCrop = cropContext.getImageData(0, 0, crop.width, crop.height);

  const generatedCanvas = document.createElement("canvas");
  generatedCanvas.width = crop.width;
  generatedCanvas.height = crop.height;
  const generatedContext = generatedCanvas.getContext("2d", { willReadFrequently: true });
  const outputWidth = generatedTargetImage.naturalWidth || generatedTargetImage.width;
  const outputHeight = generatedTargetImage.naturalHeight || generatedTargetImage.height;
  if (binaryTransport && (outputWidth !== renderSize.width || outputHeight !== renderSize.height)) {
    throw new Error("Yerel motor kayıpsız çıktı boyutunu koruyamadı.");
  }
  // Binary extent must be decoded at native render size before scaling. A
  // scaled transparent bitmap would reintroduce fractional-alpha artifacts.
  if (binaryTransport) {
    generatedCanvas.width = outputWidth;
    generatedCanvas.height = outputHeight;
  }
  generatedContext.drawImage(
    generatedTargetImage,
    0, 0, outputWidth, outputHeight,
    0, 0, generatedCanvas.width, generatedCanvas.height
  );
  let generatedCrop = generatedContext.getImageData(0, 0, generatedCanvas.width, generatedCanvas.height);
  if (preblended && !binaryTransport) {
    throw new Error("Eski yerel motor yumuşak alfa taşıyor. Kontur hatasını önlemek için Retake motorunu güncelleyip yeniden başlat.");
  }
  if (binaryTransport && (outputWidth !== crop.width || outputHeight !== crop.height)) {
    // Resample opaque preblended RGB independently from its binary extent.
    // Use nearest-neighbor extent; never read resampled low-alpha RGB.
    const nativeRGBA = generatedCrop.data;
    binaryExtentFromRGBA(nativeRGBA, outputWidth, outputHeight);
    const maskCanvas = document.createElement("canvas");
    maskCanvas.width = outputWidth;
    maskCanvas.height = outputHeight;
    const maskContext = maskCanvas.getContext("2d", { willReadFrequently: true });
    const opaque = new Uint8ClampedArray(nativeRGBA);
    const extent = new Uint8ClampedArray(nativeRGBA.length);
    const targetNative = document.createElement("canvas");
    targetNative.width = outputWidth;
    targetNative.height = outputHeight;
    const nativeContext = targetNative.getContext("2d", { willReadFrequently: true });
    nativeContext.drawImage(cropCanvas, 0, 0, outputWidth, outputHeight);
    const nativeSource = nativeContext.getImageData(0, 0, outputWidth, outputHeight).data;
    for (let i = 0; i < nativeRGBA.length; i += 4) {
      const a = nativeRGBA[i + 3];
      if (a === 0) opaque.set(nativeSource.subarray(i, i + 3), i);
      opaque[i + 3] = 255;
      extent.set([a, a, a, 255], i);
    }
    generatedContext.putImageData(new ImageData(opaque, outputWidth, outputHeight), 0, 0);
    maskContext.putImageData(new ImageData(extent, outputWidth, outputHeight), 0, 0);
    const scaled = document.createElement("canvas");
    scaled.width = crop.width;
    scaled.height = crop.height;
    const scaledContext = scaled.getContext("2d", { willReadFrequently: true });
    scaledContext.imageSmoothingQuality = "high";
    scaledContext.drawImage(generatedCanvas, 0, 0, crop.width, crop.height);
    generatedCrop = scaledContext.getImageData(0, 0, crop.width, crop.height);
    scaledContext.imageSmoothingEnabled = false;
    scaledContext.drawImage(maskCanvas, 0, 0, crop.width, crop.height);
    const maskPixels = scaledContext.getImageData(0, 0, crop.width, crop.height).data;
    for (let i = 0; i < generatedCrop.data.length; i += 4) generatedCrop.data[i + 3] = maskPixels[i];
  }
  const parserAlphaMask = binaryTransport
    ? binaryExtentFromRGBA(generatedCrop.data, crop.width, crop.height)
    : alphaMaskFromRGBA(generatedCrop.data, crop.width, crop.height);
  const hasTransparentPixels = parserAlphaMask.some((alpha) => alpha < 0.999);
  // New local FaceFusion server embeds its parser/occlusion-derived change matte
  // in PNG alpha. Retain the ellipse only as a backward-compatible fallback.
  if (binaryTransport && !preblended) {
    throw new Error("Yerel motorun kayıpsız alfa maskesi geçersiz; bozuk görüntü gösterilmedi.");
  }
  const alphaMask = binaryTransport || hasTransparentPixels
    ? parserAlphaMask
    : createFaceAlphaMask(crop.width, crop.height, crop.face);
  // Native resolution (v27): a face crop larger than the engine's limit was
  // scaled up again here, which softens it next to the original skin. Undo
  // the interpolation blur and restore the photo's own grain level, inside
  // the matte only (outside pixels stay byte-identical).
  let nativeFinish = null;
  const upscale = Math.max(crop.width / outputWidth, crop.height / outputHeight);
  if (binaryTransport && preblended && upscale >= MIN_UPSCALE) {
    nativeFinish = finishUpscaledFace(
      sourceCrop.data, generatedCrop.data, alphaMask, crop.width, crop.height, upscale,
      (Math.imul(crop.width, 73856093) ^ Math.imul(crop.height, 19349663)) >>> 0
    );
  }
  // A local engine may deliver final, already-masked RGB + an edit-extent
  // alpha. Never blend that image a second time: alpha squared erases identity
  // characteristics along facial transitions.
  const blended = preblended && (binaryTransport || hasTransparentPixels)
    ? compositePreblendedRGBA(sourceCrop.data, generatedCrop.data, alphaMask, crop.width, crop.height)
    : compositeMaskedRGBA(sourceCrop.data, generatedCrop.data, alphaMask, crop.width, crop.height);

  const fullCanvas = document.createElement("canvas");
  fullCanvas.width = imageWidth;
  fullCanvas.height = imageHeight;
  const fullContext = fullCanvas.getContext("2d", { willReadFrequently: true });
  fullContext.drawImage(sourceImage, 0, 0, imageWidth, imageHeight);
  const original = fullContext.getImageData(0, 0, imageWidth, imageHeight);
  const output = new ImageData(new Uint8ClampedArray(original.data), imageWidth, imageHeight);
  const fullMask = new Float32Array(imageWidth * imageHeight);

  for (let y = 0; y < crop.height; y += 1) {
    const sourceOffset = y * crop.width * 4;
    const outputOffset = ((crop.y + y) * imageWidth + crop.x) * 4;
    output.data.set(blended.subarray(sourceOffset, sourceOffset + crop.width * 4), outputOffset);
    fullMask.set(
      alphaMask.subarray(y * crop.width, (y + 1) * crop.width),
      (crop.y + y) * imageWidth + crop.x
    );
  }

  const integrity = inspectOutsideMaskIntegrity(original.data, output.data, fullMask, imageWidth, imageHeight);
  if (!integrity.ok) {
    throw new Error("Güvenlik kontrolü başarısız: yüz maskesi dışında " + integrity.changedPixels + " piksel değişti.");
  }

  // Measure the actual delivered crop at native resolution, after all scaling
  // and masks. A high embedding score alone never proves an edit occurred.
  const faceEdit=measureFaceEdit(sourceCrop.data,blended,crop.width,crop.height,crop.face);
  if(!faceEdit.meaningful)throw new Error('Son görüntüde yüz içi değişim yetersiz; düzenlenmemiş fotoğraf başarılı sonuç olarak gösterilmedi.');
  fullContext.putImageData(output, 0, 0);
  // Re-detect the actual delivered face locally. A landmark failure is not
  // fabricated as unchanged geometry and never discards a valid engine result.
  let geometry = null;
  const selectedFace = state.target?.faces?.[state.selectedFaceIndex];
  const sourceGeometry = portraitGeometry(selectedFace?.mesh,imageWidth,imageHeight);
  if (sourceGeometry) {
    geometry = {source:sourceGeometry,delivered:null};
    try {
      const detected = refineFaceInRegion(fullCanvas,selectedFace,imageWidth,imageHeight);
      geometry.delivered = portraitGeometry(detected?.mesh,imageWidth,imageHeight);
    } catch (error) { console.info("Son geometri ölçümü kullanılamadı.",error); }
  }
  const inputQuality = sourcePortraitQuality(sourceCrop.data,crop.width,crop.height,crop.face);
  return canvasToBlob(fullCanvas).then(blob => ({blob, faceEdit, nativeFinish, portraitGeometry:geometry, sourcePortraitQuality:inputQuality}));
}

const stones = { hero: null, wait: null, timer: null };
// Share of the hero (from the top) the sculpture is centred in on desktop,
// and the clearance kept under the sticky top bar.
const HERO_SCULPTURE_AREA = 0.6;
const HERO_SCULPTURE_TOP = 40;

function initializeStones() {
  const wide = window.matchMedia("(min-width: 851px)");
  stones.hero = mountSculpture($("#heroSculpture"), {
    scale: 1,
    // Desktop: full-width sculpture above the headline, like the Forma page.
    // Placement depends only on the hero box (not on where the text happens
    // to be once fonts load), so it is identical in both themes and on reload.
    place: (width, height, aspect) => wide.matches
      ? placeWideSculpture(width, height, aspect, height * HERO_SCULPTURE_AREA, HERO_SCULPTURE_TOP)
      : null
  });
}

// While the local engine works, the stone moves into the result panel so the
// wait is something to play with rather than a frozen screen.
function setWaitProgress(step, total, text) {
  elements.waitProgress.style.width = Math.round(100 * step / Math.max(total, 1)) + "%";
  elements.waitStep.textContent = text;
}

function showWaitStage(title) {
  elements.waitTitle.textContent = title;
  elements.waitTimer.textContent = "00:00";
  setWaitProgress(0, 1, "Yüz profili hazırlanıyor…");
  elements.waitStage.hidden = false;
  elements.outputStage.classList.add("is-waiting");
  if (!stones.wait) {
    stones.wait = mountSculpture($("#waitSculpture"), {
      scale: 0.96,
      autopilot: true,
      options: { ambientIntensity: 1.2, intro: true }
    });
  } else {
    stones.wait.refreshTheme();
    stones.wait.setAutopilot(true);
  }
  const startedAt = Date.now();
  window.clearInterval(stones.timer);
  stones.timer = window.setInterval(() => {
    elements.waitTimer.textContent = formatElapsed((Date.now() - startedAt) / 1000);
  }, 500);
}

function hideWaitStage() {
  window.clearInterval(stones.timer);
  stones.timer = null;
  elements.waitStage.hidden = true;
  elements.outputStage.classList.remove("is-waiting");
  stones.wait?.setAutopilot(false);
}

const JOB_POLL_INTERVAL_MS = 1500;

// The engine works on the swap as a background job and reports its real stage,
// so a long run never depends on a single open connection.
async function waitForEngineJob(jobId, total, timeoutMs = depthTimeoutMs("standard")) {
  const deadline = Date.now() + timeoutMs;
  let failures = 0;
  while (Date.now() < deadline) {
    await new Promise((resolve) => window.setTimeout(resolve, JOB_POLL_INTERVAL_MS));
    let job;
    try {
      const response = await engineFetch("/api/jobs/" + encodeURIComponent(jobId), {
        headers: { Accept: "application/json" }
      });
      job = await response.json();
      if (response.status === 404) throw Object.assign(new Error(job.error || "İş bulunamadı."), { fatal: true });
      failures = 0;
    } catch (error) {
      failures += 1;
      if (error.fatal || failures >= 8) throw error.fatal ? error : new Error("Motorla bağlantı koptu; işlem durumu alınamıyor.");
      continue;
    }
    if (job.status === "error") throw new Error(job.error || "FaceFusion işlemi başarısız.");
    const remaining = formatRemaining(job.eta_seconds);
    const minimumRemaining = Number(job.minimum_remaining_seconds) > 0
      ? formatRemaining(job.minimum_remaining_seconds)
      : "";
    setWaitProgress(
      job.step,
      job.total || total,
      `${Math.min(job.step + 1, job.total || total)} / ${job.total || total} · ${job.stage}` + (remaining ? ` · ${remaining}` : "") + (minimumRemaining ? ` · Ultra min. kalan ${minimumRemaining}` : "")
    );
    if (job.status === "done") return;
  }
  throw new Error(`İşlem ${Math.round(timeoutMs / 60000)} dakika içinde tamamlanmadı.`);
}

async function runFaceFusionEdit(options = {}) {
  // Also used as a click handler: only an explicit benchmark flag counts.
  const benchmark = options?.benchmark === true;
  if (state.isSwapping) return;
  if (!state.faceFusionStatus.configured) {
    showToast("Önce bilgisayarındaki FaceFusion servisini başlat.");
    return;
  }
  if (!state.target || !Number.isInteger(state.selectedFaceIndex)) {
    showToast("Önce hedef fotoğrafı ve yüzü seç.");
    return;
  }
  if (state.references.filter((item) => item.status === "ready").length !== MAX_REFERENCE_FACES) {
    showToast(`${MAX_REFERENCE_FACES} referansın tamamı yüklenmeden işlem başlatılamaz.`);
    return;
  }

  const useNvidiaJudge = isNvidiaJudgeEnabled();
  // A benchmark stays at the seven-minute depth so its results compare.
  const qualityDepth = requestedDepth(selectedQualityDepth(), { supported: supportsQualityDepth(), benchmark });
  const ultraDepths = state.faceFusionStatus.ultraDepths;
  const depthText = describeDepthDuration(qualityDepth, ultraDepths);
  const judgeNotice = useNvidiaJudge
    ? " NVIDIA görsel jürisi açık: motor, kimliği eşit en fazla 4 sonuç adayının yüz kırpımını (en çok 512 px) " +
      (state.faceFusionStatus?.nvidiaJudgeDirect ? "bilgisayarındaki NVIDIA anahtarıyla doğrudan" : "Hafize sunucun üzerinden") +
      " NVIDIA'nın barındırılan görsel modellerine gönderir. Jüri yalnız bu adaylardan en doğal görüneni seçer; 25 referans ve tam fotoğraf gönderilmez."
    : "";
  // A benchmark asked for consent once, for all of its photos.
  const consent = benchmark || await showAppDialog({
    title: state.result ? "Yeni bir görünüm oluştur?" : "Yerel düzenlemeyi başlat?",
    message: (state.engine.remote
      ? "Bu işlemde seçili fotoğrafın yüz kırpımı, 25 referanstan hazırlanan yüz profili ve gerçek doku için en uygun en fazla 3 referansın yüz kırpımı, şifreli tünel sağlayıcısı (Cloudflare veya Tailscale) üzerinden eşleştirdiğin kendi bilgisayarındaki FaceFusion motoruna gönderilir. Görseller bir AI servisine yüklenmez; işlem senin bilgisayarında yapılır."
      : "Bu işlemde seçili fotoğrafın yüz kırpımı, 25 referanstan hazırlanan yüz profili ve gerçek doku için en uygun en fazla 3 referansın yüz kırpımı bu cihazdaki FaceFusion servisine gönderilir. Görseller harici bir AI servisine yüklenmez.") + " Tek yüzlü fotoğraflarda, görünür üst boyun ve çene altındaki ten tonu güvenilir biçimde ayrılabiliyorsa yerel olarak dengelenebilir; boyun şekli değiştirilmez." + judgeNotice +
      (qualityDepth === "standard" ? "" : ` Kalite derinliği: ${depthText}. Motor daha çok ayarı ölçer, kazananın çevresinde ince ayar yapar ve birden fazla finaliste fotogerçekçilik geçişi uygular; kimlik ve doku kontrolleri aynı kalır. Sonucu alabilmek için bu sayfa açık kalmalı (ekranın uyuması engellenir).`),
    confirmLabel: state.engine.remote ? "Bilgisayarıma gönder" : "Yerel olarak devam et"
  });
  if (!consent) return;

  state.isSwapping = true;
  updateSwapAvailability();
  // Ultra always runs the complete bounded search on the server. No more
  // rotating a single head-fit variation when the user taps "generate".
  const variation = 0;
  elements.outputTitle.textContent = qualityDepth === "standard"
    ? "Ultra-420 · en az 7 dakika kalite araması"
    : `Ultra · ${depthText}`;
  elements.outputStatus.textContent = "Bilgisayarda çalışıyor";
  if (!state.result) {
    elements.outputPlaceholder.hidden = false;
    elements.outputPlaceholder.querySelector("strong").textContent = "Yüz profili hazırlanıyor…";
    elements.outputMessage.textContent = "İlk kullanımda profil hazırlanması biraz zaman alabilir.";
  }

  showWaitStage(qualityDepth === "standard"
    ? "Ultra-420 v22 · tam yüz, çene ve sakal referansları karşılaştırılıyor"
    : `Ultra ${depthLabel(qualityDepth)} · 24 ayar, ince ayar ve gerçekçilik finalistleri karşılaştırılıyor`);
  if (qualityDepth !== "standard" || benchmark) startLongJob();

  let generatedUrl = null;
  try {
    const targetSnapshot = state.target;
    const exactReference = await findExactReferenceFile(targetSnapshot.file, state.references);
    if (state.target !== targetSnapshot) throw new Error("Hedef fotoğraf değişti; işlemi yeniden başlat.");
    // All 25 references still build the identity profile. Do not copy the
    // identical target back as the direct photographic reference.
    if(exactReference)setWaitProgress(0,1,"Aynı dosya referans bankasında mevcut; diğer referanslardan yeni yüz ayrıntıları işleniyor…");
    const {
      targetCanvas,
      referenceBlobs,
      exactReferenceIndex,
      referenceWeights,
      appearanceProfileBlob,
      poseDistances,
      faceBounds,
      targetSize,
      headFit,
      jawWarp,
      realFace,
      targetLandmarks,
      crop
    } = await buildLocalFaceFusionInputs(exactReference);
    const targetBlob = await canvasToBlob(targetCanvas);

    const form = new FormData();
    form.append("target", targetBlob, "retake-target.png");
    form.append("appearance_profile", appearanceProfileBlob, "retake-25-photo-appearance-profile.png");
    referenceBlobs.forEach((blob, index) => {
      form.append("reference", blob, `retake-reference-${String(index + 1).padStart(2, "0")}.png`);
    });
    const metadata = { referenceWeights, poseDistances, faceBounds, targetSize, headFit, jawWarp, variation, neckHarmonize: state.target.faces.length === 1 && Boolean(targetLandmarks) };
    if (exactReferenceIndex >= 0) metadata.exactReferenceIndex = exactReferenceIndex;
    if (benchmark) {
      if (exactReferenceIndex < 0) throw new Error("Öz-test hedefi referans bankasında bulunamadı.");
      metadata.selfTest = true;
    }
    if (useNvidiaJudge) metadata.nvidiaJudge = true;
    // Only an engine that understands depths gets one (older ones reject nothing, but say nothing either).
    if (supportsQualityDepth()) metadata.qualityDepth = qualityDepth;
    if (targetLandmarks) metadata.targetLandmarks = targetLandmarks;
    if (realFace) {
      realFace.candidates.forEach((candidate, index) => {
        form.append("real_reference", candidate.blob, `retake-real-reference-${index + 1}.png`);
      });
      metadata.realFace = {
        // Both candidates are compared automatically in Ultra mode.
        // The explicit manual switch still forces full-face-only mode.
        mode: isRealFaceEnabled() ? "full" : "both",
        targetLandmarks: realFace.targetLandmarks,
        references: realFace.candidates.map((candidate) => ({ landmarks: candidate.landmarks }))
      };
    }
    form.append("metadata", JSON.stringify(metadata));
    // Hash only locally, before the request. Never include these keys in the
    // form or exported diagnostics. A failed hash simply disables comparison.
    const compareContext = await comparisonContext({target:targetSnapshot.file,
      referenceKey:state.referenceSetKey, faceIndex:state.selectedFaceIndex,
      depth:qualityDepth, judge:useNvidiaJudge, full:isRealFaceEnabled(), benchmark,
      prepared:[targetBlob,appearanceProfileBlob,...referenceBlobs,...(realFace?.candidates||[]).map(c=>c.blob)],metadata});

    setRuntimeStatus("Yerel yüz düzenleme · sonuç hazırlanıyor");

    setWaitProgress(0, 1, "Yüz kırpımları motora gönderiliyor…");
    let response = await engineFetch("/api/face-swap", { method: "POST", body: form });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error || `FaceFusion işlemi başarısız (HTTP ${response.status}).`);
    }
    if (response.status === 202) {
      const accepted = await response.json();
      await waitForEngineJob(accepted.job_id, accepted.total || 4, depthTimeoutMs(qualityDepth, ultraDepths));
      response = await engineFetch("/api/jobs/" + encodeURIComponent(accepted.job_id) + "/result");
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || `Sonuç alınamadı (HTTP ${response.status}).`);
      }
    }
    if (response.headers.get("X-Retake-Ultra") !== "on") {
      throw new Error("Yerel motor Ultra Mod sonucunu doğrulamadı. Motoru güncelleyip tekrar başlat.");
    }
    const completedRounds = Number(response.headers.get("X-Retake-Ultra-Rounds"));
    const attemptedRounds = Number(response.headers.get("X-Retake-Ultra-Attempted"));
    const winnerRound = Number(response.headers.get("X-Retake-Ultra-Winner"));
    const detailScore = parseSimilarity(response.headers.get("X-Retake-Detail-Score"));
    const scores = {
      nimEditor: response.headers.get('X-Retake-NIM-Editor') || '',
      nvidiaJudge: response.headers.get("X-Retake-NVIDIA-Judge") || "",
      nvidiaJudgeVotes: Number.parseInt(response.headers.get("X-Retake-NVIDIA-Judge-Votes") || "", 10),
      nvidiaJudgeAgreement: Number.parseFloat(response.headers.get("X-Retake-NVIDIA-Judge-Agreement") || ""),
      fluxStatus: response.headers.get("X-Retake-FLUX-Status") || "",
      fluxIdentityDelta: response.headers.get("X-Retake-FLUX-Identity-Delta"),
      fluxDetailDelta: response.headers.get("X-Retake-FLUX-Detail-Delta"),
      fluxStrength: response.headers.get("X-Retake-FLUX-Strength"),
      fluxBlendMode: response.headers.get("X-Retake-FLUX-Blend-Mode") || "",
      fluxSkinGain: Number.parseFloat(response.headers.get("X-Retake-FLUX-Skin-Gain") || ""),
      identityAspiration: parseSimilarity(response.headers.get("X-Retake-Identity-Aspiration")),
      finalistAttempts: Number.parseInt(response.headers.get("X-Retake-Finalist-Attempts") || "",10),
      nvidiaJudgeError: response.headers.get("X-Retake-NVIDIA-Judge-Error") || "",
      nvidiaJudgeReason: response.headers.get("X-Retake-NVIDIA-Judge-Reason") || "",
      nvidiaJudgeModels: response.headers.get("X-Retake-NVIDIA-Judge-Models") || "",
      nvidiaJudgeLayout: response.headers.get("X-Retake-NVIDIA-Judge-Layout") || "",
      nvidiaJudgeFinalists: Number.parseInt(response.headers.get("X-Retake-NVIDIA-Judge-Finalists") || "",10),
      detailGate: response.headers.get("X-Retake-Detail-Gate") || "",
      detailRegions: response.headers.get("X-Retake-Detail-Regions") || "",
      regionalEdit: response.headers.get("X-Retake-Regional-Edit") || "",
      inputPreparation: response.headers.get("X-Retake-Input-Preparation") || "",
      lowerTexture: response.headers.get("X-Retake-Lower-Texture") || "",
      referenceAppearance: response.headers.get("X-Retake-Reference-Appearance"),
      neckStatus: response.headers.get("X-Retake-Neck-Status") || "",
      neckPixels: Number.parseInt(response.headers.get("X-Retake-Neck-Pixels") || "",10),
      neckEvidence: response.headers.get("X-Retake-Neck-Evidence") || "",
      photoBalance: response.headers.get("X-Retake-Photo-Balance") === "on",
      similarity: parseSimilarity(response.headers.get("X-Retake-Similarity")),
      before: parseSimilarity(response.headers.get("X-Retake-Similarity-Before")),
      realFace: response.headers.get("X-Retake-Real-Face") === "on",
      lowerFace: response.headers.get("X-Retake-Lower-Face-Transfer") === "on",
      completeIdentity: response.headers.get("X-Retake-Complete-Identity") === "on",
      jawFocused: response.headers.get("X-Retake-Jaw-Focused") === "on",
      rimCleanup: response.headers.get("X-Retake-Rim-Cleanup") === "on",
      identitySelection: response.headers.get("X-Retake-Identity-Selection") || "",
      realism: response.headers.get("X-Retake-Realism") === "on",
      texture: response.headers.get("X-Retake-Texture-Source") || "average",
      detailScore,
      referenceMedian: parseSimilarity(response.headers.get("X-Retake-Identity-Photo-Level"))
        ?? parseSimilarity(response.headers.get("X-Retake-Identity-Median-Similarity")),
      identitySufficient: response.headers.get("X-Retake-Identity-Sufficient") || "",
      selfFidelity: response.headers.get("X-Retake-Self-Fidelity") || "",
      ultraSchedule: response.headers.get("X-Retake-Ultra-Schedule") || "",
      calibrationSamples: Number.parseInt(response.headers.get("X-Retake-Calibration-Samples") || "", 10),
      ultraDepth: response.headers.get("X-Retake-Ultra-Depth") || "",
      fineFeatures: response.headers.get("X-Retake-Fine-Features") || "",
      ultraTiming: response.headers.get("X-Retake-Ultra-Timing") || "",
      // Measurements only (no image, name or path) for the copied report.
      diagnostics: collectDiagnosticHeaders(response.headers),
      fineDetail: response.headers.get("X-Retake-Fine-Detail") || "",
      catchlights: response.headers.get("X-Retake-Catchlights") || "",
      sclera: response.headers.get("X-Retake-Sclera") || "",
      ultraRefinements: response.headers.get("X-Retake-Ultra-Refinements") || "",
      realismFinalists: Number.parseInt(response.headers.get("X-Retake-Ultra-Realism-Finalists") || "", 10),
      naturalness: parseNaturalness(
        response.headers.get("X-Retake-Naturalness"), response.headers.get("X-Retake-Naturalness-Checks")
      ),
      identityModel: /^[a-z0-9_]{3,40}$/.test(response.headers.get("X-Retake-Identity-Model") || "")
        ? response.headers.get("X-Retake-Identity-Model") : ""
    };
    showProfileHealth(
      response.headers.get("X-Retake-Identity-Median-Similarity"),
      response.headers.get("X-Retake-Identity-Outliers")
    );
    const appliedVariation = Number(response.headers.get("X-Retake-Variation"));
    const variationIndex = Number.isInteger(appliedVariation) &&
      appliedVariation >= 0 && appliedVariation < FACE_VARIATION_LABELS.length
      ? appliedVariation
      : variation;
    const generatedBlob = await response.blob();
    if (!generatedBlob.size || !generatedBlob.type.startsWith("image/")) {
      throw new Error("FaceFusion geçerli bir sonuç görseli döndürmedi.");
    }
    const generated = await loadImage(generatedBlob);
    generatedUrl = generated.url;
    const delivered = await applyGeneratedFacePatch(
      generated.image, crop, response.headers.get("X-Retake-Compositing") === "preblended",
      response.headers.get("X-Retake-Alpha-Transport") === "binary", targetSize
    );
    state.nextVariation = 0;
    const ultraLabel = (scores.ultraDepth && scores.ultraDepth !== "standard" ? `Ultra ${depthLabel(scores.ultraDepth)}` : "Ultra-420") + " · en iyi sonuç" + (
      Number.isInteger(completedRounds) && completedRounds >= 1
        ? ` (${completedRounds}${Number.isInteger(attemptedRounds) && attemptedRounds >= completedRounds ? "/" + attemptedRounds : ""} doğrulanmış tur)` : ""
    );
    if (delivered.nativeFinish && scores.diagnostics) {
      const finish = delivered.nativeFinish;
      scores.diagnostics["browser-native-finish"] =
        `scale=${finish.scale};sharpen=${finish.sharpen};grain=${finish.resultGrain}>${finish.sourceGrain};added=${finish.addedGrain}`;
    }
    if (scores.diagnostics) {
      if (delivered.portraitGeometry) scores.diagnostics['browser-portrait-geometry'] = describePortraitGeometry(delivered.portraitGeometry);
      if (delivered.sourcePortraitQuality) scores.diagnostics['browser-source-quality'] = describeSourcePortraitQuality(delivered.sourcePortraitQuality);
    }
    const qualityComparison = createQualityBaseline(compareContext,scores.diagnostics);
    addResult(delivered.blob, ultraLabel, {...scores,qualityComparison,faceEdit:delivered.faceEdit,nativeFinish:delivered.nativeFinish,portraitGeometry:delivered.portraitGeometry,sourcePortraitQuality:delivered.sourcePortraitQuality});
    elements.disabledNote.textContent = "Ultra Mod her işlemde yeni adayları karşılaştırır; yalnızca en iyi doğrulanmış fotoğraf gösterilir.";
    const scoreText = [
      describeSimilarity(scores.similarity, scores.before, scores.referenceMedian),
      describeNaturalness(scores.naturalness)
    ].filter(Boolean).join(" · ");
    showToast(
      "Ultra sonuç hazır" +
      (Number.isInteger(winnerRound) && winnerRound > 0 ? ` · seçilen tur ${winnerRound}` : "") +
      (scoreText ? ` · ${scoreText}` : "") + "."
    );
    setRuntimeStatus("Yüz düzenleme tamamlandı · sonuç hazır");
    return { ...scores, winnerRound: Number.isInteger(winnerRound) ? winnerRound : null };
  } catch (error) {
    // The benchmark records the failure in its table and moves on.
    if (benchmark) throw error;
    state.profileIsBuilding = false;
    if (!state.identityProfile?.validCount) {
      updateProfileIndicator({
        title: "Profil hazırlanamadı",
        detail: error?.message || "Yerel profil oluşturma adımı başarısız.",
        progress: 0,
        status: "error"
      });
    }
    console.error("Yerel FaceFusion işlemi başarısız:", error);
    if (state.result) {
      elements.outputTitle.textContent = "Önceki sonuç korunuyor";
      elements.outputStatus.textContent = "Yeniden denenebilir";
    } else {
      elements.outputTitle.textContent = "İşlem tamamlanamadı";
      elements.outputStatus.textContent = "Hata";
      elements.outputPlaceholder.hidden = false;
      elements.outputPlaceholder.querySelector("strong").textContent = "Yüz düzenleme başarısız";
      elements.outputMessage.textContent = error.message || "FaceFusion isteği başarısız oldu.";
      elements.outputStage.classList.remove("has-result");
      elements.outputSymbol.hidden = false;
    }
    showToast(error.message || "FaceFusion yüz değiştirme başarısız.");
    setRuntimeStatus("FaceFusion · hata");
  } finally {
    if (generatedUrl) URL.revokeObjectURL(generatedUrl);
    endLongJob();
    hideWaitStage();
    state.isSwapping = false;
    updateSwapAvailability();
  }
}

// ---- Self-test benchmark ---------------------------------------------
function updateBenchmarkAvailability(referenceCount) {
  if (!elements.benchmarkStart) return;
  const running = Boolean(state.benchmark?.running);
  const supported = Boolean(state.faceFusionStatus.configured && state.faceFusionStatus.selfTestBenchmark);
  elements.benchmarkStart.disabled = running || !supported || !state.detector || state.isSwapping ||
    state.referencesLoading || referenceCount !== MAX_REFERENCE_FACES;
  elements.benchmarkStop.hidden = !running;
  if (!running && elements.benchmarkStatus && !state.benchmark?.rows) {
    elements.benchmarkStatus.textContent = !state.faceFusionStatus.configured
      ? "Önce bilgisayarındaki motoru başlat."
      : !supported
        ? "Bu motor sürümü öz-testi desteklemiyor: git pull ve update_retake_ultra.ps1 ile güncelle."
        : referenceCount !== MAX_REFERENCE_FACES || state.referencesLoading || !state.detector
          ? "25 referans ve yüz algılama hazır olunca başlatılabilir."
          : "Hazır. Görseller yalnız bu cihaz ile yerel motor arasında kalır.";
  }
}

function formatNumber(value, digits) {
  return Number.isFinite(value) ? value.toFixed(digits).replace(".", ",") : "—";
}

function renderBenchmarkRows(rows) {
  elements.benchmarkResults.hidden = rows.length === 0;
  elements.benchmarkRows.replaceChildren(...rows.map((row) => {
    const tr = document.createElement("tr");
    const cell = (text, className) => {
      const td = document.createElement("td");
      td.textContent = text;
      if (className) td.className = className;
      return td;
    };
    tr.append(cell(`${row.index + 1}. ${row.file || ""}`));
    if (row.error) {
      const td = cell(row.error, "benchmark-error");
      td.colSpan = 5;
      tr.append(td);
      return tr;
    }
    const score = fidelityScore(row.ssim, row.deltaE);
    tr.append(
      cell(formatNumber(row.ssim, 3)), cell(formatNumber(row.deltaE, 1)),
      cell(score === null ? "—" : formatNumber(score, 3)),
      cell(Number.isFinite(row.similarity) ? formatNumber(row.similarity * 100, 1) : "—"),
      cell(row.schedule === "learned-fidelity" ? "öğrenilmiş" : row.schedule === "learned-proxy" ? "öğreniyor" : "varsayılan")
    );
    return tr;
  }));
}

async function runSelfTestBenchmark() {
  if (state.isSwapping || state.benchmark?.running) return;
  if (!state.faceFusionStatus.configured || !state.faceFusionStatus.selfTestBenchmark) {
    showToast("Öz-test için motorun güncel sürümü çalışıyor olmalı.");
    return;
  }
  const items = pickBenchmarkReferences(state.references, Number(elements.benchmarkCount.value));
  if (items.length === 0) {
    showToast("Kıyaslama için hazır referans fotoğrafı yok.");
    return;
  }
  const judgeNotice = isNvidiaJudgeEnabled()
    ? " NVIDIA görsel jürisi açık olduğu için her testte en fazla 4 sonuç adayının yüz kırpımı Hafize üzerinden NVIDIA'ya da gönderilir."
    : "";
  const consent = await showAppDialog({
    title: "Öz-test kıyaslaması başlatılsın mı?",
    message: `${items.length} referans fotoğrafın sırayla hedef yapılıp her biri için Ultra çalıştırılır (her biri en az 7 dakika). Her testte o fotoğraf kimlik kaynaklarından tamamen çıkarılır; sonuç orijinal yüze karşı ölçülür ve motor hangi ayarların en doğal göründüğünü bu bilgisayarda öğrenir (yalnız sayılar saklanır). Görseller bu cihaz ile ${state.engine.remote ? "eşleştirdiğin bilgisayardaki" : "bu bilgisayardaki"} motor arasında kalır.` + judgeNotice,
    confirmLabel: "Kıyaslamayı başlat"
  });
  if (!consent) return;
  state.benchmark = { running: true, stop: false, rows: [] };
  updateSwapAvailability();
  renderBenchmarkRows([]);
  elements.benchmarkDownload.hidden = true;
  elements.benchmarkSummary.textContent = "";
  try {
    const rows = await runBenchmark({
      items,
      shouldStop: () => Boolean(state.benchmark?.stop),
      onProgress: ({ position, total, file, rows: done }) => {
        renderBenchmarkRows(done);
        elements.benchmarkStatus.textContent = file
          ? `Test ${position + 1}/${total}: ${file} · Ultra çalışıyor…`
          : "Kıyaslama tamamlandı.";
        elements.benchmarkSummaryNote.textContent = file ? `${position + 1}/${total}` : "Tamamlandı";
      },
      runOne: async (item, index) => {
        const file = new File([item.blob], item.file || `referans-${index + 1}.jpg`, { type: item.blob.type || "image/jpeg" });
        await setTargetPhoto(file);
        if (!state.target || state.target.file !== file) throw new Error("Fotoğraf hedef olarak yüklenemedi.");
        if (state.target.faces.length !== 1) throw new Error(`${state.target.faces.length} yüz algılandı; öz-test için tek yüz gerekir.`);
        const result = await runFaceFusionEdit({ benchmark: true });
        if (!result) throw new Error("İşlem başlatılamadı.");
        return result;
      }
    });
    const summary = summarizeBenchmark(rows);
    state.benchmark = { running: false, rows, report: buildBenchmarkReport(rows, { engineModel: "retake_hybrid_25_reference_ultra_v22" }) };
    renderBenchmarkRows(rows);
    elements.benchmarkSummary.textContent = summary.measured
      ? `Ortalama: SSIM ${formatNumber(summary.meanSsim, 3)} · ΔE ${formatNumber(summary.meanDeltaE, 1)} · doğallık ${formatNumber(summary.meanFidelity, 3)} (${summary.measured} ölçüm${summary.failed ? `, ${summary.failed} hata` : ""}). Aynı fotoğrafın JPEG yeniden kaydı ≈ 0,97 / 1,3 / 0,94.`
      : "Ölçüm alınamadı; tablodaki hata nedenlerine bak.";
    elements.benchmarkDownload.hidden = rows.length === 0;
    elements.benchmarkStatus.textContent = state.benchmark.rows.length < items.length
      ? "Kıyaslama durduruldu. Toplanan ölçümler motorun öğrenmesine eklendi."
      : "Kıyaslama tamamlandı. Motor sonraki işlemlerde en doğal ölçülen ayarları önce dener.";
  } catch (error) {
    state.benchmark = { running: false, rows: state.benchmark?.rows || [] };
    elements.benchmarkStatus.textContent = "Kıyaslama durdu: " + (error?.message || "bilinmeyen hata");
  } finally {
    if (state.benchmark) state.benchmark.running = false;
    updateSwapAvailability();
  }
}

function downloadBenchmarkReport() {
  const report = state.benchmark?.report;
  if (!report) return;
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `retake-oz-test-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.json`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// One text with this result's measurements and the engine's state, to paste
// into a chat. Clipboard first; a .txt download when the clipboard refuses.
async function copyDiagnostics() {
  const item = state.result;
  if (!item?.diagnostics) return;
  let health = null;
  try {
    const response = await engineFetch("/health", { headers: { Accept: "application/json" }, signal: AbortSignal.timeout?.(4000) });
    if (response.ok) health = await response.json();
  } catch { /* the engine may be off: the result's own headers still help */ }
  const report = buildDiagnosticReport({
    headers: item.diagnostics, health, depth: item.diagnostics["x-retake-ultra-depth"] || "", appVersion: WEB_QUALITY_REVISION
  });
  const comparison=compareQualityBaseline(getQualityBaseline(),item.qualityComparison);
  const reportWithComparison=comparison?report+"\n\n"+comparison:report;
  try {
    await navigator.clipboard.writeText(reportWithComparison);
    showToast("Tanı raporu kopyalandı — sohbete yapıştırabilirsin (görsel içermez).");
  } catch {
    const url = URL.createObjectURL(new Blob([reportWithComparison], { type: "text/plain;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "retake-tani-raporu.txt";
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
    showToast("Pano kullanılamadı; tanı raporu .txt olarak indirildi.");
  }
}

async function downloadResult() {
  if (!state.result || !state.target) return;
  try {
    const file = await buildExportFile();
    const url = URL.createObjectURL(file);
    const link = document.createElement("a");
    link.href = url;
    link.download = file.name;
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
  } catch (error) {
    showToast(error.message || "Sonuç kaydedilemedi.");
  }
}

async function shareResult() {
  if (!state.result || !state.target) return;
  try {
    const file = await buildExportFile();
    if (!navigator.canShare?.({ files: [file] })) {
      showToast("Bu cihaz dosya paylaşımını desteklemiyor; Sonucu indir düğmesini kullan.");
      return;
    }
    await navigator.share({ files: [file], title: "Retake" });
  } catch (error) {
    if (error?.name !== "AbortError") showToast(error.message || "Paylaşım başlatılamadı.");
  }
}

function applyEngineSettings(settings) {
  state.engine = { origin: settings.origin, remote: settings.remote, token: settings.token };
  elements.engineAddress.value = settings.remote ? settings.origin : "";
  elements.engineToken.value = settings.token;
}

async function saveEngineForm(event) {
  event.preventDefault();
  if (state.isSwapping) return;
  const settings = resolveEngineSettings(elements.engineAddress.value, elements.engineToken.value);
  if (!settings.ok) {
    elements.engineFormStatus.textContent = settings.reason;
    return;
  }
  applyEngineSettings(settings);
  const stored = saveEngineSettings(window.localStorage, settings);
  elements.engineFormStatus.textContent = "Bağlanılıyor…";
  await checkFaceFusionStatus();
  elements.engineFormStatus.textContent = state.faceFusionStatus.configured
    ? "Motor hazır." + (stored ? "" : " Ayar bu oturum için geçerli; tarayıcı kalıcı saklamaya izin vermedi.")
    : state.faceFusionStatus.message;
}

async function resetEngineForm() {
  if (state.isSwapping) return;
  elements.engineAddress.value = "";
  elements.engineToken.value = "";
  const settings = resolveEngineSettings("", "");
  applyEngineSettings(settings);
  saveEngineSettings(window.localStorage, settings);
  await checkFaceFusionStatus();
  elements.engineFormStatus.textContent = "Bu bilgisayardaki motor kullanılacak.";
}

function initializeEngineSettings() {
  let storage = null;
  try {
    storage = window.localStorage;
  } catch {
    // Private mode: pairing still works for this session.
  }
  applyEngineSettings(loadEngineSettings(storage));
  elements.engineForm.addEventListener("submit", saveEngineForm);
  elements.engineReset.addEventListener("click", resetEngineForm);
  elements.compareRange.addEventListener("input", updateResultView);
  elements.strengthRange.addEventListener("input", updateResultView);
  restoreRealFacePreference();
  restoreNvidiaJudgePreference();
  restoreQualityDepthPreference();
  elements.shareButton.addEventListener("click", shareResult);
  elements.diagnosticsButton?.addEventListener("click", copyDiagnostics);
  elements.baselineSave?.addEventListener("click", saveQualityBaseline);
  elements.baselineClear?.addEventListener("click", clearQualityBaseline);
  elements.shareButton.hidden = typeof navigator.share !== "function" || typeof navigator.canShare !== "function";
  // When the app comes back to the foreground (phone unlocked, tab restored)
  // re-check the engine instead of showing a stale status.
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && !state.isSwapping) checkFaceFusionStatus();
  });
}

async function initializeDetector() {
  setRuntimeStatus("Yüz algılama modeli yükleniyor…");
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/wasm"
    );
    state.vision = vision;
    const options = {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "IMAGE",
      numFaces: 20,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5
    };

    try {
      state.detector = await FaceLandmarker.createFromOptions(vision, options);
    } catch (gpuError) {
      console.info("GPU yüz algılama kullanılamadı; CPU deneniyor.", gpuError);
      options.baseOptions.delegate = "CPU";
      state.detector = await FaceLandmarker.createFromOptions(vision, options);
    }

    if (state.target) {
      state.target.faces = await detectFaces(state.target.image);
      state.selectedFaceIndex = state.target.faces.length > 0 ? 0 : null;
      clearResult();
      elements.targetMeta.textContent =
        state.target.image.naturalWidth + " × " + state.target.image.naturalHeight + " px · " +
        formatBytes(state.target.file.size) + " · " +
        (state.target.faces.length === 0 ? "Fotoğrafta yüz bulunamadı." : state.target.faces.length + " yüz algılandı.");
      drawTargetPreview();
      renderFaceChips();
    }
    setRuntimeStatus("Yüz algılama hazır · Yerel yüz tespiti");
    renderReferences();
    updateSwapAvailability();
  } catch (error) {
    console.error("MediaPipe başlatılamadı:", error);
    elements.engineTitle.textContent = "Yüz algılama motoru yüklenemedi";
    elements.engineDescription.textContent =
      "Yüz algılama modeli yüklenemedi. İnternet bağlantısını kontrol edip sayfayı yenile.";
    setRuntimeStatus("Yüz algılama başlatılamadı");
    updateSwapAvailability();
    showToast("Yüz algılama modeli yüklenemedi. İnternet bağlantısını kontrol et.");
  }
}

function initializeTheme() {
  let theme = "dark";
  try {
    const saved = localStorage.getItem("retake-theme");
    if (saved === "light" || saved === "dark") theme = saved;
    else if (window.matchMedia?.("(prefers-color-scheme: light)").matches) theme = "light";
  } catch {
    // Theme persistence is optional.
  }
  document.documentElement.dataset.theme = theme;
  updateThemeButton();
}

function updateThemeButton() {
  const light = document.documentElement.dataset.theme === "light";
  elements.themeButton.setAttribute("aria-label", light ? "Koyu temaya geç" : "Açık temaya geç");
  elements.themeButton.title = light ? "Koyu tema" : "Açık tema";
  const themeColor = document.querySelector('meta[name="theme-color"]');
  if (themeColor) themeColor.content = light ? "#ebe8e2" : "#0e0d0c";
}

function toggleTheme() {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem("retake-theme", next);
  } catch {
    // Theme still changes for this session.
  }
  updateThemeButton();
  stones.hero?.refreshTheme();
  stones.wait?.refreshTheme();
}

function initializeInteractions() {
  elements.targetInput.addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (file) await setTargetPhoto(file);
    event.target.value = "";
  });

  elements.targetDropzone.addEventListener("click", (event) => {
    if (state.isSwapping || state.targetLoading) return;
    if (event.target === elements.targetCanvas || event.target.closest(".preview-hint")) return;
    elements.targetInput.click();
  });
  elements.targetDropzone.addEventListener("keydown", (event) => {
    if (state.isSwapping || state.targetLoading) return;
    if (event.target !== elements.targetDropzone) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      elements.targetInput.click();
    }
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    elements.targetDropzone.addEventListener(eventName, (event) => {
      if (state.isSwapping || state.targetLoading) return;
      event.preventDefault();
      elements.targetDropzone.classList.add("drag-over");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    elements.targetDropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.targetDropzone.classList.remove("drag-over");
    });
  });
  elements.targetDropzone.addEventListener("drop", async (event) => {
    if (state.isSwapping || state.targetLoading) return;
    const file = event.dataTransfer?.files?.[0];
    if (file) await setTargetPhoto(file);
  });

  elements.targetCanvas.addEventListener("click", (event) => {
    if (!state.target || state.target.faces.length === 0) return;
    const rect = elements.targetCanvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    const candidates = state.target.faces
      .map((face, index) => ({ face, index }))
      .filter(({ face }) => x >= face.x && x <= face.x + face.width && y >= face.y && y <= face.y + face.height)
      .sort((a, b) => (a.face.width * a.face.height) - (b.face.width * b.face.height));
    if (candidates[0]) selectTargetFace(candidates[0].index);
  });

  elements.clearTarget.addEventListener("click", clearTargetPhoto);
  elements.reloadReferences.addEventListener("click", () => loadLocalReferences({ force: true }));
  elements.clearReferences.addEventListener("click", clearReferenceData);
  elements.chooseReferences.addEventListener("click", () => elements.referencesInput.click());
  elements.referencesInput.addEventListener("change", async () => {
    await loadUserReferences(Array.from(elements.referencesInput.files || []));
    elements.referencesInput.value = "";
  });
  elements.swapButton.addEventListener("click", runFaceFusionEdit);
  elements.benchmarkStart?.addEventListener("click", runSelfTestBenchmark);
  elements.benchmarkStop?.addEventListener("click", () => {
    if (!state.benchmark?.running) return;
    state.benchmark.stop = true;
    elements.benchmarkStatus.textContent = "Çalışan test bitince kıyaslama duracak.";
  });
  elements.benchmarkDownload?.addEventListener("click", downloadBenchmarkReport);
  elements.downloadButton.addEventListener("click", downloadResult);
  elements.themeButton.addEventListener("click", toggleTheme);
  elements.appDialog.addEventListener("close", () => {
    const resolve = state.dialogResolve;
    state.dialogResolve = null;
    resolve?.(elements.appDialog.returnValue === "confirm");
  });

  window.addEventListener("beforeinstallprompt", (event) => {
    event.preventDefault();
    state.installPrompt = event;
    elements.installButton.hidden = false;
  });
  elements.installButton.addEventListener("click", async () => {
    if (!state.installPrompt) return;
    state.installPrompt.prompt();
    await state.installPrompt.userChoice;
    state.installPrompt = null;
    elements.installButton.hidden = true;
  });

  window.addEventListener("pagehide", (event) => {
    if (event.persisted) return;
    if (state.target?.url) URL.revokeObjectURL(state.target.url);
    state.results.forEach((item) => URL.revokeObjectURL(item.url));
    state.references.forEach((item) => {
      if (item.url) URL.revokeObjectURL(item.url);
    });
    state.detector?.close();
    state.referenceDetector?.close();
    state.referenceDetectorFull?.close();
  });
}

const HEAVY_INIT_DELAY_MS = 2600;

function whenPageSettled(delayMs) {
  return new Promise((resolve) => {
    let done = false;
    const go = () => {
      if (done) return;
      done = true;
      window.clearTimeout(timer);
      document.removeEventListener("pointerdown", onIntent, true);
      document.removeEventListener("keydown", onIntent, true);
      document.removeEventListener("dragenter", onIntent, true);
      resolve();
    };
    const onIntent = (event) => {
      // Playing with the sculpture is not a reason to start loading.
      if (event.target?.closest?.(".sculpture-host")) return;
      go();
    };
    const timer = window.setTimeout(() => {
      if ("requestIdleCallback" in window) window.requestIdleCallback(go, { timeout: 1500 });
      else go();
    }, delayMs);
    document.addEventListener("pointerdown", onIntent, true);
    document.addEventListener("keydown", onIntent, true);
    document.addEventListener("dragenter", onIntent, true);
  });
}

async function initializeApp() {
  initializeTheme();
  initializeInteractions();
  initializeStones();
  initializeEngineSettings();
  initializeHoldToCompare();

  // The sculpture's opening animation gets the first seconds to itself; the
  // heavy work (face model, 25 reference photos) starts once the page is
  // idle, or straight away when the user reaches for a photo.
  state.referencesLoading = true;
  updateSwapAvailability();
  const engineCheck = checkFaceFusionStatus();
  await whenPageSettled(HEAVY_INIT_DELAY_MS);
  await Promise.allSettled([
    // Skip the default pool if the user already started loading their own.
    state.referenceLoadToken === 0 ? loadLocalReferences() : Promise.resolve(0),
    initializeDetector(),
    engineCheck
  ]);

  const loaded = state.references.filter((reference) => reference.status === "ready").length;
  setRuntimeStatus(
    (state.detector ? "Yüz algılama hazır" : "Yüz algılama kullanılamıyor") +
    " · Yerel referans: " + loaded + " / " + MAX_REFERENCE_FACES
  );
  updateSwapAvailability();

  if ("serviceWorker" in navigator && window.isSecureContext) {
    navigator.serviceWorker.register("./sw.js").catch((error) => {
      console.info("Service worker kurulamadı; uygulama çevrimiçi kullanılabilir.", error);
    });
  }
}

initializeApp();

function getQualityBaseline() {
  try { return readQualityBaseline(window.localStorage); } catch { return null; }
}

function updateQualityBaseline() {
  if (!elements.baselineSave || !elements.baselineClear || !elements.baselineStatus) return;
  const baseline=getQualityBaseline(), current=state.result?.qualityComparison;
  elements.baselineSave.disabled=!current||state.isSwapping;
  elements.baselineClear.disabled=!baseline||state.isSwapping;
  elements.baselineStatus.textContent=baseline
    ? (current?compareQualityBaseline(baseline,current):"Yerel ölçüm kaydı var; aynı fotoğraf ve ayarlarla yeni sonuç üret.")
    : "İstersen bu sonucu yerel kıyas ölçütü olarak kaydet. Görsel kaydedilmez.";
}

function saveQualityBaseline() {
  const current=state.result?.qualityComparison;
  if(!current||state.isSwapping)return;
  try {
    window.localStorage.setItem(QUALITY_BASELINE_KEY,JSON.stringify(current));
    showToast("Bu sonucun ölçümleri bu tarayıcıya kıyas ölçütü olarak kaydedildi.");
  } catch { showToast("Tarayıcı yerel ölçüm kaydına izin vermedi."); }
  updateQualityBaseline();
}

function clearQualityBaseline() {
  if(state.isSwapping)return;
  try {
    window.localStorage.removeItem(QUALITY_BASELINE_KEY);
    showToast("Yerel kıyas ölçütü silindi.");
  } catch { showToast("Yerel kıyas ölçütü silinemedi."); }
  updateQualityBaseline();
}
