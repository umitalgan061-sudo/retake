import { FaceDetector, FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/vision_bundle.mjs";
import {
  formatBytes,
  getLandmarkBounds,
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
import { alphaMaskFromRGBA, compositeMaskedRGBA, inspectOutsideMaskIntegrity } from "./src/face-only.js";
import { calculateReferenceWeight, getSimilarityTransform, weightedAverageRGBA } from "./src/identity-profile.js";
import {
  estimateHeadFit,
  getTargetRenderSize,
  mapLandmarksFromRegion,
  measureFaceShapeRatio,
  rectIoU,
  summarizeReferenceShape
} from "./src/head-geometry.js";
import { mountSculpture, placeWideSculpture } from "./src/forma-sculpture.js";
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
  downloadButton: $("#downloadButton"),
  resultImage: $("#resultImage"),
  resultView: $("#resultView"),
  originalImage: $("#originalImage"),
  resultTools: $("#resultTools"),
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
    !state.targetLoading && !state.referencesLoading && !state.isSwapping;
  elements.swapButton.disabled = !enabled;
  elements.downloadButton.disabled = !state.result || state.isSwapping;
  elements.shareButton.disabled = !state.result || state.isSwapping;
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
          ? "Düzenleme için bu cihazda yerel FaceFusion servisini başlat."
          : !hasTarget
            ? "Başlamak için bir fotoğraf yükle ve değiştirilecek yüzü seç."
            : "Saç stili kaynak fotoğraftan korunur. Yeniden oluşturma farklı kafa oranı ve sakal harmanları dener.";
  }
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
      state.faceFusionStatus = {
        checked: true,
        configured: true,
        model: health.model || "uniface_256",
        pixelBoost: health.pixel_boost || "768x768",
        message: "FaceFusion 3.9.2 ve CUDA hazır."
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
    connectionLabel.textContent = state.engine.remote
      ? (state.faceFusionStatus.configured ? "EŞLEŞEN MOTOR HAZIR" : "EŞLEŞEN MOTOR ÇEVRİMDIŞI")
      : (state.faceFusionStatus.configured ? "YEREL MOTOR HAZIR" : "YEREL MOTOR ÇEVRİMDIŞI");
  }
  elements.engineSummary.textContent = state.engine.remote
    ? (state.faceFusionStatus.configured ? "Eşleşti" : "Bağlanamadı")
    : "Bu bilgisayar";
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
  return { ...bounds, shapeRatio: measureFaceShapeRatio(landmarks, sourceWidth, sourceHeight) };
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
    .filter((face) => face && rectIoU(face, box) > 0.2)
    .sort((a, b) => rectIoU(b, box) - rectIoU(a, box))[0] || null;
}

async function detectFaces(image) {
  if (!state.detector) throw new Error("Yüz algılama motoru henüz hazır değil.");
  const sourceWidth = image.naturalWidth || image.width;
  const sourceHeight = image.naturalHeight || image.height;
  const faces = (state.detector.detect(image).faceLandmarks || [])
    .map((landmarks) => faceFromLandmarks(landmarks, sourceWidth, sourceHeight))
    .filter(Boolean);

  // Second pass: small, distant, turned or partly covered faces.
  for (const range of ["full", "short"]) {
    try {
      const detector = await getReferenceFaceDetector(range);
      const candidates = (detector.detect(image).detections || [])
        .map((item) => normalizeFallbackFace(item, sourceWidth, sourceHeight))
        .filter((item) => item && item.confidence >= 0.5);
      for (const candidate of candidates) {
        if (faces.some((face) => rectIoU(face, candidate.bounds) > 0.25)) continue;
        const refined = refineFaceInRegion(image, candidate.bounds, sourceWidth, sourceHeight);
        if (refined && faces.some((face) => rectIoU(face, refined) > 0.25)) continue;
        faces.push(refined || { ...candidate.bounds, shapeRatio: null });
      }
    } catch (error) {
      console.info(`Ek yüz taraması (${range}) kullanılamadı.`, error);
    }
  }
  return faces.sort((a, b) => (b.width * b.height) - (a.width * a.height));
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
    caption.textContent = String(index + 1);
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
  updateResultView();
  renderResultHistory();
  updateSwapAvailability();
}

function addResult(blob, label) {
  const item = { blob, url: createObjectURL(blob), label };
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
async function buildExportFile() {
  const format = elements.exportFormat.value === "jpeg" ? "jpeg" : "png";
  const strength = resultStrength();
  let blob = state.result.blob;
  if (format === "jpeg" || strength < 1) {
    const [original, result] = await Promise.all([loadImage(state.target.file), loadImage(state.result.blob)]);
    try {
      const canvas = document.createElement("canvas");
      canvas.width = original.image.naturalWidth;
      canvas.height = original.image.naturalHeight;
      const context = canvas.getContext("2d", { alpha: false });
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
    state.target = { file, image: loaded.image, url: loaded.url, faces };
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
    showToast(detectionMessage);
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
    const results = await Promise.allSettled(references.map(async (reference, index) => {
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
    }));

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

const PROFILE_PIPELINE_VERSION = "retake-25-reference-headfit-v5";
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

async function buildLocalFaceFusionInputs() {
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
        shapeRatio: Number.isFinite(cached.shapeRatio) ? cached.shapeRatio : null
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

  return {
    targetCanvas,
    referenceBlobs: profile.referenceBlobs,
    referenceWeights: profile.weights,
    appearanceProfileBlob: profile.appearanceProfileBlob,
    profileCount: profile.validCount,
    totalProfileWeight: profile.totalWeight,
    faceBounds,
    targetSize: { width: targetCanvas.width, height: targetCanvas.height },
    headFit,
    crop
  };
}

function applyGeneratedFacePatch(generatedTargetImage, crop) {
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
  generatedContext.drawImage(
    generatedTargetImage,
    0, 0, outputWidth, outputHeight,
    0, 0, crop.width, crop.height
  );
  const generatedCrop = generatedContext.getImageData(0, 0, crop.width, crop.height);
  const parserAlphaMask = alphaMaskFromRGBA(generatedCrop.data, crop.width, crop.height);
  const hasTransparentPixels = parserAlphaMask.some((alpha) => alpha < 0.999);
  // New local FaceFusion server embeds its parser/occlusion-derived change matte
  // in PNG alpha. Retain the ellipse only as a backward-compatible fallback.
  const alphaMask = hasTransparentPixels
    ? parserAlphaMask
    : createFaceAlphaMask(crop.width, crop.height, crop.face);
  const blended = compositeMaskedRGBA(
    sourceCrop.data,
    generatedCrop.data,
    alphaMask,
    crop.width,
    crop.height
  );

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

  fullContext.putImageData(output, 0, 0);
  return canvasToBlob(fullCanvas);
}

const stones = { hero: null, wait: null, timer: null };

function initializeStones() {
  const wide = window.matchMedia("(min-width: 851px)");
  const copy = $(".hero-copy");
  stones.hero = mountSculpture($("#heroSculpture"), {
    scale: 0.98,
    // Desktop: full-width sculpture above the headline, like the Forma page.
    place: (width, height, aspect) => {
      if (!wide.matches) return null;
      const host = $("#heroSculpture").getBoundingClientRect();
      const textTop = (copy?.getBoundingClientRect().top ?? host.top + height * 0.7) - host.top;
      // Keep clear of the sticky top bar (the host starts under it).
      return placeWideSculpture(width, height, aspect, Math.max(height * 0.45, textTop - 12), 64);
    }
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
const JOB_TIMEOUT_MS = 20 * 60 * 1000;

// The engine works on the swap as a background job and reports its real stage,
// so a long run never depends on a single open connection.
async function waitForEngineJob(jobId, total) {
  const deadline = Date.now() + JOB_TIMEOUT_MS;
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
    setWaitProgress(job.step, job.total || total, `${Math.min(job.step + 1, job.total || total)} / ${job.total || total} · ${job.stage}`);
    if (job.status === "done") return;
  }
  throw new Error("İşlem 20 dakika içinde tamamlanmadı.");
}

async function runFaceFusionEdit() {
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

  const consent = await showAppDialog({
    title: state.result ? "Yeni bir görünüm oluştur?" : "Yerel düzenlemeyi başlat?",
    message: state.engine.remote
      ? "Bu işlemde seçili fotoğrafın yüz kırpımı ve 25 referanstan hazırlanan yüz profili, şifreli tünel sağlayıcısı (Cloudflare veya Tailscale) üzerinden eşleştirdiğin kendi bilgisayarındaki FaceFusion motoruna gönderilir. Görseller bir AI servisine yüklenmez; işlem senin bilgisayarında yapılır."
      : "Bu işlemde seçili fotoğrafın yüz kırpımı ve 25 referanstan hazırlanan yüz profili bu cihazdaki FaceFusion servisine gönderilir. Görseller harici bir AI servisine yüklenmez.",
    confirmLabel: state.engine.remote ? "Bilgisayarıma gönder" : "Yerel olarak devam et"
  });
  if (!consent) return;

  state.isSwapping = true;
  updateSwapAvailability();
  const variation = state.nextVariation;
  elements.outputTitle.textContent = state.result ? "Yeni görünüm hazırlanıyor" : "Yüz düzenleniyor";
  elements.outputStatus.textContent = "Bilgisayarda çalışıyor";
  if (!state.result) {
    elements.outputPlaceholder.hidden = false;
    elements.outputPlaceholder.querySelector("strong").textContent = "Yüz profili hazırlanıyor…";
    elements.outputMessage.textContent = "İlk kullanımda profil hazırlanması biraz zaman alabilir.";
  }

  showWaitStage(state.result ? "Yeni görünüm hazırlanıyor" : "Yüzün bilgisayarında işleniyor");

  let generatedUrl = null;
  try {
    const {
      targetCanvas,
      referenceBlobs,
      referenceWeights,
      appearanceProfileBlob,
      faceBounds,
      targetSize,
      headFit,
      crop
    } = await buildLocalFaceFusionInputs();
    const targetBlob = await canvasToBlob(targetCanvas);

    const form = new FormData();
    form.append("target", targetBlob, "retake-target.png");
    form.append("appearance_profile", appearanceProfileBlob, "retake-25-photo-appearance-profile.png");
    referenceBlobs.forEach((blob, index) => {
      form.append("reference", blob, `retake-reference-${String(index + 1).padStart(2, "0")}.png`);
    });
    form.append("metadata", JSON.stringify({ referenceWeights, faceBounds, targetSize, headFit, variation }));

    setRuntimeStatus("Yerel yüz düzenleme · sonuç hazırlanıyor");

    setWaitProgress(0, 1, "Yüz kırpımları motora gönderiliyor…");
    let response = await engineFetch("/api/face-swap", { method: "POST", body: form });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error || `FaceFusion işlemi başarısız (HTTP ${response.status}).`);
    }
    if (response.status === 202) {
      const accepted = await response.json();
      await waitForEngineJob(accepted.job_id, accepted.total || 4);
      response = await engineFetch("/api/jobs/" + encodeURIComponent(accepted.job_id) + "/result");
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || `Sonuç alınamadı (HTTP ${response.status}).`);
      }
    }
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
    const resultBlob = await applyGeneratedFacePatch(generated.image, crop);
    state.nextVariation = (variation + 1) % 4;
    addResult(resultBlob, FACE_VARIATION_LABELS[variationIndex]);
    elements.disabledNote.textContent = "Saç stili kaynak fotoğraftan korunur. Yeniden oluşturma farklı kafa oranı ve sakal harmanları dener.";
    showToast(`Sonuç hazır · ${FACE_VARIATION_LABELS[variationIndex]}.`);
    setRuntimeStatus("Yüz düzenleme tamamlandı · sonuç hazır");
  } catch (error) {
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
    hideWaitStage();
    state.isSwapping = false;
    updateSwapAvailability();
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
  elements.shareButton.addEventListener("click", shareResult);
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

    elements.engineDescription.textContent =
      "MediaPipe yüzleri tarayıcıda algılar. Kimlik aktarımı, bilgisayarındaki CUDA destekli FaceFusion modeliyle yapılır.";
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
  if (themeColor) themeColor.content = light ? "#f1f2e9" : "#10120f";
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

async function initializeApp() {
  initializeTheme();
  initializeInteractions();
  initializeStones();
  initializeEngineSettings();

  await Promise.allSettled([
    loadLocalReferences(),
    initializeDetector(),
    checkFaceFusionStatus()
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