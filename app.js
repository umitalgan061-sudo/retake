import { FaceDetector, FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/vision_bundle.mjs";
import {
  formatBytes,
  getLandmarkBounds,
  MAX_IMAGE_BYTES,
  MAX_REFERENCE_FACES,
  validateImageFile
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
import { createReferenceSetKey, getCachedProfile, saveCachedProfile } from "./src/profile-storage.js";

const MAX_PREVIEW_EDGE = 2048;
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const FACE_DETECTOR_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";

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
  reloadReferences: $("#reloadReferencesButton"),
  swapButton: $("#swapButton"),
  downloadButton: $("#downloadButton"),
  resultImage: $("#resultImage"),
  outputPlaceholder: $("#outputPlaceholder"),
  outputMessage: $("#outputMessage"),
  outputStage: $(".output-stage"),
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
  toast: $("#toast")
};

const state = {
  detector: null,
  vision: null,
  referenceDetector: null,
  referenceDetectorPromise: null,
  target: null,
  references: [],
  identityProfile: null,
  referenceSetKey: null,
  referencesLoading: false,
  referenceFaceCropCount: null,
  selectedReferenceCount: 0,
  referenceLoadError: null,
  referenceLoadToken: 0,
  selectedFaceIndex: null,
  faceFusionStatus: { checked: false, configured: false },
  isSwapping: false,
  result: null,
  toastTimer: null,
  installPrompt: null
};

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  window.clearTimeout(state.toastTimer);
  state.toastTimer = window.setTimeout(() => elements.toast.classList.remove("show"), 3200);
}

function setRuntimeStatus(message) {
  elements.runtimeStatus.textContent = message;
}

function localFaceFusionBase() {
  const configured = document.querySelector('meta[name="retake-facefusion-base"]')?.content?.trim();
  const value = configured || "http://127.0.0.1:8765";
  try {
    const parsed = new URL(value);
    return parsed.protocol === "http:" && ["127.0.0.1", "localhost"].includes(parsed.hostname)
      ? parsed.origin.replace(/\/+$/, "")
      : "";
  } catch {
    return "";
  }
}

function updateSwapAvailability() {
  const referenceCount = state.references.filter((item) => item.status === "ready").length;
  const hasTarget = Boolean(state.target && Number.isInteger(state.selectedFaceIndex) &&
    state.target.faces[state.selectedFaceIndex]);
  const enabled = state.faceFusionStatus.configured && state.detector &&
    referenceCount === MAX_REFERENCE_FACES && hasTarget && !state.isSwapping;
  elements.swapButton.disabled = !enabled;
  elements.downloadButton.disabled = !state.result || state.isSwapping;
  const tag = elements.swapButton.querySelector("span");
  if (tag) tag.textContent = state.isSwapping ? "İşleniyor" :
    (enabled ? "Kişisel yüz profili hazır" : (state.faceFusionStatus.configured ? "Fotoğraf seç" : "Bilgisayara bağlan"));
}

async function checkFaceFusionStatus() {
  const base = localFaceFusionBase();
  if (!base) {
    state.faceFusionStatus = { checked: true, configured: false, message: "Yerel FaceFusion adresi geçersiz." };
  } else {
    try {
      const response = await fetch(base + "/health", {
        method: "GET",
        cache: "no-store",
        headers: { Accept: "application/json" }
      });
      const health = await response.json().catch(() => ({}));
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
        message: detail.includes("Failed to fetch") || detail.includes("NetworkError")
          ? "127.0.0.1:8765 bağlantısı kurulamadı. tools\\start_facefusion_retake.ps1 dosyasını çalıştır; tarayıcı yerel ağ izni sorarsa izin ver."
          : detail
      };
    }
  }

  if (state.faceFusionStatus.configured) {
    elements.engineTitle.textContent = "25 fotoğraflı kişisel profil motoru hazır";
    elements.engineDescription.textContent =
      "Tüm referanslar kalite ve açılarına göre hizalanıp ağırlıklı bir görünüm profiline dönüştürülür. AlphaFace kimlik profilini, UniFace ise harmanlanmış görünüm ve sakal ayrıntılarını işler.";
  } else {
    elements.engineTitle.textContent = "Yerel FaceFusion servisi bekleniyor";
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

function detectFaces(image) {
  if (!state.detector) throw new Error("Yüz algılama motoru henüz hazır değil.");
  const result = state.detector.detect(image);
  return (result.faceLandmarks || [])
    .map((landmarks) => getLandmarkBounds(landmarks))
    .filter(Boolean);
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
  if (state.result?.url) URL.revokeObjectURL(state.result.url);
  state.result = null;
  elements.resultImage.hidden = true;
  elements.resultImage.removeAttribute("src");
  elements.outputPlaceholder.hidden = false;
  elements.outputStage.classList.remove("has-result");
  elements.outputSymbol.hidden = false;
  elements.outputTitle.textContent = "Önizleme";
  elements.outputStatus.textContent = "Beklemede";  elements.outputMessage.textContent = "FaceFusion motoru hazır olduğunda sonuç burada görünecek.";
  updateSwapAvailability();
}

function selectTargetFace(index) {
  if (!state.target || !Number.isInteger(index) || !state.target.faces[index]) return;
  state.selectedFaceIndex = index;
  clearResult();
  drawTargetPreview();
  renderFaceChips();
  updateSwapAvailability();
}

async function setTargetPhoto(file) {
  const validation = validateImageFile(file);
  if (!validation.valid) {
    showToast(validation.reason);
    return;
  }
  try {
    const loaded = await loadImage(file);
    let faces = [];
    let detectionMessage = "Yüz algılama yapılamadı.";
    if (state.detector) {
      try {
        faces = detectFaces(loaded.image);
        detectionMessage = faces.length === 0
          ? "Fotoğrafta yüz bulunamadı."
          : faces.length + " yüz algılandı.";
      } catch (error) {
        detectionMessage = error.message;
      }
    } else {
      detectionMessage = "Yüz algılama motoru hazırlanıyor; fotoğraf şimdilik önizleme için açıldı.";
    }

    if (state.target?.url) URL.revokeObjectURL(state.target.url);
    clearResult();
    state.target = { file, image: loaded.image, url: loaded.url, faces };
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
    showToast(error.message || "Fotoğraf yüklenemedi.");
  }
}

function clearTargetPhoto() {
  if (state.target?.url) URL.revokeObjectURL(state.target.url);
  clearResult();
  state.target = null;
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

  if (readyCount === 0) {
    const title = elements.referencesEmpty.querySelector("strong");
    const description = elements.referencesEmpty.querySelector("p");
    title.textContent = state.referencesLoading
      ? "Yerel yüzler yükleniyor"
      : "Yerel referans havuzu okunamadı";
    description.textContent = state.referencesLoading
      ? `${MAX_REFERENCE_FACES} fotoğraf reference-faces/manifest.json listesinden okunuyor.`
      : (state.referenceLoadError || "reference-faces/manifest.json dosyasını kontrol et.");
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
      placeholder.textContent = reference.status === "loading" ? "Yükleniyor…" : "Görsel alınamadı";
      card.append(placeholder);
    }

    const name = document.createElement("span");
    name.className = "reference-name";
    name.textContent = reference.name;

    const label = document.createElement("span");
    label.className = "reference-state";
    label.textContent = reference.status !== "ready"
      ? (reference.status === "loading" ? "YÜKLENİYOR" : "ERİŞİLEMİYOR")
      : reference.profileStatus === "included"
        ? (reference.profileMethod === "FaceDetector" ? "PROFİLDE · YEDEK" : "PROFİLDE")
        : reference.profileStatus === "excluded"
          ? "PROFİLE ALINAMADI"
          : "YERELDE";
    if (reference.profileError) label.title = reference.profileError;

    card.append(name, label);
    elements.referencesGrid.append(card);
  });

  elements.reloadReferences.disabled = state.referencesLoading;
  elements.reloadReferences.textContent = state.referencesLoading ? "Yükleniyor…" : "Klasörden yenile";
  updateSwapAvailability();
}

async function loadLocalReferences({ force = false } = {}) {
  const token = ++state.referenceLoadToken;
  state.identityProfile = null;
  state.referenceSetKey = null;
  state.referencesLoading = true;
  state.referenceLoadError = null;
  state.references.forEach((reference) => {
    if (reference.url) URL.revokeObjectURL(reference.url);
  });
  state.references = [];
  renderReferences();

  try {
    const manifestResponse = await fetch(REFERENCE_MANIFEST_URL, {
      cache: force ? "reload" : "no-store"
    });
    if (!manifestResponse.ok) {
      throw new Error("reference-faces/manifest.json bulunamadı.");
    }

    const references = parseReferenceManifest(await manifestResponse.json());
    if (token !== state.referenceLoadToken) return 0;
    state.references = references.map((reference) => ({ ...reference, status: "loading" }));
    renderReferences();

    const imageBaseUrl = new URL("./reference-faces/", window.location.href).href;
    const results = await Promise.allSettled(references.map(async (reference) => {
      const endpoint = getReferenceImageUrl(reference.file, imageBaseUrl);
      const response = await fetch(endpoint, {
        cache: force ? "reload" : "no-store",
        headers: { Accept: "image/avif,image/webp,image/png,image/jpeg" }
      });
      if (!response.ok) throw new Error(`${reference.file}: HTTP ${response.status}`);

      const blob = await response.blob();
      if (!blob.size || blob.size > MAX_IMAGE_BYTES) {
        throw new Error(`${reference.file}: görsel boş veya 25 MB sınırını aşıyor.`);
      }

      const loaded = await loadImage(blob);
      return { ...reference, status: "ready", url: loaded.url, image: loaded.image, blob, size: blob.size };
    }));

    if (token !== state.referenceLoadToken) {
      results.forEach((result) => {
        if (result.status === "fulfilled") URL.revokeObjectURL(result.value.url);
      });
      return 0;
    }

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
    state.referenceLoadError = failedCount
      ? `${loadedCount}/${MAX_REFERENCE_FACES} görsel yüklendi; ${failedCount} dosyayı manifest ve dosya adıyla karşılaştır.`
      : null;

    renderReferences();
    showToast(`${loadedCount}/${MAX_REFERENCE_FACES} yerel referans yüklendi.`);
    return loadedCount;
  } catch (error) {
    if (token !== state.referenceLoadToken) return 0;
    state.references = [];
    state.referencesLoading = false;
    state.referenceLoadError = error.message || "Yerel referans dosyaları okunamadı.";
    renderReferences();
    showToast(state.referenceLoadError);
    return 0;
  }
}

async function canvasToBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("Görsel hazırlanamadı.")), "image/png");
  });
}

const MIN_FACEFUSION_REFERENCE_COUNT = MAX_REFERENCE_FACES;
const PROFILE_PIPELINE_VERSION = "retake-25-reference-fallback-v3";
const FACEFUSION_REFERENCE_WIDTH = 512;
const FACEFUSION_REFERENCE_HEIGHT = 682;
const FACEFUSION_TARGET_WIDTH = 784;
const FACEFUSION_TARGET_HEIGHT = 1024;

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
  const yMax = Math.floor(height * 0.88);  const lumaAt = (x, y) => {
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

async function getReferenceFaceDetector() {
  if (state.referenceDetector) return state.referenceDetector;
  if (!state.referenceDetectorPromise) {
    state.referenceDetectorPromise = (async () => {
      const vision = state.vision || await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/wasm"
      );
      const options = {
        baseOptions: { modelAssetPath: FACE_DETECTOR_MODEL_URL, delegate: "GPU" },
        runningMode: "IMAGE",
        minDetectionConfidence: 0.20,
        minSuppressionThreshold: 0.30
      };
      try {
        return await FaceDetector.createFromOptions(vision, options);
      } catch (gpuError) {
        console.info("Referans yedek yüz algılama GPU'da başlatılamadı; CPU deneniyor.", gpuError);
        options.baseOptions.delegate = "CPU";
        return FaceDetector.createFromOptions(vision, options);
      }
    })();
  }
  try {
    state.referenceDetector = await state.referenceDetectorPromise;
    return state.referenceDetector;
  } catch (error) {
    state.referenceDetectorPromise = null;
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
  const weight = detectionMethod === "FaceDetector"
    ? Math.min(2, quality.weight)
    : quality.weight;

  return {
    canvas,
    alignedCanvas,
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

  let fallbackError = "BlazeFace yedek algılayıcısı yüz bulamadı.";
  try {
    const fallbackDetector = await getReferenceFaceDetector();
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
          reference, detected, sourceWidth, sourceHeight, "FaceDetector"
        );
      } catch (error) {
        fallbackError = error?.message || "BlazeFace hizalaması başarısız.";
      }
    }
  } catch (error) {
    fallbackError = error?.message || "BlazeFace yedek algılayıcısı başlatılamadı.";
  }

  throw new Error(`FaceLandmarker: ${primaryError}; BlazeFace: ${fallbackError}`);
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

  const targetCanvas = document.createElement("canvas");
  targetCanvas.width = FACEFUSION_TARGET_WIDTH;
  targetCanvas.height = FACEFUSION_TARGET_HEIGHT;
  const targetContext = targetCanvas.getContext("2d", { willReadFrequently: true });
  if (!targetContext) throw new Error("Hedef yüz görseli oluşturulamadı.");
  targetContext.fillStyle = "#10151d";
  targetContext.fillRect(0, 0, targetCanvas.width, targetCanvas.height);
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
      state.identityProfile = {
        key: profileKey,
        token: state.referenceLoadToken,
        referenceBlobs: cached.referenceBlobs,
        weights: cached.weights,
        appearanceProfileBlob: cached.appearanceProfileBlob,
        validCount: cached.validCount,
        totalWeight: cached.totalWeight
      };
      showToast(`Kayıtlı kişisel yüz profili yüklendi · ${cached.validCount}/${MAX_REFERENCE_FACES} referans.`);
    } else {
      const candidates = [];
      const referenceDiagnostics = [];
      for (const reference of readyReferences) {
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
        renderReferences();
      }

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
        referenceDiagnostics
      };
      state.identityProfile = {
        key: profileKey,
        token: state.referenceLoadToken,
        ...profile
      };
      const persisted = await saveCachedProfile(profileKey, profile);
      state.referenceFaceCropCount = appearance.validCount;
      state.selectedReferenceCount = appearance.validCount;
      renderReferences();
      showToast(
        persisted
          ? `${appearance.validCount}/${MAX_REFERENCE_FACES} fotoğraf harmanlandı · profil bu tarayıcıya kaydedildi.`
          : `${appearance.validCount}/${MAX_REFERENCE_FACES} fotoğraf harmanlandı · profil bu oturumda hazır.`
      );
    }
  }

  const profile = state.identityProfile;
  state.referenceFaceCropCount = profile.validCount;
  state.selectedReferenceCount = profile.validCount;  const faceBounds = {
    x: crop.face.x * FACEFUSION_TARGET_WIDTH / crop.width,
    y: crop.face.y * FACEFUSION_TARGET_HEIGHT / crop.height,
    width: crop.face.width * FACEFUSION_TARGET_WIDTH / crop.width,
    height: crop.face.height * FACEFUSION_TARGET_HEIGHT / crop.height
  };

  return {
    targetCanvas,
    referenceBlobs: profile.referenceBlobs,
    referenceWeights: profile.weights,
    appearanceProfileBlob: profile.appearanceProfileBlob,
    profileCount: profile.validCount,
    totalProfileWeight: profile.totalWeight,
    faceBounds,
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

  const consent = window.confirm(
    "YEREL İŞLEM ONAYI\\n\\n" +
    "Hedef yüz crop'u, 25 referansın tamamı ve hizalanmış ağırlıklı görünüm profili bu bilgisayardaki FaceFusion servisine (127.0.0.1:8765) gönderilecek. " +
    "Kimlik profili tüm referansların ağırlıklı yüz gömmelerini kullanır; görünüm profili tüm hizalanmış referansları birleştirir. İki yerel model çıktısı tek fotoğrafta harmanlanır. " +
    "Fotoğraflar harici bir AI servisine gönderilmeyecek. Devam edilsin mi?"
  );
  if (!consent) return;

  state.isSwapping = true;
  updateSwapAvailability();
  clearResult();
  elements.outputTitle.textContent = "Kişisel yüz profili uygulanıyor";
  elements.outputStatus.textContent = "Bilgisayarda çalışıyor";
  elements.outputPlaceholder.hidden = false;
  elements.outputPlaceholder.querySelector("strong").textContent = "Referans kimliği hazırlanıyor…";
  elements.outputMessage.textContent = "Hedef yüz ve yüzü algılanan referans fotoğrafları yerel FaceFusion motoruna aktarılıyor.";

  let generatedUrl = null;
  try {
    const base = localFaceFusionBase();
    const {
      targetCanvas,
      referenceBlobs,
      referenceWeights,
      appearanceProfileBlob,
      profileCount,
      totalProfileWeight,
      faceBounds,
      crop
    } = await buildLocalFaceFusionInputs();
    const targetBlob = await canvasToBlob(targetCanvas);

    const form = new FormData();
    form.append("target", targetBlob, "retake-target.png");
    form.append("appearance_profile", appearanceProfileBlob, "retake-25-photo-appearance-profile.png");
    referenceBlobs.forEach((blob, index) => {
      form.append("reference", blob, `retake-reference-${String(index + 1).padStart(2, "0")}.png`);
    });
    form.append("metadata", JSON.stringify({ referenceWeights, faceBounds }));

    setRuntimeStatus(`Retake Hybrid Profile · ${profileCount} referans · ağırlıklı kimlik ve görünüm profili`);
    elements.outputPlaceholder.querySelector("strong").textContent = "25 fotoğraflı yüz profili uygulanıyor…";
    elements.outputMessage.textContent =
      `${profileCount} hizalanmış referans AlphaFace kimlik profiline ve görünüm harmanına aktarılıyor. İki yerel geçiş çalıştırılacak.`;

    const response = await fetch(base + "/api/face-swap", {
      method: "POST",
      body: form,
      cache: "no-store"
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error || `FaceFusion işlemi başarısız (HTTP ${response.status}).`);
    }
    const generatedBlob = await response.blob();
    if (!generatedBlob.size || !generatedBlob.type.startsWith("image/")) {
      throw new Error("FaceFusion geçerli bir sonuç görseli döndürmedi.");
    }
    const generated = await loadImage(generatedBlob);
    generatedUrl = generated.url;
    const resultBlob = await applyGeneratedFacePatch(generated.image, crop);
    const resultUrl = createObjectURL(resultBlob);
    if (state.result?.url) URL.revokeObjectURL(state.result.url);
    state.result = { blob: resultBlob, url: resultUrl };
    elements.resultImage.src = resultUrl;
    elements.resultImage.hidden = false;
    elements.outputPlaceholder.hidden = true;
    elements.outputStage.classList.add("has-result");
    elements.outputSymbol.hidden = true;
    elements.outputTitle.textContent = "25 fotoğraflı profil sonucu";
    elements.outputStatus.textContent = "Yerel hibrit çıktı";
    elements.downloadButton.disabled = false;
    elements.disabledNote.textContent =
      `${profileCount}/${MAX_REFERENCE_FACES} referans yüzü hizalanıp kalite/açı ağırlıklarıyla tek kişisel profile katıldı (toplam ağırlık: ${totalProfileWeight}). AlphaFace kimlik ve UniFace görünüm geçişleri FaceFusion parser/occlusion maskeleriyle işlendi; PNG alfa maskesi gerçek değişen piksellerden türetildi. Maske dışındaki kaynak pikseller korunur.`;
    showToast(`Hibrit kişisel profil hazır · ${profileCount} referans harmanlandı.`);
    setRuntimeStatus(`Retake Hybrid Profile · ${profileCount} referans · sonuç hazır`);
  } catch (error) {
    console.error("Yerel FaceFusion işlemi başarısız:", error);
    elements.outputTitle.textContent = "İşlem tamamlanamadı";
    elements.outputStatus.textContent = "Hata";
    elements.outputPlaceholder.hidden = false;
    elements.outputPlaceholder.querySelector("strong").textContent = "Yerel FaceFusion işlemi başarısız";
    elements.outputMessage.textContent = error.message || "FaceFusion isteği başarısız oldu.";
    elements.outputStage.classList.remove("has-result");
    elements.outputSymbol.hidden = false;
    showToast(error.message || "FaceFusion yüz değiştirme başarısız.");
    setRuntimeStatus("FaceFusion · hata");
  } finally {
    if (generatedUrl) URL.revokeObjectURL(generatedUrl);
    state.isSwapping = false;
    updateSwapAvailability();
  }
}

function downloadResult() {
  if (!state.result?.url) return;
  const link = document.createElement("a");
  link.href = state.result.url;
  link.download = "retake-facefusion-" + Date.now() + ".png";
  document.body.append(link);
  link.click();
  link.remove();
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
      state.target.faces = detectFaces(state.target.image);
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
    setRuntimeStatus("Yüz algılama başlatılamadı");    updateSwapAvailability();
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
}

function initializeInteractions() {
  elements.targetInput.addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (file) await setTargetPhoto(file);
    event.target.value = "";
  });

  elements.targetDropzone.addEventListener("click", (event) => {
    if (event.target === elements.targetCanvas || event.target.closest(".preview-hint")) return;
    elements.targetInput.click();
  });
  elements.targetDropzone.addEventListener("keydown", (event) => {
    if (event.target !== elements.targetDropzone) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      elements.targetInput.click();
    }
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    elements.targetDropzone.addEventListener(eventName, (event) => {
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
  elements.swapButton.addEventListener("click", runFaceFusionEdit);
  elements.downloadButton.addEventListener("click", downloadResult);
  elements.themeButton.addEventListener("click", toggleTheme);

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
    if (state.result?.url) URL.revokeObjectURL(state.result.url);
    state.references.forEach((item) => {
      if (item.url) URL.revokeObjectURL(item.url);
    });
    state.detector?.close();
    state.referenceDetector?.close();
  });
}

async function initializeApp() {
  initializeTheme();
  initializeInteractions();

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