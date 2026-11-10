import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/vision_bundle.mjs";
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
  NVIDIA_BOARD_HEIGHT,
  NVIDIA_TARGET_PANEL_WIDTH,
  getFaceCropRect,
  createFaceAlphaMask
} from "./src/nvidia-pipeline.js";
import { compositeMaskedRGBA, inspectOutsideMaskIntegrity } from "./src/face-only.js";

const MAX_PREVIEW_EDGE = 2048;
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

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
  target: null,
  references: [],
  referencesLoading: false,
  referenceFaceCropCount: null,
  selectedReferenceCount: 0,
  referenceLoadError: null,
  referenceLoadToken: 0,
  selectedFaceIndex: null,
  nvidiaStatus: { checked: false, configured: false },
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

function localComfyBase() {
  const configured = document.querySelector('meta[name="retake-comfy-base"]')?.content?.trim();
  const value = configured || "http://127.0.0.1:8188";
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
  const enabled = state.nvidiaStatus.configured && state.detector &&
    referenceCount === MAX_REFERENCE_FACES && hasTarget && !state.isSwapping;
  elements.swapButton.disabled = !enabled;
  elements.downloadButton.disabled = !state.result || state.isSwapping;
  const tag = elements.swapButton.querySelector("span");
  if (tag) tag.textContent = state.isSwapping ? "İşleniyor" :
    (enabled ? "Yerel FLUX hazır" : (state.nvidiaStatus.configured ? "Fotoğraf seç" : "Bilgisayara bağlan"));
}

async function checkNvidiaStatus() {
  const base = localComfyBase();
  if (!base) {
    state.nvidiaStatus = {
      checked: true,
      configured: false,
      message: "Yerel ComfyUI adresi geçersiz."
    };
  } else {
    try {
      const response = await fetch(base + "/system_stats", {
        method: "GET",
        cache: "no-store",
        headers: { Accept: "application/json" }
      });
      if (!response.ok) throw new Error("ComfyUI HTTP " + response.status);
      const nodeResponse = await fetch(base + "/object_info/UnetLoaderGGUF", {
        cache: "no-store",
        headers: { Accept: "application/json" }
      });
      if (!nodeResponse.ok) {
        throw new Error("ComfyUI çalışıyor ancak UnetLoaderGGUF düğümü bulunamadı. GGUF eklentisini ve ComfyUI başlangıç kayıtlarını kontrol et.");
      }
      state.nvidiaStatus = {
        checked: true,
        configured: true,
        message: "Yerel ComfyUI ve GGUF düğümü hazır."
      };
    } catch (error) {
      const detail = error?.message || "Yerel ComfyUI bağlantısı kurulamadı.";
      state.nvidiaStatus = {
        checked: true,
        configured: false,
        message: detail.includes("Failed to fetch") || detail.includes("NetworkError")
          ? "127.0.0.1:8188 bağlantısı kurulamadı. ComfyUI'yi --enable-cors-header https://retake06.web.app parametresiyle başlat; tarayıcı yerel ağ izni sorarsa izin ver."
          : detail
      };
    }
  }

  if (state.nvidiaStatus.configured) {
    elements.engineTitle.textContent = "Yerel FLUX hazır";
    elements.engineDescription.textContent =
      `ComfyUI bilgisayarında çalışıyor. 25 referans arasından seçilen en net ${KONTEXT_REFERENCE_COUNT} yüz ayrı kimlik girdisi olarak RTX A4000'de işlenecek.`;
  } else {
    elements.engineTitle.textContent = "Yerel FLUX servisi bekleniyor";
    elements.engineDescription.textContent = state.nvidiaStatus.message;
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
  elements.outputStatus.textContent = "Beklemede";
  elements.outputMessage.textContent = "Yerel FLUX modeli hazır olduğunda sonuç burada görünecek.";
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
    label.textContent = reference.status === "ready"
      ? "YERELDE"
      : (reference.status === "loading" ? "YÜKLENİYOR" : "ERİŞİLEMİYOR");

    card.append(name, label);
    elements.referencesGrid.append(card);
  });

  elements.reloadReferences.disabled = state.referencesLoading;
  elements.reloadReferences.textContent = state.referencesLoading ? "Yükleniyor…" : "Klasörden yenile";
  updateSwapAvailability();
}

async function loadLocalReferences({ force = false } = {}) {
  const token = ++state.referenceLoadToken;
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
      return { ...reference, status: "ready", url: loaded.url, image: loaded.image, size: blob.size };
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
    const loadedCount = state.references.filter((reference) => reference.status === "ready").length;
      const failedCount = MAX_REFERENCE_FACES - loadedCount;
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

const KONTEXT_REFERENCE_COUNT = 4;
const KONTEXT_REFERENCE_WIDTH = 384;
const KONTEXT_REFERENCE_HEIGHT = 512;

function createReferenceFaceInput(reference) {
  const image = reference.image;
  const sourceWidth = image.naturalWidth || image.width;
  const sourceHeight = image.naturalHeight || image.height;
  const faces = detectFaces(image).sort((a, b) => {
    const areaA = a.width * sourceWidth * a.height * sourceHeight;
    const areaB = b.width * sourceWidth * b.height * sourceHeight;
    return areaB - areaA;
  });
  const face = faces[0];
  if (!face) return null;

  const crop = getFaceCropRect(
    sourceWidth,
    sourceHeight,
    face,
    KONTEXT_REFERENCE_WIDTH / KONTEXT_REFERENCE_HEIGHT
  );
  const canvas = document.createElement("canvas");
  canvas.width = KONTEXT_REFERENCE_WIDTH;
  canvas.height = KONTEXT_REFERENCE_HEIGHT;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Referans yüzü için görsel alanı oluşturulamadı.");
  context.drawImage(
    image,
    crop.x, crop.y, crop.width, crop.height,
    0, 0, canvas.width, canvas.height
  );

  return {
    canvas,
    area: face.width * sourceWidth * face.height * sourceHeight,
    name: reference.name
  };
}

function buildLocalComfyInputs() {
  const readyReferences = state.references.filter((reference) => reference.status === "ready" && reference.image);
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

  // The target is the only image being edited. Identity references are encoded separately.
  const targetCanvas = document.createElement("canvas");
  targetCanvas.width = NVIDIA_TARGET_PANEL_WIDTH;
  targetCanvas.height = NVIDIA_BOARD_HEIGHT;
  const targetContext = targetCanvas.getContext("2d", { willReadFrequently: true });
  if (!targetContext) throw new Error("Hedef yüz görseli oluşturulamadı.");
  targetContext.fillStyle = "#10151d";
  targetContext.fillRect(0, 0, targetCanvas.width, targetCanvas.height);
  targetContext.drawImage(
    state.target.image,
    crop.x, crop.y, crop.width, crop.height,
    0, 0, targetCanvas.width, targetCanvas.height
  );

  const candidates = [];
  let failedReferenceCrops = 0;
  readyReferences.forEach((reference) => {
    try {
      const candidate = createReferenceFaceInput(reference);
      if (candidate) candidates.push(candidate);
      else failedReferenceCrops += 1;
    } catch (error) {
      failedReferenceCrops += 1;
      console.warn("Referans yüzü hazırlanamadı:", reference.name, error);
    }
  });

  candidates.sort((a, b) => b.area - a.area);
  const selectedReferences = candidates.slice(0, KONTEXT_REFERENCE_COUNT);
  if (selectedReferences.length !== KONTEXT_REFERENCE_COUNT) {
    throw new Error(
      `Ayrı kimlik girdileri için en az ${KONTEXT_REFERENCE_COUNT} yüz gerekli; ${selectedReferences.length} yüz hazırlanabildi. ` +
      "Referans fotoğraflarının netliğini ve yüz görünürlüğünü kontrol et."
    );
  }

  state.referenceFaceCropCount = candidates.length;
  state.selectedReferenceCount = selectedReferences.length;
  showToast(
    `${candidates.length}/${MAX_REFERENCE_FACES} yüz bulundu · en net ${selectedReferences.length} referans ayrı ayrı gönderilecek.`
  );
  if (failedReferenceCrops > 0) {
    console.warn(`${failedReferenceCrops} referans fotoğrafında yüz kırpımı yapılamadı.`);
  }

  return {
    targetCanvas,
    referenceCanvases: selectedReferences.map((reference) => reference.canvas),
    selectedReferenceNames: selectedReferences.map((reference) => reference.name),
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
  const alphaMask = createFaceAlphaMask(crop.width, crop.height, crop.face);
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

async function uploadComfyImage(base, blob, filename) {
  const form = new FormData();
  form.append("image", blob, filename);
  form.append("type", "input");
  form.append("overwrite", "true");

  const response = await fetch(base + "/upload/image", {
    method: "POST",
    body: form
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || !payload.name) {
    throw new Error(payload.error || "ComfyUI görseli yükleyemedi (HTTP " + response.status + ").");
  }
  return payload.name;
}

async function waitForComfyResult(base, promptId, timeoutMs = 15 * 60 * 1000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    await new Promise((resolve) => window.setTimeout(resolve, 1200));
    const response = await fetch(base + "/history/" + encodeURIComponent(promptId), {
      cache: "no-store",
      headers: { Accept: "application/json" }
    });
    if (!response.ok) throw new Error("ComfyUI işlem geçmişi okunamadı (HTTP " + response.status + ").");
    const history = await response.json();
    const entry = history[promptId];
    const images = entry?.outputs?.["136"]?.images;
    if (Array.isArray(images) && images.length > 0) {
      const image = images[0];
      const query = new URLSearchParams({
        filename: image.filename,
        subfolder: image.subfolder || "",
        type: image.type || "output"
      });
      const imageResponse = await fetch(base + "/view?" + query.toString(), {
        cache: "no-store"
      });
      if (!imageResponse.ok) throw new Error("ComfyUI sonuç görseli alınamadı (HTTP " + imageResponse.status + ").");
      return imageResponse.blob();
    }
    if (entry?.status?.completed === true || entry?.status?.status_str === "error") {
      const details = JSON.stringify(entry.status.messages || []);
      throw new Error("ComfyUI iş akışı tamamlanamadı: " + details.slice(0, 1200));
    }
  }
  throw new Error("ComfyUI işlemi 15 dakika içinde tamamlanmadı. Bilgisayarındaki ComfyUI kuyruğunu kontrol et.");
}

async function runNvidiaFaceEdit() {
  if (state.isSwapping) return;
  if (!state.nvidiaStatus.configured) {
    showToast("Önce bilgisayarındaki ComfyUI servisini başlat.");
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
    `Seçtiğin yüzün crop'u ve 25 fotoğraf arasından seçilen en net ${KONTEXT_REFERENCE_COUNT} yüz crop'u bu tarayıcıdan bilgisayarındaki ComfyUI servisine (127.0.0.1:8188) gönderilecek. ` +
    "Model RTX A4000 üzerinde yerel çalışacak; görseller NVIDIA bulut API'sine veya başka bir harici AI servisine gönderilmeyecek. Devam edilsin mi?"
  );
  if (!consent) return;

  state.isSwapping = true;
  updateSwapAvailability();
  clearResult();
  elements.outputTitle.textContent = "Yerel FLUX ile işleniyor";
  elements.outputStatus.textContent = "Bilgisayarda çalışıyor";
  elements.outputPlaceholder.hidden = false;
  elements.outputPlaceholder.querySelector("strong").textContent = "Fotoğraflar yerel modele gönderiliyor…";
  elements.outputMessage.textContent =
    `Hedef fotoğraf ve seçilen ${KONTEXT_REFERENCE_COUNT} kimlik referansı ayrı görseller olarak ComfyUI'ye aktarılıyor. İşlem RTX A4000 üzerinde yapılacak.`;

  let generatedUrl = null;
  try {
    const base = localComfyBase();
    const { targetCanvas, referenceCanvases, crop } = buildLocalComfyInputs();
    const [targetBlob, ...referenceBlobs] = await Promise.all([
      canvasToBlob(targetCanvas),
      ...referenceCanvases.map((canvas) => canvasToBlob(canvas))
    ]);

    setRuntimeStatus("Yerel FLUX · hedef ve ayrı kimlik referansları ComfyUI'ye yükleniyor…");
    const [targetName, ...referenceNames] = await Promise.all([
      uploadComfyImage(base, targetBlob, "retake-target-face.png"),
      ...referenceBlobs.map((blob, index) =>
        uploadComfyImage(base, blob, `retake-identity-reference-${index + 1}.png`)
      )
    ]);

    const workflowResponse = await fetch("./workflows/retake-flux-api.json", {
      cache: "no-store",
      headers: { Accept: "application/json" }
    });
    if (!workflowResponse.ok) throw new Error("Retake FLUX API iş akışı indirilemedi.");
    const workflow = await workflowResponse.json();
    workflow["191"].inputs.image = targetName;
    referenceNames.forEach((name, index) => {
      workflow[String(192 + index)].inputs.image = name;
    });
    workflow["6"].inputs.text = [
      `Edit the single target portrait image. ${KONTEXT_REFERENCE_COUNT} additional reference images are supplied separately as identity references through ReferenceLatent conditioning.`,
      "The target image is the only canvas to edit. All reference images show the same reference person and are authoritative identity sources, not additional people to place in the scene.",
      "Completely replace the target person's facial identity with the identity shared by all reference images. This is a full facial identity replacement, not a beauty retouch or a minor feature adjustment.",
      "The new face must match the reference person's overall facial geometry and identity: forehead and temples, face width and shape, eyebrows, eyes and eyelids, nose bridge and tip, cheeks, lips and mouth, jawline, chin and apparent age.",
      "Do not preserve the target person's original facial identity. Preserve only the target's head pose, camera angle, expression, gaze direction and natural lighting as compatible with the new identity.",
      "Keep the target person's hairstyle, hairline, ears, neck, clothing, body and background unchanged. Do not copy hair, clothing, accessories or background from the reference images.",
      "Create one photorealistic portrait image only. Do not create a collage, contact sheet, extra faces, additional people, text or panels."
    ].join(" ");

    workflow["31"].inputs.seed = Math.floor(Math.random() * 9007199254740000);
    workflow["136"].inputs.filename_prefix = "Retake-Local-FLUX";
    setRuntimeStatus("Yerel FLUX · RTX A4000 üzerinde çalışıyor…");
    elements.outputPlaceholder.querySelector("strong").textContent = "RTX A4000 fotoğrafı düzenliyor…";
    elements.outputMessage.textContent = "ComfyUI iş kuyruğuna gönderildi. Bu aşama bilgisayarındaki ekran kartında çalışır.";

    const queueResponse = await fetch(base + "/prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({
        prompt: workflow,
        client_id: "retake-" + (crypto.randomUUID ? crypto.randomUUID() : Date.now())
      })
    });
    const queuePayload = await queueResponse.json().catch(() => ({}));
    if (!queueResponse.ok || !queuePayload.prompt_id) {
      const validation = queuePayload.node_errors
        ? JSON.stringify(queuePayload.node_errors).slice(0, 1200)
        : (queuePayload.error || "ComfyUI iş akışını kuyruğa alamadı.");
      throw new Error(validation);
    }

    const generatedBlob = await waitForComfyResult(base, queuePayload.prompt_id);
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
    elements.outputTitle.textContent = "Yerel FLUX sonucu";
    elements.outputStatus.textContent = "Yerel çıktı";
    elements.downloadButton.disabled = false;
    elements.disabledNote.textContent =
      `${state.referenceFaceCropCount ?? "?"}/${MAX_REFERENCE_FACES} referansta yüz bulundu; en net ${state.selectedReferenceCount}/${KONTEXT_REFERENCE_COUNT} yüz ayrı kimlik girdisi olarak kullanıldı. Kimlik benzerliği garanti edilmez.`;
    showToast(`Yerel FLUX çıktısı hazır · ${state.selectedReferenceCount} ayrı kimlik referansı kullanıldı.`);
    setRuntimeStatus(`Yerel FLUX · ${state.selectedReferenceCount} ayrı referans · sonuç hazır`);
  } catch (error) {
    console.error("Yerel ComfyUI FLUX işlemi başarısız:", error);
    elements.outputTitle.textContent = "İşlem tamamlanamadı";
    elements.outputStatus.textContent = "Hata";
    elements.outputPlaceholder.hidden = false;
    elements.outputPlaceholder.querySelector("strong").textContent = "Yerel FLUX işlemi başarısız";
    elements.outputMessage.textContent = error.message || "ComfyUI isteği başarısız oldu.";
    elements.outputStage.classList.remove("has-result");
    elements.outputSymbol.hidden = false;
    showToast(error.message || "Yerel FLUX düzenleme başarısız.");
    setRuntimeStatus("Yerel FLUX · hata");
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
  link.download = "retake-nvidia-face-edit-" + Date.now() + ".png";
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
      "MediaPipe yüzleri cihazda GPU ile algılamayı dener, desteklenmezse CPU'ya geçer. Yerel Face Swap modeli yoktur.";
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
  elements.swapButton.addEventListener("click", runNvidiaFaceEdit);
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
  });
}

async function initializeApp() {
  initializeTheme();
  initializeInteractions();

  await Promise.allSettled([
    loadLocalReferences(),
    initializeDetector(),
    checkNvidiaStatus()
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