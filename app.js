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
      `ComfyUI bilgisayarında çalışıyor. Hedef yüz crop'u ve ${MAX_REFERENCE_FACES} referans fotoğrafı internet üzerinden bir AI servisine gönderilmeden RTX A4000'de işlenecek.`;
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

function drawContainedImage(context, image, x, y, width, height) {
  const sourceWidth = image.naturalWidth || image.width;
  const sourceHeight = image.naturalHeight || image.height;
  const scale = Math.min(width / sourceWidth, height / sourceHeight);
  const drawWidth = sourceWidth * scale;
  const drawHeight = sourceHeight * scale;
  context.drawImage(image, x + (width - drawWidth) / 2, y + (height - drawHeight) / 2, drawWidth, drawHeight);
}

function drawReferenceFaceCrop(context, image, x, y, width, height) {
  try {
    const sourceWidth = image.naturalWidth || image.width;
    const sourceHeight = image.naturalHeight || image.height;
    const faces = detectFaces(image)
      .sort((a, b) => (b.width * b.height) - (a.width * a.height));
    const face = faces[0];
    if (!face) throw new Error("Referans fotoğrafında yüz algılanamadı.");

    const crop = getFaceCropRect(sourceWidth, sourceHeight, face, width / height);
    context.drawImage(
      image,
      crop.x, crop.y, crop.width, crop.height,
      x, y, width, height
    );
    return true;
  } catch (error) {
    console.warn("Referans yüzü kırpılamadı; tam görsel kullanılacak.", error);
    drawContainedImage(context, image, x, y, width, height);
    return false;
  }
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

  // ComfyUI ImageStitch receives two separate, equally sized image panels.
  const targetCanvas = document.createElement("canvas");
  targetCanvas.width = NVIDIA_TARGET_PANEL_WIDTH;
  targetCanvas.height = NVIDIA_BOARD_HEIGHT;
  const targetContext = targetCanvas.getContext("2d", { willReadFrequently: true });
  if (!targetContext) throw new Error("Hedef yüz panosu oluşturulamadı.");
  targetContext.fillStyle = "#10151d";
  targetContext.fillRect(0, 0, targetCanvas.width, targetCanvas.height);
  targetContext.drawImage(
    state.target.image,
    crop.x, crop.y, crop.width, crop.height,
    0, 0, targetCanvas.width, targetCanvas.height
  );
  targetContext.fillStyle = "rgba(10, 15, 22, .78)";
  targetContext.fillRect(0, 0, targetCanvas.width, 38);
  targetContext.fillStyle = "#ffffff";
  targetContext.font = "700 16px system-ui";
  targetContext.fillText("TARGET FACE", 22, 25);

  const referencesCanvas = document.createElement("canvas");
  referencesCanvas.width = NVIDIA_TARGET_PANEL_WIDTH;
  referencesCanvas.height = NVIDIA_BOARD_HEIGHT;
  const context = referencesCanvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Referans panosu oluşturulamadı.");
  context.fillStyle = "#18212b";
  context.fillRect(0, 0, referencesCanvas.width, referencesCanvas.height);
  context.fillStyle = "#f0f5f7";
  context.font = "700 17px system-ui";
  context.fillText(`${MAX_REFERENCE_FACES} FACE-CROPPED REFERENCES`, 22, 29);
  context.fillStyle = "#9caaba";
  context.font = "12px system-ui";
  context.fillText("All photos show the same person", 22, 50);

  const columns = 5;
  const rows = 5;
  const startY = 66;
  const gapX = 10;
  const gapY = 10;
  const outerX = 16;
  const outerWidth = referencesCanvas.width - 32;
  const cellWidth = (outerWidth - gapX * (columns - 1)) / columns;
  const cellHeight = (referencesCanvas.height - startY - 20 - gapY * (rows - 1)) / rows;
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";

  let faceCroppedReferences = 0;
  readyReferences.forEach((reference, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    const x = outerX + column * (cellWidth + gapX);
    const y = startY + row * (cellHeight + gapY);
    context.fillStyle = "#f7f9fa";
    context.fillRect(x, y, cellWidth, cellHeight);
    if (drawReferenceFaceCrop(context, reference.image, x + 5, y + 4, cellWidth - 10, cellHeight - 8)) {
      faceCroppedReferences += 1;
    }
  });

  showToast(`${faceCroppedReferences}/${MAX_REFERENCE_FACES} referans yüzü yakınlaştırıldı.`);
  return { targetCanvas, referencesCanvas, crop };
}

function applyGeneratedFacePatch(generatedBoardImage, crop) {
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
  const outputWidth = generatedBoardImage.naturalWidth || generatedBoardImage.width;
  const outputHeight = generatedBoardImage.naturalHeight || generatedBoardImage.height;
  const outputTargetPanelWidth = Math.floor(outputWidth / 2);
  generatedContext.drawImage(
    generatedBoardImage,
    0, 0, outputTargetPanelWidth, outputHeight,
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
    `Seçtiğin yüzün crop'u ve ${MAX_REFERENCE_FACES} referans fotoğrafı bu tarayıcıdan bilgisayarındaki ComfyUI servisine (127.0.0.1:8188) gönderilecek. ` +
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
    "Hedef yüz crop'u ve 25 referans fotoğrafı bilgisayarındaki ComfyUI'ye aktarılıyor. İşlem RTX A4000 üzerinde yapılacak.";

  let generatedUrl = null;
  try {
    const base = localComfyBase();
    const { targetCanvas, referencesCanvas, crop } = buildLocalComfyInputs();
    const [targetBlob, referencesBlob] = await Promise.all([
      canvasToBlob(targetCanvas),
      canvasToBlob(referencesCanvas)
    ]);

    setRuntimeStatus("Yerel FLUX · görseller ComfyUI'ye yükleniyor…");
    const [targetName, referencesName] = await Promise.all([
      uploadComfyImage(base, targetBlob, "retake-target-face.png"),
      uploadComfyImage(base, referencesBlob, "retake-identity-references.png")
    ]);

    const workflowResponse = await fetch("./workflows/retake-flux-api.json", {
      cache: "no-store",
      headers: { Accept: "application/json" }
    });
    if (!workflowResponse.ok) throw new Error("Retake FLUX API iş akışı indirilemedi.");
    const workflow = await workflowResponse.json();
    workflow["191"].inputs.image = targetName;
    workflow["192"].inputs.image = referencesName;
    workflow["6"].inputs.text = [
      "The input is one composite image with two panels arranged side by side.",
      "LEFT PANEL: This is the target portrait. Replace the person's facial identity with the identity shown in the RIGHT PANEL.",
      "RIGHT PANEL: These are enlarged face crops of the same reference person from different photos and angles. Treat them as the authoritative source for facial identity.",
      "Identity transfer is the primary task. Do not keep or reproduce the target person's original facial identity. Match the reference person's stable facial traits, including face shape, eyes, eyebrows, nose, lips, cheeks, jawline and age appearance.",
      "Apply the new identity naturally to the target face while preserving the target's head pose, expression, gaze direction and camera angle.",
      "Keep the target hairstyle, hairline, ears, neck, clothing, body, lighting, shadows and background unchanged. Do not copy hairstyles, clothing or backgrounds from the reference photos.",
      "Use realistic anatomy, natural skin texture and photographic lighting. Do not edit the right panel, add people, add text or create a collage.",
      "Return the complete two-panel composition in the original arrangement."
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
      "FLUX.1-Kontext-dev bilgisayarındaki RTX A4000'de çalıştı. Kimlik benzerliği garanti edilmez; sonucu kontrol ederek indir.";
    showToast("Yerel FLUX çıktısı hazır. Lütfen yüz benzerliğini kontrol et.");
    setRuntimeStatus("Yerel FLUX · sonuç hazır");
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