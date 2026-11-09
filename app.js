import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/vision_bundle.mjs";
import {
  formatBytes,
  getLandmarkBounds,
  validateImageFile
} from "./src/core.js";
import { REFERENCE_CATALOG, getReferenceImageUrl } from "./src/reference-catalog.js";
import {
  NVIDIA_BOARD_WIDTH,
  NVIDIA_BOARD_HEIGHT,
  NVIDIA_TARGET_PANEL_WIDTH,
  getFaceCropRect,
  createFaceAlphaMask,
  createNvidiaEditPrompt
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

function nvidiaApiBase() {
  const configured = document.querySelector('meta[name="retake-ai-base"]')?.content?.trim();
  return (configured || (window.location.origin + "/api/retake")).replace(/\/+$/, "");
}

function updateSwapAvailability() {
  const referenceCount = state.references.filter((item) => item.status === "ready").length;
  const hasTarget = Boolean(state.target && Number.isInteger(state.selectedFaceIndex) &&
    state.target.faces[state.selectedFaceIndex]);
  const enabled = state.nvidiaStatus.configured && state.detector &&
    referenceCount === REFERENCE_CATALOG.length && hasTarget && !state.isSwapping;
  elements.swapButton.disabled = !enabled;
  elements.downloadButton.disabled = !state.result || state.isSwapping;
  const tag = elements.swapButton.querySelector("span");
  if (tag) tag.textContent = state.isSwapping ? "İşleniyor" :
    (enabled ? "NVIDIA hazır" : (state.nvidiaStatus.configured ? "Fotoğraf seç" : "Sunucu kurulmalı"));
}

async function checkNvidiaStatus() {
  try {
    const response = await fetch(nvidiaApiBase() + "/status", {
      method: "GET",
      cache: "no-store",
      headers: { Accept: "application/json" }
    });
    const payload = await response.json().catch(() => ({}));
    state.nvidiaStatus = {
      checked: true,
      configured: Boolean(response.ok && payload.configured),
      message: payload.message || (response.ok ? "" : "NVIDIA sunucu bağlantısı yanıt vermedi.")
    };
  } catch (error) {
    state.nvidiaStatus = {
      checked: true,
      configured: false,
      message: error.message || "NVIDIA sunucusuna bağlanılamadı."
    };
  }

  if (state.nvidiaStatus.configured) {
    elements.engineTitle.textContent = "NVIDIA FLUX bağlantısı hazır";
    elements.engineDescription.textContent =
      "Deneysel görüntü düzenleme motoru hazır. Hedef yüz crop'u ve 20 kimlik referansı yalnızca onayından sonra NVIDIA'ya gönderilir.";
  } else {
    elements.engineTitle.textContent = "NVIDIA sunucu anahtarı bekleniyor";
    elements.engineDescription.textContent =
      state.nvidiaStatus.message ||
      "NVIDIA API anahtarı Firebase Functions Secret olarak eklenmeli. Anahtar tarayıcıya veya GitHub'a konmamalı.";
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
  elements.outputTitle.textContent = "Önizleme";
  elements.outputStatus.textContent = "Beklemede";
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
  elements.referenceCount.textContent = readyCount + " / " + REFERENCE_CATALOG.length;
  elements.referencesEmpty.hidden = readyCount > 0;

  if (readyCount === 0) {
    const title = elements.referencesEmpty.querySelector("strong");
    const description = elements.referencesEmpty.querySelector("p");
    title.textContent = state.referencesLoading
      ? "Veritabanındaki yüzler yükleniyor"
      : "Kayıtlı yüz havuzuna ulaşılamadı";
    description.textContent = state.referencesLoading
      ? "Kayıtlı 20 referans doğrudan mevcut görsel veritabanından okunuyor. Yeniden fotoğraf eklemen gerekmiyor."
      : (state.referenceLoadError || "Sunucudaki /api/images/{id} uç noktasını kontrol et ve yeniden dene.");
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
      ? "VERİTABANINDA"
      : (reference.status === "loading" ? "YÜKLENİYOR" : "ERİŞİLEMİYOR");

    card.append(name, label);
    elements.referencesGrid.append(card);
  });

  elements.reloadReferences.disabled = state.referencesLoading;
  elements.reloadReferences.textContent = state.referencesLoading ? "Yükleniyor…" : "Veritabanından yenile";
  updateSwapAvailability();
}

async function loadDatabaseReferences({ force = false } = {}) {
  const token = ++state.referenceLoadToken;
  state.referencesLoading = true;
  state.referenceLoadError = null;
  state.references = REFERENCE_CATALOG.map((reference) => ({ ...reference, status: "loading" }));
  renderReferences();

  const configuredBase = document.querySelector('meta[name="retake-api-base"]')?.content?.trim() || "";
  const apiBase = configuredBase || window.location.origin;

  const results = await Promise.allSettled(REFERENCE_CATALOG.map(async (reference) => {
    const endpoint = getReferenceImageUrl(reference.id, apiBase, window.location.origin);
    const response = await fetch(endpoint, {
      method: "GET",
      credentials: new URL(endpoint).origin === window.location.origin ? "same-origin" : "omit",
      cache: force ? "reload" : "no-store",
      headers: { Accept: "image/avif,image/webp,image/png,image/jpeg" }
    });

    if (!response.ok) {
      throw new Error("API " + response.status + " yanıtı verdi.");
    }

    const blob = await response.blob();
    if (!blob.size) throw new Error("Veritabanı boş görsel döndürdü.");

    const loaded = await loadImage(blob);
    return {
      ...reference,
      status: "ready",
      url: loaded.url,
      image: loaded.image,
      mimeType: response.headers.get("content-type") || blob.type,
      size: blob.size
    };
  }));

  if (token !== state.referenceLoadToken) {
    results.forEach((result) => {
      if (result.status === "fulfilled") URL.revokeObjectURL(result.value.url);
    });
    return 0;
  }

  state.references.forEach((reference) => {
    if (reference.url) URL.revokeObjectURL(reference.url);
  });

  state.references = results.map((result, index) => result.status === "fulfilled"
    ? result.value
    : ({
        ...REFERENCE_CATALOG[index],
        status: "error",
        error: result.reason?.message || "Bilinmeyen API hatası"
      }));

  const loadedCount = state.references.filter((reference) => reference.status === "ready").length;
  const failedCount = REFERENCE_CATALOG.length - loadedCount;
  state.referencesLoading = false;
  state.referenceLoadError = failedCount
    ? (loadedCount
        ? loadedCount + " referans yüklendi; " + failedCount + " kayıt alınamadı."
        : "Kayıtlı yüzler API'den alınamadı. " +
          (state.references.find((item) => item.status === "error")?.error || ""))
    : null;

  renderReferences();
  if (failedCount) {
    showToast(loadedCount + " / " + REFERENCE_CATALOG.length + " kayıtlı referans yüklendi.");
  } else {
    showToast("20 kayıtlı referans başarıyla yüklendi.");
  }
  return loadedCount;
}

async function canvasToBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("Görsel hazırlanamadı.")), "image/png");
  });
}

async function blobToDataURL(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Görsel verisi okunamadı."));
    reader.readAsDataURL(blob);
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

function buildNvidiaContactSheet() {
  const readyReferences = state.references.filter((reference) => reference.status === "ready" && reference.image);
  if (readyReferences.length !== REFERENCE_CATALOG.length) {
    throw new Error("NVIDIA işlemi için veritabanındaki 20 referansın tamamı yüklenmiş olmalı.");
  }
  const face = state.target?.faces?.[state.selectedFaceIndex];
  if (!face) throw new Error("Önce hedef fotoğraftaki değiştirilecek yüzü seç.");

  const crop = getFaceCropRect(
    state.target.image.naturalWidth,
    state.target.image.naturalHeight,
    face
  );
  const canvas = document.createElement("canvas");
  canvas.width = NVIDIA_BOARD_WIDTH;
  canvas.height = NVIDIA_BOARD_HEIGHT;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("NVIDIA referans panosu oluşturulamadı.");

  context.fillStyle = "#10151d";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#18212b";
  context.fillRect(NVIDIA_TARGET_PANEL_WIDTH, 0, NVIDIA_BOARD_WIDTH - NVIDIA_TARGET_PANEL_WIDTH, NVIDIA_BOARD_HEIGHT);

  context.drawImage(
    state.target.image,
    crop.x, crop.y, crop.width, crop.height,
    0, 0, NVIDIA_TARGET_PANEL_WIDTH, NVIDIA_BOARD_HEIGHT
  );

  context.fillStyle = "rgba(10, 15, 22, .78)";
  context.fillRect(0, 0, NVIDIA_TARGET_PANEL_WIDTH, 38);
  context.fillStyle = "#ffffff";
  context.font = "700 16px system-ui";
  context.fillText("TARGET FACE", 22, 25);

  const panelX = NVIDIA_TARGET_PANEL_WIDTH;
  context.fillStyle = "#f0f5f7";
  context.font = "700 17px system-ui";
  context.fillText("20 IDENTITY REFERENCES", panelX + 22, 29);
  context.fillStyle = "#9caaba";
  context.font = "12px system-ui";
  context.fillText("All photos show the same person", panelX + 22, 50);

  const columns = 4;
  const rows = 5;
  const startY = 66;
  const gapX = 10;
  const gapY = 10;
  const outerX = panelX + 16;
  const outerWidth = NVIDIA_BOARD_WIDTH - panelX - 32;
  const cellWidth = (outerWidth - gapX * (columns - 1)) / columns;
  const cellHeight = (NVIDIA_BOARD_HEIGHT - startY - 20 - gapY * (rows - 1)) / rows;
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";

  readyReferences.forEach((reference, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    const x = outerX + column * (cellWidth + gapX);
    const y = startY + row * (cellHeight + gapY);
    context.fillStyle = "#f7f9fa";
    context.fillRect(x, y, cellWidth, cellHeight);
    drawContainedImage(context, reference.image, x + 5, y + 4, cellWidth - 10, cellHeight - 8);
  });

  return { canvas, crop };
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
  generatedContext.drawImage(
    generatedBoardImage,
    0, 0, NVIDIA_TARGET_PANEL_WIDTH, NVIDIA_BOARD_HEIGHT,
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

async function runNvidiaFaceEdit() {
  if (state.isSwapping) return;
  if (!state.nvidiaStatus.configured) {
    showToast("NVIDIA sunucu anahtarı yapılandırılmamış. Kurulum dosyasını kontrol et.");
    return;
  }
  if (!state.target || !Number.isInteger(state.selectedFaceIndex)) {
    showToast("Önce hedef fotoğrafı ve yüzü seç.");
    return;
  }
  if (state.references.filter((item) => item.status === "ready").length !== REFERENCE_CATALOG.length) {
    showToast("20 referansın tamamı yüklenmeden işlem başlatılamaz.");
    return;
  }

  const consent = window.confirm(
    "GİZLİLİK ONAYI\n\n" +
    "Yalnızca seçtiğin yüzün bulunduğu crop ve veritabanındaki 20 referans fotoğrafı birleştirilerek NVIDIA FLUX servisine gönderilecek. " +
    "Hedef fotoğrafın geri kalanı gönderilmeyecek. Görseller NVIDIA altyapısında işlenecek. " +
    "Devam etmek istiyor musun?"
  );
  if (!consent) return;

  state.isSwapping = true;
  updateSwapAvailability();
  elements.outputTitle.textContent = "NVIDIA ile işleniyor";
  elements.outputStatus.textContent = "Yükleniyor";
  elements.outputPlaceholder.hidden = false;
  elements.outputPlaceholder.querySelector("strong").textContent = "Yüz düzenleniyor…";
  elements.outputPlaceholder.querySelector("p").textContent =
    "Seçilen yüz crop'u ve 20 referans NVIDIA'ya gönderildi. Orijinal fotoğrafın geri kalanı tarayıcıda kalıyor.";

  let generatedUrl = null;
  try {
    const { canvas, crop } = buildNvidiaContactSheet();
    const boardBlob = await canvasToBlob(canvas);
    if (boardBlob.size > 10 * 1024 * 1024) {
      throw new Error("Referans panosu 10 MB sınırını aşıyor.");
    }
    const imageDataUrl = await blobToDataURL(boardBlob);
    setRuntimeStatus("NVIDIA FLUX · görsel gönderiliyor…");

    const response = await fetch(nvidiaApiBase() + "/edit", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ imageDataUrl, consent: true })
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = payload.error || ("NVIDIA API " + response.status + " yanıtı verdi.");
      throw new Error(detail);
    }
    if (typeof payload.imageDataUrl !== "string" || !payload.imageDataUrl.startsWith("data:image/")) {
      throw new Error("NVIDIA geçerli bir görsel çıktısı döndürmedi.");
    }

    const generated = await loadImage(await (await fetch(payload.imageDataUrl)).blob());
    generatedUrl = generated.url;
    const resultBlob = await applyGeneratedFacePatch(generated.image, crop);
    const resultUrl = createObjectURL(resultBlob);
    if (state.result?.url) URL.revokeObjectURL(state.result.url);
    state.result = { blob: resultBlob, url: resultUrl };
    elements.resultImage.src = resultUrl;
    elements.resultImage.hidden = false;
    elements.outputPlaceholder.hidden = true;
    elements.outputTitle.textContent = "Sonuç önizlemesi";
    elements.outputStatus.textContent = "Deneysel çıktı";
    elements.downloadButton.disabled = false;
    elements.disabledNote.textContent =
      "NVIDIA FLUX genel amaçlı görsel düzenleme modelidir. Kimlik benzerliği garanti edilmez; sonucu kontrol ederek indir.";
    showToast("Deneysel NVIDIA çıktısı hazır. Lütfen yüz benzerliğini kontrol et.");
    setRuntimeStatus("NVIDIA FLUX · sonuç hazır");
  } catch (error) {
    console.error("NVIDIA FLUX işlemi başarısız:", error);
    elements.outputTitle.textContent = "İşlem tamamlanamadı";
    elements.outputStatus.textContent = "Hata";
    elements.outputPlaceholder.hidden = false;
    elements.outputPlaceholder.querySelector("strong").textContent = "NVIDIA işlemi başarısız";
    elements.outputPlaceholder.querySelector("p").textContent = error.message || "NVIDIA API isteği başarısız oldu.";
    showToast(error.message || "NVIDIA yüz düzenleme işlemi başarısız.");
    setRuntimeStatus("NVIDIA FLUX · hata");
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
      "Yüz algılama cihaz üzerinde hazır. NVIDIA FLUX deneysel düzenleme için kullanılabilir; kimlik aktarımı garanti edilmez.";
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
  elements.reloadReferences.addEventListener("click", () => loadDatabaseReferences({ force: true }));
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
    loadDatabaseReferences(),
    initializeDetector(),
    checkNvidiaStatus()
  ]);

  const loaded = state.references.filter((reference) => reference.status === "ready").length;
  setRuntimeStatus(
    (state.detector ? "Yüz algılama hazır" : "Yüz algılama kullanılamıyor") +
    " · Veritabanı: " + loaded + " / " + REFERENCE_CATALOG.length + " referans"
  );
  updateSwapAvailability();

  if ("serviceWorker" in navigator && window.isSecureContext) {
    navigator.serviceWorker.register("./sw.js").catch((error) => {
      console.info("Service worker kurulamadı; uygulama çevrimiçi kullanılabilir.", error);
    });
  }
}

initializeApp();