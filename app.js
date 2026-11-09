import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/vision_bundle.mjs";
import {
  MAX_REFERENCE_FACES,
  formatBytes,
  getLandmarkBounds,
  validateImageFile
} from "./src/core.js";

const DB_NAME = "retake-local-library";
const DB_VERSION = 1;
const STORE_NAME = "references";
const DETECTOR_VERSION = "mediapipe-face-landmarker-1.2.2";
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
  referenceInput: $("#referenceInput"),
  addReferences: $("#addReferencesButton"),
  clearReferences: $("#clearReferencesButton"),
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
  database: null,
  target: null,
  references: [],
  selectedFaceIndex: null,
  busy: false,
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

function setBusy(busy) {
  state.busy = busy;
  elements.addReferences.disabled = busy || !state.detector || state.references.length >= MAX_REFERENCE_FACES;
  elements.targetInput.disabled = busy;
  elements.referenceInput.disabled = busy;
}

function openDatabase() {
  return new Promise((resolve, reject) => {
    if (!("indexedDB" in window)) {
      reject(new Error("Bu tarayıcı yerel fotoğraf kitaplığını desteklemiyor."));
      return;
    }
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        const store = database.createObjectStore(STORE_NAME, { keyPath: "id" });
        store.createIndex("createdAt", "createdAt", { unique: false });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("Yerel veritabanı açılamadı."));
    request.onblocked = () => reject(new Error("Yerel veritabanı başka bir sekme tarafından kilitlendi."));
  });
}

function databaseRequest(mode, operation) {
  return new Promise((resolve, reject) => {
    if (!state.database) {
      reject(new Error("Yerel fotoğraf kitaplığı kullanılamıyor."));
      return;
    }
    const transaction = state.database.transaction(STORE_NAME, mode);
    const request = operation(transaction.objectStore(STORE_NAME));
    let result;
    request.onsuccess = () => { result = request.result; };
    request.onerror = () => reject(request.error || new Error("Yerel kayıt işlemi başarısız."));
    transaction.oncomplete = () => resolve(result);
    transaction.onerror = () => reject(transaction.error || new Error("Yerel kayıt işlemi başarısız."));
    transaction.onabort = () => reject(transaction.error || new Error("Yerel kayıt işlemi iptal edildi."));
  });
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

function selectTargetFace(index) {
  if (!state.target || !Number.isInteger(index) || !state.target.faces[index]) return;
  state.selectedFaceIndex = index;
  drawTargetPreview();
  renderFaceChips();
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
    state.target = { file, image: loaded.image, url: loaded.url, faces };
    state.selectedFaceIndex = faces.length > 0 ? 0 : null;
    elements.targetEmpty.hidden = true;
    elements.targetPreview.hidden = false;
    elements.clearTarget.hidden = false;
    elements.targetMeta.textContent =
      loaded.image.naturalWidth + " × " + loaded.image.naturalHeight + " px · " + formatBytes(file.size) + " · " + detectionMessage;
    drawTargetPreview();
    renderFaceChips();
    showToast(detectionMessage);
  } catch (error) {
    showToast(error.message || "Fotoğraf yüklenemedi.");
  }
}

function clearTargetPhoto() {
  if (state.target?.url) URL.revokeObjectURL(state.target.url);
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
}

function renderReferences() {
  elements.referencesGrid.replaceChildren();
  elements.referenceCount.textContent = state.references.length + " / " + MAX_REFERENCE_FACES;
  elements.referencesEmpty.hidden = state.references.length > 0;
  elements.clearReferences.hidden = state.references.length === 0;

  state.references.forEach((reference) => {
    const card = document.createElement("article");
    card.className = "reference-card";

    const preview = document.createElement("div");
    preview.className = "reference-select";

    const image = document.createElement("img");
    image.src = reference.url;
    image.alt = "Kimlik referansı: " + reference.name;
    image.loading = "lazy";
    image.decoding = "async";

    const name = document.createElement("span");
    name.className = "reference-name";
    name.textContent = reference.name;

    const label = document.createElement("span");
    label.className = "reference-state";
    label.textContent = "KİMLİK HAVUZUNDA";

    preview.append(image, name, label);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "reference-delete";
    remove.textContent = "Kitaplıktan kaldır";
    remove.setAttribute("aria-label", reference.name + " referansını sil");
    remove.addEventListener("click", () => removeReference(reference.id));

    card.append(preview, remove);
    elements.referencesGrid.append(card);
  });

  elements.addReferences.disabled =
    state.busy || !state.detector || state.references.length >= MAX_REFERENCE_FACES || !state.database;
}

async function loadSavedReferences() {
  if (!state.database) return;
  const records = await databaseRequest("readonly", (store) => store.getAll());
  records.sort((a, b) => a.createdAt - b.createdAt);
  state.references = records.map((record) => ({
    ...record,
    url: createObjectURL(record.blob)
  }));
  renderReferences();
}

function yieldToBrowser() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

async function importReferencePhotos(files) {
  if (state.busy) return;
  if (!state.detector) {
    showToast("Referansları doğrulamak için önce yüz algılama motorunun yüklenmesini bekleyin.");
    return;
  }
  const selectedFiles = Array.from(files || []);
  if (selectedFiles.length === 0) return;

  const remaining = MAX_REFERENCE_FACES - state.references.length;
  if (remaining <= 0) {
    showToast("20 referans fotoğrafı sınırına ulaştın.");
    return;
  }

  setBusy(true);
  let added = 0;
  let skipped = 0;
  const queue = selectedFiles.slice(0, remaining);

  try {
    for (let index = 0; index < queue.length; index += 1) {
      const file = queue[index];
      const validation = validateImageFile(file);
      if (!validation.valid) {
        skipped += 1;
        continue;
      }

      let loaded;
      try {
        loaded = await loadImage(file);
        const faces = detectFaces(loaded.image);
        if (faces.length !== 1) {
          skipped += 1;
          URL.revokeObjectURL(loaded.url);
          loaded = null;
          continue;
        }

        const record = {
          id: crypto.randomUUID(),
          name: file.name || ("Referans " + (state.references.length + 1)),
          blob: file,
          mimeType: file.type,
          size: file.size,
          createdAt: Date.now() + index,
          detectorVersion: DETECTOR_VERSION
        };
        await databaseRequest("readwrite", (store) => store.put(record));
        state.references.push({ ...record, url: loaded.url });
        loaded = null;
        added += 1;
      } catch (error) {
        skipped += 1;
        if (loaded?.url) URL.revokeObjectURL(loaded.url);
        console.warn("Referans içe aktarılamadı:", error);
      }

      renderReferences();
      setRuntimeStatus("Referanslar denetleniyor: " + (index + 1) + " / " + queue.length);
      await yieldToBrowser();
    }

    renderReferences();
    showToast(added + " referans eklendi · " + skipped + " dosya atlandı.");
  } finally {
    setBusy(false);
    renderReferences();
    setRuntimeStatus(state.detector ? "Yüz algılama hazır · Fotoğraflar cihazda işleniyor" : "Yüz algılama kullanılamıyor");
    elements.referenceInput.value = "";
  }
}

async function removeReference(id) {
  const reference = state.references.find((item) => item.id === id);
  if (!reference || !window.confirm('"' + reference.name + '" referansını kitaplıktan kaldırmak istiyor musun?')) return;
  try {
    await databaseRequest("readwrite", (store) => store.delete(id));
    URL.revokeObjectURL(reference.url);
    state.references = state.references.filter((item) => item.id !== id);
    renderReferences();
    showToast("Referans kitaplıktan kaldırıldı.");
  } catch (error) {
    showToast(error.message || "Referans silinemedi.");
  }
}

async function clearReferences() {
  if (state.references.length === 0) return;
  if (!window.confirm("Kitaplıktaki " + state.references.length + " referans fotoğrafının tamamı silinsin mi?")) return;
  try {
    await databaseRequest("readwrite", (store) => store.clear());
    state.references.forEach((item) => URL.revokeObjectURL(item.url));
    state.references = [];
    renderReferences();
    showToast("Referans kitaplığı temizlendi.");
  } catch (error) {
    showToast(error.message || "Kitaplık temizlenemedi.");
  }
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
      "Yüz algılama cihaz üzerinde hazır. Gerçek kimlik korumalı Face Swap modeli henüz entegre edilmedi.";
    if (state.target) {
      state.target.faces = detectFaces(state.target.image);
      state.selectedFaceIndex = state.target.faces.length > 0 ? 0 : null;
      elements.targetMeta.textContent =
        state.target.image.naturalWidth + " × " + state.target.image.naturalHeight + " px · " +
        formatBytes(state.target.file.size) + " · " +
        (state.target.faces.length === 0 ? "Fotoğrafta yüz bulunamadı." : state.target.faces.length + " yüz algılandı.");
      drawTargetPreview();
      renderFaceChips();
    }
    setRuntimeStatus("Yüz algılama hazır · Yerel işlem etkin");
    renderReferences();
  } catch (error) {
    console.error("MediaPipe başlatılamadı:", error);
    elements.engineTitle.textContent = "Yüz algılama motoru yüklenemedi";
    elements.engineDescription.textContent =
      "İlk kullanımda internet bağlantısını kontrol et ve sayfayı yenile. Fotoğraflar bu durumda da sunucuya gönderilmez.";
    setRuntimeStatus("Yüz algılama başlatılamadı");
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
  elements.referenceInput.addEventListener("change", (event) => importReferencePhotos(event.target.files));
  elements.addReferences.addEventListener("click", () => elements.referenceInput.click());
  elements.clearReferences.addEventListener("click", clearReferences);
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
    state.references.forEach((item) => URL.revokeObjectURL(item.url));
    state.detector?.close();
    state.database?.close();
  });
}

async function initializeApp() {
  initializeTheme();
  initializeInteractions();

  try {
    state.database = await openDatabase();
    await loadSavedReferences();
    try {
      await navigator.storage?.persist?.();
    } catch {
      // Persistent storage is a best-effort browser capability.
    }
  } catch (error) {
    console.warn("Yerel veritabanı kullanılamıyor:", error);
    elements.referencesEmpty.querySelector("p").textContent =
      "Bu tarayıcı yerel depolamayı desteklemiyor. Referans kitaplığı kullanılamıyor.";
    elements.addReferences.disabled = true;
  }

  await initializeDetector();

  if ("serviceWorker" in navigator && window.isSecureContext) {
    navigator.serviceWorker.register("./sw.js").catch((error) => {
      console.info("Service worker kurulamadı; uygulama çevrimiçi kullanılabilir.", error);
    });
  }
}

initializeApp();