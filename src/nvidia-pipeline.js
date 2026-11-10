export const NVIDIA_BOARD_WIDTH = 1568;
export const NVIDIA_BOARD_HEIGHT = 1024;
export const NVIDIA_TARGET_PANEL_WIDTH = NVIDIA_BOARD_WIDTH / 2;
export const NVIDIA_TARGET_PANEL_HEIGHT = NVIDIA_BOARD_HEIGHT;

function finitePositive(value) {
  return Number.isFinite(value) && value > 0;
}

export function getFaceCropRect(imageWidth, imageHeight, normalizedFace, aspectRatio = NVIDIA_TARGET_PANEL_WIDTH / NVIDIA_TARGET_PANEL_HEIGHT) {
  if (!finitePositive(imageWidth) || !finitePositive(imageHeight) || !finitePositive(aspectRatio)) {
    throw new TypeError("Image dimensions and aspect ratio must be positive.");
  }
  const values = [normalizedFace?.x, normalizedFace?.y, normalizedFace?.width, normalizedFace?.height];
  if (values.some((value) => !Number.isFinite(value)) ||
      normalizedFace.width <= 0 || normalizedFace.height <= 0) {
    throw new TypeError("A valid normalized face rectangle is required.");
  }

  const face = {
    x: normalizedFace.x * imageWidth,
    y: normalizedFace.y * imageHeight,
    width: normalizedFace.width * imageWidth,
    height: normalizedFace.height * imageHeight
  };
  if (face.x < 0 || face.y < 0 ||
      face.x + face.width > imageWidth + 1 ||
      face.y + face.height > imageHeight + 1) {
    throw new RangeError("Face bounds are outside the source image.");
  }

  let cropHeight = Math.max(face.height * 1.85, (face.width * 1.55) / aspectRatio);
  let cropWidth = cropHeight * aspectRatio;
  const fit = Math.min(1, imageWidth / cropWidth, imageHeight / cropHeight);
  cropWidth *= fit;
  cropHeight *= fit;

  const centerX = face.x + face.width / 2;
  const centerY = face.y + face.height / 2;
  const x = Math.max(0, Math.min(imageWidth - cropWidth, centerX - cropWidth / 2));
  const y = Math.max(0, Math.min(imageHeight - cropHeight, centerY - cropHeight / 2));

  const rect = {
    x: Math.floor(x),
    y: Math.floor(y),
    width: Math.floor(cropWidth),
    height: Math.floor(cropHeight)
  };
  const localFace = {
    x: face.x - rect.x,
    y: face.y - rect.y,
    width: face.width,
    height: face.height
  };
  if (rect.width <= 0 || rect.height <= 0 ||
      localFace.x < 0 || localFace.y < 0 ||
      localFace.x + localFace.width > rect.width + 1 ||
      localFace.y + localFace.height > rect.height + 1) {
    throw new RangeError("Unable to create a crop that contains the selected face.");
  }
  return { ...rect, face: localFace };
}

export function createFaceAlphaMask(width, height, faceBounds, feather = 0.18) {
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    throw new TypeError("Mask dimensions must be positive integers.");
  }
  if (!faceBounds || [faceBounds.x, faceBounds.y, faceBounds.width, faceBounds.height].some((value) => !Number.isFinite(value)) ||
      faceBounds.width <= 0 || faceBounds.height <= 0) {
    throw new TypeError("A valid face rectangle is required.");
  }
  if (!Number.isFinite(feather) || feather <= 0 || feather >= 1) {
    throw new RangeError("Feather must be between 0 and 1.");
  }

  const centerX = faceBounds.x + faceBounds.width / 2;
  // Shift the blend region slightly downward and extend it below the detected
  // chin. This lets the generated lower-face area (including a beard) survive
  // compositing instead of being clipped by a tight oval around facial skin.
  const centerY = faceBounds.y + faceBounds.height * 0.62;
  const radiusX = faceBounds.width * 0.76;
  const radiusY = faceBounds.height * 0.82;
  const mask = new Float32Array(width * height);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const dx = (x + 0.5 - centerX) / radiusX;
      const dy = (y + 0.5 - centerY) / radiusY;
      const distance = Math.sqrt(dx * dx + dy * dy);
      const alpha = Math.max(0, Math.min(1, (1 - distance) / feather));
      mask[y * width + x] = alpha;
    }
  }
  return mask;
}

export function createNvidiaEditPrompt() {
  return [
    "This is a two-panel image-editing reference board.",
    "LEFT PANEL: the target portrait crop that must be edited.",
    "RIGHT PANEL: twenty-five different reference photographs of ONE SAME PERSON. They are identity references, not extra people to add to the target.",
    "Edit only the face of the person in the LEFT PANEL so that the face identity closely resembles the same person shown in the RIGHT PANEL.",
    "Use all twenty-five references together to infer stable identity traits. Do not choose or copy only one thumbnail.",
    "Preserve the target portrait's head pose, camera angle, expression, gaze direction, skin texture, lighting, shadows, color balance and photographic realism.",
    "If the target person's eyes are clearly closed, make them naturally open using the open-eye appearance visible in the references, without changing gaze direction or exaggerating the eyelids.",
    "Keep the target person's hairline, hairstyle, ears, neck, clothing, body, background and every object unchanged.",
    "Do not change the layout, do not edit the right reference panel, do not add text, do not add people, and do not create a collage.",
    "Return the same two-panel image with the original aspect ratio. This is a local portrait edit, not a full-scene regeneration."
  ].join(" ");
}


const MAX_NVIDIA_OUTPUT_BYTES = 14 * 1024 * 1024;

/**
 * Normalize the NVIDIA API Catalog response in the browser.
 * The Worker streams the upstream JSON untouched to stay within the
 * Cloudflare Free CPU budget.
 */
export function imageDataUrlFromNvidiaResponse(payload) {
  const candidates = [
    payload?.artifacts?.[0]?.base64,
    payload?.artifacts?.[0]?.image,
    payload?.images?.[0]?.b64_json,
    payload?.data?.[0]?.b64_json,
    payload?.image_base64,
    payload?.image
  ];

  for (const candidate of candidates) {
    if (typeof candidate !== "string" || !candidate.length) continue;
    if (/^data:image\/(png|jpeg|webp);base64,/i.test(candidate)) {
      const encoded = candidate.slice(candidate.indexOf(",") + 1);
      if (Math.floor(encoded.length * 0.75) > MAX_NVIDIA_OUTPUT_BYTES) {
        throw new RangeError("NVIDIA çıktısı 14 MB sınırını aşıyor.");
      }
      return candidate;
    }
    if (/^[A-Za-z0-9+/]+={0,2}$/.test(candidate)) {
      if (Math.floor(candidate.length * 0.75) > MAX_NVIDIA_OUTPUT_BYTES) {
        throw new RangeError("NVIDIA çıktısı 14 MB sınırını aşıyor.");
      }
      return "data:image/png;base64," + candidate;
    }
  }
  throw new Error("NVIDIA yanıtında desteklenen base64 görsel çıktısı bulunamadı.");
}
