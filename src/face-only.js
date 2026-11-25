const CHANNELS = 4;
const isByteArray = (value) => value instanceof Uint8Array || value instanceof Uint8ClampedArray;

/**
 * Blend a generated face patch into an original RGBA crop.
 * alphaMask is one normalized float (0...1) per pixel. Pixels with alpha 0
 * are copied byte-for-byte from the original; pixels outside the face mask
 * therefore remain untouched.
 */
export function compositeMaskedRGBA(sourceRGBA, generatedRGBA, alphaMask, width, height) {
  const expectedPixels = width * height;
  const expectedChannels = expectedPixels * CHANNELS;
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    throw new TypeError("width and height must be positive integers");
  }
  if (!isByteArray(sourceRGBA) || sourceRGBA.length !== expectedChannels) {
    throw new TypeError("sourceRGBA must be a Uint8Array with width * height * 4 channels");
  }
  if (!isByteArray(generatedRGBA) || generatedRGBA.length !== expectedChannels) {
    throw new TypeError("generatedRGBA must be a Uint8Array with width * height * 4 channels");
  }
  if (!(alphaMask instanceof Float32Array || alphaMask instanceof Float64Array) ||
      alphaMask.length !== expectedPixels) {
    throw new TypeError("alphaMask must be a Float32Array or Float64Array with one value per pixel");
  }

  const output = new Uint8ClampedArray(expectedChannels);
  for (let pixel = 0; pixel < expectedPixels; pixel += 1) {
    const alpha = alphaMask[pixel];
    if (!Number.isFinite(alpha) || alpha < 0 || alpha > 1) {
      throw new RangeError("alphaMask values must be finite and between 0 and 1");
    }

    const offset = pixel * CHANNELS;
    if (alpha === 0) {
      output.set(sourceRGBA.subarray(offset, offset + CHANNELS), offset);
      continue;
    }
    // Only colour is blended. The generated patch carries the change matte in
    // its alpha channel; mixing that in left a semi-transparent ring around
    // the face in PNG exports. The photo keeps its own opacity.
    const inverse = 1 - alpha;
    for (let channel = 0; channel < 3; channel += 1) {
      output[offset + channel] = alpha === 1
        ? generatedRGBA[offset + channel]
        : Math.round(sourceRGBA[offset + channel] * inverse + generatedRGBA[offset + channel] * alpha);
    }
    output[offset + 3] = sourceRGBA[offset + 3];
  }
  return output;
}

/**
 * Insert an ALREADY COMPOSITED local-engine PNG over the source image.
 *
 * The FaceFusion backend has applied the soft mask, tone harmonization and
 * multiband blending to the RGB values before writing its RGBA PNG. Applying
 * that PNG's alpha once more would square the feather mask and dilute the
 * swapped identity, especially around the eyes, jaw and mouth.
 *
 * Here alpha is an *edit-extent matte*: 0 means keep the original pixel;
 * any nonzero value means copy the final preblended RGB. The source's opacity
 * is always preserved. Only use this function for confirmed preblended output,
 * not ordinary RGBA foreground patches.
 */
export function compositePreblendedRGBA(sourceRGBA, preblendedRGBA, extentMask, width, height) {
  const pixels = width * height;
  const channels = pixels * CHANNELS;
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    throw new TypeError("width and height must be positive integers");
  }
  if (!isByteArray(sourceRGBA) || sourceRGBA.length !== channels ||
      !isByteArray(preblendedRGBA) || preblendedRGBA.length !== channels) {
    throw new TypeError("RGBA buffers must match width * height * 4 channels");
  }
  if (!(extentMask instanceof Float32Array || extentMask instanceof Float64Array) ||
      extentMask.length !== pixels) {
    throw new TypeError("extentMask must contain one value per pixel");
  }

  const result = new Uint8ClampedArray(sourceRGBA);
  for (let pixel = 0; pixel < pixels; pixel += 1) {
    const alpha = extentMask[pixel];
    if (!Number.isFinite(alpha) || alpha < 0 || alpha > 1) {
      throw new RangeError("extentMask values must be finite and between 0 and 1");
    }
    if (alpha === 0) continue;
    const offset = pixel * CHANNELS;
    result[offset] = preblendedRGBA[offset];
    result[offset + 1] = preblendedRGBA[offset + 1];
    result[offset + 2] = preblendedRGBA[offset + 2];
  }
  return result;
}

/**
 * Reports changes outside the binary face area (mask <= 0).
 * Useful as a post-compositing integrity gate before exporting the image.
 */
export function inspectOutsideMaskIntegrity(sourceRGBA, outputRGBA, alphaMask, width, height) {
  const pixels = width * height;
  const channels = pixels * CHANNELS;
  if (!isByteArray(sourceRGBA) || sourceRGBA.length !== channels ||
      !isByteArray(outputRGBA) || outputRGBA.length !== channels ||
      !(alphaMask instanceof Float32Array || alphaMask instanceof Float64Array) ||
      alphaMask.length !== pixels) {
    throw new TypeError("Image buffers and mask dimensions do not match");
  }

  let changedPixels = 0;
  for (let pixel = 0; pixel < pixels; pixel += 1) {
    const alpha = alphaMask[pixel];
    if (!Number.isFinite(alpha) || alpha < 0 || alpha > 1) {
      throw new RangeError("alphaMask values must be finite and between 0 and 1");
    }
    if (alpha > 0) continue;
    const offset = pixel * CHANNELS;
    let changed = false;
    for (let channel = 0; channel < CHANNELS; channel += 1) {
      if (sourceRGBA[offset + channel] !== outputRGBA[offset + channel]) {
        changed = true;
        break;
      }
    }
    if (changed) changedPixels += 1;
  }

  return { ok: changedPixels === 0, changedPixels };
}
/**
 * Extract a normalized alpha mask from an RGBA image.
 * Transparent pixels are preserved from the original source during compositing.
 */
export function alphaMaskFromRGBA(rgba, width, height) {
  const expectedPixels = width * height;
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    throw new TypeError("width and height must be positive integers");
  }
  if (!isByteArray(rgba) || rgba.length !== expectedPixels * CHANNELS) {
    throw new TypeError("rgba must contain width * height * 4 channels");
  }

  const mask = new Float32Array(expectedPixels);
  for (let pixel = 0; pixel < expectedPixels; pixel += 1) {
    mask[pixel] = rgba[pixel * CHANNELS + 3] / 255;
  }
  return mask;
}

/** The already-feathered RGB must never be decoded with fractional alpha. */
export function binaryExtentFromRGBA(rgba, width, height) {
  const mask = alphaMaskFromRGBA(rgba, width, height);
  if (mask.some(alpha => alpha !== 0 && alpha !== 1)) {
    throw new Error("Yerel motorun kayıpsız alfa maskesi geçersiz.");
  }
  return mask;
}
