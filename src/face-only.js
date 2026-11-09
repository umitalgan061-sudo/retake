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
    if (alpha === 1) {
      output.set(generatedRGBA.subarray(offset, offset + CHANNELS), offset);
      continue;
    }

    const inverse = 1 - alpha;
    for (let channel = 0; channel < CHANNELS; channel += 1) {
      output[offset + channel] = Math.round(
        sourceRGBA[offset + channel] * inverse + generatedRGBA[offset + channel] * alpha
      );
    }
  }
  return output;
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