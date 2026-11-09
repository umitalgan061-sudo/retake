export const MAX_INPUT_BYTES = 10 * 1024 * 1024;
export const MAX_OUTPUT_BYTES = 14 * 1024 * 1024;

export function decodePngDataUrl(value) {
  if (typeof value !== "string") throw new TypeError("imageDataUrl must be a string.");
  const match = /^data:image\/png;base64,([A-Za-z0-9+/]+={0,2})$/.exec(value);
  if (!match) throw new TypeError("Yalnızca PNG formatında görsel gönderilebilir.");
  const buffer = Buffer.from(match[1], "base64");
  if (!buffer.length || buffer.length > MAX_INPUT_BYTES) {
    throw new RangeError("Görsel 10 MB sınırını aşıyor veya boş.");
  }
  const pngSignature = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  if (buffer.length < 8 || !buffer.subarray(0, 8).equals(pngSignature)) {
    throw new TypeError("Görsel verisi geçerli bir PNG değil.");
  }
  return buffer;
}

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
      const data = candidate.slice(candidate.indexOf(",") + 1);
      if (Buffer.byteLength(data, "base64") > MAX_OUTPUT_BYTES) {
        throw new RangeError("NVIDIA çıktısı 14 MB sınırını aşıyor.");
      }
      return candidate;
    }
    if (/^[A-Za-z0-9+/]+={0,2}$/.test(candidate)) {
      if (Buffer.byteLength(candidate, "base64") > MAX_OUTPUT_BYTES) {
        throw new RangeError("NVIDIA çıktısı 14 MB sınırını aşıyor.");
      }
      return "data:image/png;base64," + candidate;
    }
  }
  throw new Error("NVIDIA yanıtında desteklenen base64 görsel çıktısı bulunamadı.");
}

export function describeNvidiaFailure(status, stage = "inference") {
  if (status === 401) return "NVIDIA API anahtarı geçersiz veya süresi dolmuş.";
  if (status === 403) return "NVIDIA anahtarında invoke_function yetkisi veya bu görsel uç noktasına erişim yok.";
  if (status === 402) return "NVIDIA tarafında kullanım kotası veya ödeme sınırı aşıldı.";
  if (status === 404) return "FLUX.1-Kontext-dev uç noktası bu NVIDIA hesabında etkin değil. Model kartında indirilebilir olarak listeleniyor; kullanılabilir bir NIM endpoint'i gerekebilir.";
  if (status === 413) return "Görsel NVIDIA'nın kabul ettiği boyut sınırını aşıyor.";
  if (status === 422) return "NVIDIA görsel girdisini reddetti. Bu hesap/endpoint özel görsel yüklemeyi desteklemiyor olabilir.";
  if (status === 429) return "NVIDIA istek sınırına ulaşıldı. Biraz sonra tekrar dene.";
  if (status >= 500) return "NVIDIA servisi geçici olarak hata verdi.";
  return stage === "asset" ? "NVIDIA görsel varlığı yüklenemedi." : "NVIDIA isteği başarısız oldu.";
}
