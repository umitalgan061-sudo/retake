const DATABASE_NAME = "retake-local-face-profile";
const DATABASE_VERSION = 1;
const STORE_NAME = "profiles";
const ACTIVE_PROFILE_ID = "active";

function openDatabase() {
  return new Promise((resolve, reject) => {
    if (!("indexedDB" in globalThis)) {
      reject(new Error("Bu tarayıcı IndexedDB yerel profil saklamasını desteklemiyor."));
      return;
    }

    const request = globalThis.indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("Yerel profil veritabanı açılamadı."));
    request.onblocked = () => reject(new Error("Yerel profil veritabanı başka bir sekmede kilitli."));
  });
}

function toHex(buffer) {
  return Array.from(new Uint8Array(buffer), (value) => value.toString(16).padStart(2, "0")).join("");
}

export async function createReferenceSetKey(references) {
  if (!Array.isArray(references) || references.length === 0) {
    throw new TypeError("Profil anahtarı için referans listesi boş olamaz.");
  }

  const parts = [];
  for (const reference of references) {
    if (!reference?.file || !reference.blob || typeof reference.blob.size !== "number") {
      throw new TypeError("Her referans dosya adı ve Blob içermeli.");
    }

    let fingerprint = `${reference.file}:${reference.blob.size}:${reference.blob.type || "image"}`;
    if (globalThis.crypto?.subtle && typeof reference.blob.arrayBuffer === "function") {
      const bytes = await reference.blob.arrayBuffer();
      const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
      fingerprint = `${reference.file}:${toHex(digest)}`;
    }
    parts.push(fingerprint);
  }

  const source = parts.join("|");
  if (globalThis.crypto?.subtle) {
    const digest = await globalThis.crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(source)
    );
    return toHex(digest);
  }
  // Fallback still invalidates when the manifest file names or byte sizes change.
  return source;
}

export async function getCachedProfile(key) {
  if (typeof key !== "string" || !key) return null;
  let database;
  try {
    database = await openDatabase();
    return await new Promise((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, "readonly");
      const request = transaction.objectStore(STORE_NAME).get(ACTIVE_PROFILE_ID);
      request.onsuccess = () => {
        const record = request.result;
        resolve(record?.key === key ? record : null);
      };
      request.onerror = () => reject(request.error || new Error("Yerel profil okunamadı."));
      transaction.onabort = () => reject(transaction.error || new Error("Yerel profil okuma işlemi iptal edildi."));
    });
  } catch (error) {
    console.warn("Yerel profil önbelleği okunamadı; profil yeniden oluşturulacak.", error);
    return null;
  } finally {
    database?.close();
  }
}

export async function saveCachedProfile(key, profile) {
  if (typeof key !== "string" || !key || !profile || typeof profile !== "object") return false;
  let database;
  try {
    database = await openDatabase();
    await new Promise((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, "readwrite");
      transaction.objectStore(STORE_NAME).put({
        id: ACTIVE_PROFILE_ID,
        key,
        ...profile,
        updatedAt: Date.now()
      });
      transaction.oncomplete = resolve;
      transaction.onerror = () => reject(transaction.error || new Error("Yerel profil kaydedilemedi."));
      transaction.onabort = () => reject(transaction.error || new Error("Yerel profil kaydetme işlemi iptal edildi."));
    });

    try {
      await globalThis.navigator?.storage?.persist?.();
    } catch {
      // Browser-managed persistence is optional; IndexedDB remains usable.
    }
    return true;
  } catch (error) {
    console.warn("Profil bu oturumda kullanılacak ancak IndexedDB'ye kaydedilemedi.", error);
    return false;
  } finally {
    database?.close();
  }
}
