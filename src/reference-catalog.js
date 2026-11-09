export const REFERENCE_CATALOG = Object.freeze([
  { id: "img_689e31cf6bf4dbf904cc", name: "Referans 01" },
  { id: "img_a5279436b1cad67fbf72", name: "Referans 02" },
  { id: "img_8b35f530d4d04fececd5", name: "Referans 03" },
  { id: "img_fd0ef2aeed5ad36af66a", name: "Referans 04" },
  { id: "img_8a5d80862b97e5b29abf", name: "Referans 05" },
  { id: "img_e2780856c9af98365f44", name: "Referans 06" },
  { id: "img_28fbb939c7a767479ac0", name: "Referans 07" },
  { id: "img_7663cd19510990048105", name: "Referans 08" },
  { id: "img_34f153bfc5f28b8dce4c", name: "Referans 09" },
  { id: "img_6fa4852739e14e713bc1", name: "Referans 10" },
  { id: "img_c80dee13885d3313c03d", name: "Referans 11" },
  { id: "img_82572738f6a6ae582036", name: "Referans 12" },
  { id: "img_7fa1796d0b09f04e1113", name: "Referans 13" },
  { id: "img_b4d56199fdba19a10ac4", name: "Referans 14" },
  { id: "img_edaa8e0cdb3fc2d938e2", name: "Referans 15" },
  { id: "img_0e35116410bec2fe5e57", name: "Referans 16" },
  { id: "img_0a9ab62887f9897d0433", name: "Referans 17" },
  { id: "img_132f5d5ac30314567824", name: "Referans 18" },
  { id: "img_1cc335bd3dba47f93694", name: "Referans 19" },
  { id: "img_ffe603155653b909c890", name: "Referans 20" }
]);

export function resolveReferenceApiBase(apiBaseUrl, origin) {
  const candidate = (typeof apiBaseUrl === "string" && apiBaseUrl.trim()) || origin;
  if (!candidate) throw new Error("Referans API adresi tanımlı değil.");
  const url = new URL(candidate);
  if (url.protocol !== "https:" && url.protocol !== "http:") {
    throw new Error("Referans API yalnızca HTTP veya HTTPS kullanabilir.");
  }
  return url.origin;
}

export function getReferenceImageUrl(id, apiBaseUrl, origin = globalThis.location?.origin) {
  if (typeof id !== "string" || !/^img_[a-f0-9]{20}$/.test(id)) {
    throw new TypeError("Geçersiz referans görsel kimliği.");
  }
  const base = resolveReferenceApiBase(apiBaseUrl, origin);
  return new URL("/api/images/" + encodeURIComponent(id), base).href;
}
