import { readFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const varsPath = path.join(root, ".dev.vars");

function readNvidiaKey(content) {
  const line = content.split(/\r?\n/).find((entry) => /^\s*NVIDIA_API_KEY\s*=/.test(entry));
  if (!line) throw new Error(".dev.vars dosyasında NVIDIA_API_KEY bulunamadı.");
  let value = line.slice(line.indexOf("=") + 1).trim();
  if ((value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))) {
    value = value.slice(1, -1);
  }
  if (!/^nvapi-[A-Za-z0-9_-]{12,}$/.test(value)) {
    throw new Error("NVIDIA_API_KEY biçimi geçersiz. NVIDIA API anahtarını .dev.vars dosyasına tek satır olarak yaz.");
  }
  return value;
}

function run(command, args, input) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: root,
      stdio: [input === undefined ? "inherit" : "pipe", "inherit", "inherit"],
      shell: process.platform === "win32"
    });
    child.once("error", reject);
    child.once("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error("Komut başarısız oldu: " + command + " (kod " + code + ")."));
    });
    if (input !== undefined) child.stdin.end(input);
  });
}

let apiKey;
try {
  const vars = await readFile(varsPath, "utf8");
  apiKey = readNvidiaKey(vars);
} catch (error) {
  console.error(error.message);
  console.error("Önce depo kökünde .dev.vars oluştur; .dev.vars.example şablonunu kullan.");
  process.exit(1);
}

const npx = process.platform === "win32" ? "npx.cmd" : "npx";

try {
  console.log("Cloudflare Worker kodu dağıtılıyor...");
  await run(npx, ["--yes", "wrangler@latest", "deploy"]);
  console.log("NVIDIA anahtarı Cloudflare Secret alanına aktarılıyor...");
  await run(npx, ["--yes", "wrangler@latest", "secret", "put", "NVIDIA_API_KEY"], apiKey + "\n");
  console.log("Tamamlandı. Yerel .dev.vars dosyasındaki anahtar Cloudflare Secret'a aktarıldı.");
} catch (error) {
  console.error(error.message);
  process.exit(1);
} finally {
  apiKey = "";
}
