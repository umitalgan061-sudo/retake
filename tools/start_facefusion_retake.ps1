param(
    [ValidateSet("512x512", "768x768", "1024x1024")]
    [string]$FaceSwapperPixelBoost = "1024x1024",
    [switch]$EnableFaceEnhancer,
    # Telefon / tablet / PWA'dan kullanmak için: eşleştirme kodunu zorunlu kılar.
    [switch]$EnablePairing
)

$ErrorActionPreference = "Stop"

$FaceFusionDir = "C:\AI\FaceFusion"
$ServerScript = Join-Path $PSScriptRoot "facefusion_local_server.py"

if (-not (Test-Path (Join-Path $FaceFusionDir "facefusion.py"))) {
    throw "FaceFusion bulunamadı. Önce tools\install_facefusion_windows.ps1 dosyasını çalıştır."
}
if (-not (Test-Path $ServerScript)) {
    throw "Retake FaceFusion yerel sunucu dosyası bulunamadı: $ServerScript"
}

Write-Host "Retake yerel FaceFusion servisi başlatılıyor..." -ForegroundColor Cyan
Write-Host "Adres: http://127.0.0.1:8765" -ForegroundColor Green
Write-Host "Bu PowerShell penceresini işlem sırasında açık bırak." -ForegroundColor Yellow

$ServerArgs = @(
    "--facefusion-dir", $FaceFusionDir,
    "--host", "127.0.0.1",
    "--port", "8765",
    "--face-swapper-pixel-boost", $FaceSwapperPixelBoost,
    "--origin", "https://retake06-ua.web.app",
    "--origin", "https://retake06.web.app"
)
if ($EnableFaceEnhancer) {
    $ServerArgs += "--enable-face-enhancer"
    Write-Host "GFPGAN 1.4 son aşaması açık (%25 karışım)." -ForegroundColor Yellow
}

if ($EnablePairing) {
    $ServerArgs += "--enable-pairing"
    Write-Host "Telefon/PWA eşleştirmesi açık. Ayrı bir pencerede şifreli tüneli başlat:" -ForegroundColor Yellow
    Write-Host "    cloudflared tunnel --url http://127.0.0.1:8765" -ForegroundColor Green
    Write-Host "Tünelin verdiği https adresini ve aşağıda yazacak eşleştirme kodunu Retake > Motor bağlantısı bölümüne gir." -ForegroundColor Yellow
}

conda run --no-capture-output -n facefusion python $ServerScript @ServerArgs
