param(
    [ValidateSet("512x512", "768x768", "1024x1024")]
    [string]$FaceSwapperPixelBoost = "1024x1024",
    [switch]$EnableFaceEnhancer,
    # Telefon / tablet / PWA'dan kullanmak için: eşleştirme kodunu zorunlu kılar.
    [switch]$EnablePairing,
    # Asıl fotoğrafın ışık-gölge kabartmasının yeni yüze aktarılma gücü (0 kapalı, 0.7 varsayılan, 1.2 en çok).
    [double]$ShadingStrength = -1,
    # Fotoğraf gerçekçiliği katmanını (tools\install_realism.ps1) kurulu olsa bile kapat.
    [switch]$DisableRealism,
    [string]$RealismDir = "C:\AI\RetakeRealism"
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

if ($ShadingStrength -ge 0) {
    $ServerArgs += @("--shading-strength", $ShadingStrength.ToString([System.Globalization.CultureInfo]::InvariantCulture))
    Write-Host "Gölgelendirme aktarımı gücü: $ShadingStrength" -ForegroundColor Yellow
}
$RealismPython = Join-Path $RealismDir "python.txt"
if (-not $DisableRealism -and (Test-Path $RealismPython)) {
    $ServerArgs += @(
        "--realism-python", (Get-Content $RealismPython -Raw).Trim(),
        "--realism-models", (Join-Path $RealismDir "models")
    )
    Write-Host "Fotoğraf gerçekçiliği katmanı açık (difüzyon + kimlik yönlendirmesi)." -ForegroundColor Green
} elseif (-not $DisableRealism) {
    Write-Host "İpucu: gerçekçi cilt/dudak/göz için tek seferlik kurulum: .\tools\install_realism.ps1" -ForegroundColor Yellow
}
if ($EnablePairing) {
    $ServerArgs += "--enable-pairing"
    Write-Host "Telefon/PWA eşleştirmesi açık. Ayrı bir pencerede şifreli tüneli başlat:" -ForegroundColor Yellow
    Write-Host "    cloudflared tunnel --url http://127.0.0.1:8765" -ForegroundColor Green
    Write-Host "Tünelin verdiği https adresini ve aşağıda yazacak eşleştirme kodunu Retake > Motor bağlantısı bölümüne gir." -ForegroundColor Yellow
}

conda run --no-capture-output -n facefusion python $ServerScript @ServerArgs
