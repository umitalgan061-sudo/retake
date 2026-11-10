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

conda run --no-capture-output -n facefusion python $ServerScript --facefusion-dir $FaceFusionDir --host 127.0.0.1 --port 8765 --origin "https://retake06.web.app"
