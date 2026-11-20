# Retake - Kisisel model (LoRA) egitimi. Bir kez calistirilir.
#
# 25 referans fotografinla, gercekcilik katmaninin difuzyon modelini sana ozel
# hale getirir: sakalin (yanaklara kadar), cildin, kaslarin ve dudaklarin.
# Once tools\install_realism.ps1 kurulmus olmali. NVIDIA GPU (8 GB+) gerekir.
# Sure: GPU'ya gore yaklasik 10-40 dakika.
#
# Kullanim (retake klasorunde):
#   .\tools\train_personal_model.ps1
#   .\tools\train_personal_model.ps1 -References "C:\baska\klasor"
# Fotograflarin degisirse yeniden calistir. Motoru sonra yeniden baslat.

param(
    [string]$RealismDir = "C:\AI\RetakeRealism",
    [string]$References = (Join-Path (Split-Path $PSScriptRoot -Parent) "reference-faces"),
    [int]$Steps = 0
)

$ErrorActionPreference = "Stop"
function Save-Model([string]$Url, [string]$Target) {
    # Windows curl (schannel) refuses when the certificate revocation list
    # cannot be reached (CRYPT_E_NO_REVOCATION_CHECK, common behind company
    # networks); --ssl-no-revoke skips only that check. PowerShell's own
    # downloader is the fallback. The file must really arrive.
    $part = "$Target.part"
    Remove-Item -Force -ErrorAction SilentlyContinue $part
    curl.exe -L --fail --retry 3 --ssl-no-revoke -o $part $Url
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $part) -or (Get-Item $part).Length -eq 0) {
        Write-Host "   curl ile indirilemedi, PowerShell ile deneniyor..." -ForegroundColor Yellow
        Remove-Item -Force -ErrorAction SilentlyContinue $part
        $previous = $ProgressPreference
        $ProgressPreference = "SilentlyContinue"
        try {
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            Invoke-WebRequest -Uri $Url -OutFile $part -UseBasicParsing
        } catch {
            throw "Indirilemedi: $Url ($($_.Exception.Message))"
        } finally {
            $ProgressPreference = $previous
        }
    }
    if (-not (Test-Path $part) -or (Get-Item $part).Length -eq 0) { throw "Indirilemedi: $Url" }
    Move-Item -Force $part $Target
}

$PythonFile = Join-Path $RealismDir "python.txt"
if (-not (Test-Path $PythonFile)) {
    throw "Gercekcilik katmani kurulu degil. Once: .\tools\install_realism.ps1"
}
$Python = (Get-Content $PythonFile -Raw).Trim()
$Models = Join-Path $RealismDir "models"
if (-not (Test-Path $References)) { throw "Referans klasoru bulunamadi: $References" }

$Base = Join-Path $Models "Realistic_Vision_V6.0_NV_B1_fp16.safetensors"
if (-not (Test-Path $Base)) {
    Write-Host "Temel model indiriliyor (~2,1 GB)..." -ForegroundColor Cyan
    Save-Model "https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE/resolve/main/Realistic_Vision_V6.0_NV_B1_fp16.safetensors" $Base
}

# OpenCV 5 dropped the classic face detector the trainer uses; also make
# sure the packages added after the first install are there.
Write-Host "Paketler kontrol ediliyor..." -ForegroundColor Cyan
& $Python -m pip install --quiet "opencv-python-headless<5" "peft==0.13.2" "truststore"
if ($LASTEXITCODE -ne 0) { throw "Gerekli paketler kurulamadi." }

$Trainer = Join-Path $PSScriptRoot "retake_personal_model.py"
$TrainArgs = @("--models-dir", $Models, "--references", $References)
if ($Steps -gt 0) { $TrainArgs += @("--steps", $Steps) }
Write-Host "Kisisel model egitiliyor: $References" -ForegroundColor Cyan
& $Python $Trainer @TrainArgs
if ($LASTEXITCODE -ne 0) { throw "Egitim basarisiz; yukaridaki hatayi kontrol et." }

Write-Host ""
Write-Host "Kisisel model hazir. Motoru yeniden baslat: .\tools\start_facefusion_retake.ps1" -ForegroundColor Green
