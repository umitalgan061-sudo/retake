# Retake - Fotograf gercekciligi katmanini kurar (istege bagli, NVIDIA GPU gerekir).
#
# FaceFusion'in 256 px yuzunu, gercek fotograflarla egitilmis bir difuzyon modeli
# (Realistic Vision 6, SD 1.5 inpainting) kisinin kendi ArcFace kimligiyle
# (IP-Adapter FaceID) yeniden isler: gercek cilt, gozenek, dudak ve goz dokusu.
#
# - Ayri bir conda ortami kurar: retake-realism (FaceFusion ortamina dokunmaz).
# - Modelleri C:\AI\RetakeRealism\models altina indirir (~2,6 GB).
# - Toplam disk: ~8 GB. GPU bellegi: ~4 GB.
#
# Kullanim (retake klasorunde):
#   .\tools\install_realism.ps1
# Sonra motoru her zamanki gibi baslat; katman kendiliginden acilir:
#   .\tools\start_facefusion_retake.ps1

param(
    [string]$InstallDir = "C:\AI\RetakeRealism",
    [string]$EnvName = "retake-realism",
    # CUDA 12.1 tekerlekleri RTX 20/30/40 serisi ve guncel surucularla calisir.
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu121"
)

$ErrorActionPreference = "Stop"
$Models = Join-Path $InstallDir "models"
New-Item -ItemType Directory -Force -Path $Models | Out-Null

Write-Host "1/4 conda ortami: $EnvName" -ForegroundColor Cyan
$envs = conda env list | Out-String
if ($envs -notmatch "(?m)^$([regex]::Escape($EnvName))\s") {
    conda create -y -n $EnvName python=3.11
    if ($LASTEXITCODE -ne 0) { throw "conda ortami olusturulamadi." }
}

Write-Host "2/4 PyTorch (CUDA) ve diffusers kuruluyor..." -ForegroundColor Cyan
conda run --no-capture-output -n $EnvName python -m pip install --upgrade pip
conda run --no-capture-output -n $EnvName python -m pip install "torch==2.5.1" --index-url $TorchIndex
if ($LASTEXITCODE -ne 0) { throw "PyTorch kurulamadi." }
conda run --no-capture-output -n $EnvName python -m pip install `
    "diffusers==0.31.0" "transformers==4.46.3" "accelerate==1.1.1" "huggingface_hub==0.26.2" `
    "peft==0.13.2" "safetensors" "opencv-python-headless" "numpy<2.2" "pillow"
if ($LASTEXITCODE -ne 0) { throw "diffusers kurulamadi." }

Write-Host "3/4 Modeller indiriliyor (~2,6 GB)..." -ForegroundColor Cyan
$Downloads = @(
    @{ Url = "https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE/resolve/main/Realistic_Vision_V6.0_NV_B1_inpainting_fp16.safetensors";
       Path = "Realistic_Vision_V6.0_NV_B1_inpainting_fp16.safetensors" },
    @{ Url = "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors";
       Path = "sd-vae-ft-mse\diffusion_pytorch_model.safetensors" },
    @{ Url = "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json";
       Path = "sd-vae-ft-mse\config.json" },
    @{ Url = "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin";
       Path = "ip-adapter-faceid\ip-adapter-faceid_sd15.bin" }
)
foreach ($item in $Downloads) {
    $target = Join-Path $Models $item.Path
    New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
    if ((Test-Path $target) -and ((Get-Item $target).Length -gt 0)) {
        Write-Host "   zaten var: $($item.Path)"
        continue
    }
    Write-Host "   indiriliyor: $($item.Path)"
    curl.exe -L --fail --retry 3 -o "$target.part" $item.Url
    if ($LASTEXITCODE -ne 0) { throw "Indirilemedi: $($item.Url)" }
    Move-Item -Force "$target.part" $target
}

Write-Host "4/4 Model yukleme denemesi (GPU)..." -ForegroundColor Cyan
$Python = (conda run -n $EnvName python -c "import sys; print(sys.executable)").Trim()
Set-Content -Path (Join-Path $InstallDir "python.txt") -Value $Python -Encoding ascii
$Worker = Join-Path $PSScriptRoot "retake_realism_worker.py"
& $Python $Worker --check --models-dir $Models --swapped x --target x --bounds "{}" --identity-file x --out-dir x
if ($LASTEXITCODE -ne 0) { throw "Modeller yuklenemedi; yukaridaki hatayi kontrol et." }

Write-Host ""
Write-Host "Fotograf gercekciligi katmani hazir." -ForegroundColor Green
Write-Host "Motoru yeniden baslat: .\tools\start_facefusion_retake.ps1" -ForegroundColor Green
