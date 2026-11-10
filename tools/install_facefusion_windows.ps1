$ErrorActionPreference = "Stop"

$FaceFusionDir = "C:\AI\FaceFusion"
$FaceFusionVersion = "3.9.2"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda bulunamadı. Anaconda Prompt veya Miniconda kurulumunu açıp bu betiği tekrar çalıştır."
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git bulunamadı. Git for Windows kurulduktan sonra bu betiği tekrar çalıştır."
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "FaceFusion için FFmpeg kuruluyor..." -ForegroundColor Cyan
        winget install -e --id Gyan.FFmpeg --source winget --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -ne 0) {
            throw "FFmpeg kurulumu başarısız. Winget 'winget' kaynağında Gyan.FFmpeg paketini kuramadı."
        }

        # WinGet may update the user or machine PATH after this PowerShell session started.
        $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        $env:Path = (@($machinePath, $userPath) -join ";")

        if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
            throw "FFmpeg kuruldu ancak ffmpeg.exe PATH üzerinde bulunamadı. PowerShell'i tamamen kapatıp yeniden aç, ardından bu betiği tekrar çalıştır."
        }
    } else {
        throw "FFmpeg bulunamadı ve winget kullanılamıyor. Önce Gyan.FFmpeg kur, yeni PowerShell aç ve tekrar dene."
    }
}

New-Item -ItemType Directory -Force -Path "C:\AI" | Out-Null

$envList = conda env list --json | ConvertFrom-Json
$faceFusionEnvExists = $false
foreach ($envPath in $envList.envs) {
    if ((Split-Path $envPath -Leaf) -eq "facefusion") {
        $faceFusionEnvExists = $true
        break
    }
}
if (-not $faceFusionEnvExists) {
    Write-Host "FaceFusion için ayrı Python 3.12 ortamı oluşturuluyor..." -ForegroundColor Cyan
    conda create -y -n facefusion python=3.12 pip=25.0
    if ($LASTEXITCODE -ne 0) { throw "FaceFusion Conda ortamı oluşturulamadı." }
}

if (-not (Test-Path (Join-Path $FaceFusionDir ".git"))) {
    if (Test-Path $FaceFusionDir) {
        throw "$FaceFusionDir klasörü var ancak Git deposu değil. Klasörü yedekleyip yeniden adlandır."
    }
    Write-Host "FaceFusion $FaceFusionVersion indiriliyor..." -ForegroundColor Cyan
    git clone --branch $FaceFusionVersion --depth 1 https://github.com/facefusion/facefusion.git $FaceFusionDir
    if ($LASTEXITCODE -ne 0) { throw "FaceFusion deposu indirilemedi." }
} else {
    Push-Location $FaceFusionDir
    try {
        git fetch --tags --force
        if ($LASTEXITCODE -ne 0) { throw "FaceFusion etiketleri indirilemedi." }
        git checkout $FaceFusionVersion
        if ($LASTEXITCODE -ne 0) { throw "FaceFusion $FaceFusionVersion sürümüne geçilemedi." }
    } finally {
        Pop-Location
    }
}

Write-Host "CUDA 12.9 runtime ve cuDNN 9.10 kuruluyor..." -ForegroundColor Cyan
conda install -y -n facefusion nvidia/label/cuda-12.9.1::cuda-runtime nvidia/label/cudnn-9.10.0::cudnn
if ($LASTEXITCODE -ne 0) { throw "CUDA runtime veya cuDNN kurulamadı." }

Write-Host "FaceFusion CUDA bağımlılıkları kuruluyor..." -ForegroundColor Cyan
Push-Location $FaceFusionDir
try {
    conda run --no-capture-output -n facefusion python install.py cuda@12
    if ($LASTEXITCODE -ne 0) { throw "FaceFusion CUDA bağımlılıkları kurulamadı." }
} finally {
    Pop-Location
}

Write-Host "CUDA sağlayıcısı kontrol ediliyor..." -ForegroundColor Cyan
conda run --no-capture-output -n facefusion python -c "import onnxruntime as ort; print('ONNX Runtime:', ort.__version__); print('Providers:', ort.get_available_providers()); assert 'CUDAExecutionProvider' in ort.get_available_providers(), 'CUDA provider bulunamadı'"
if ($LASTEXITCODE -ne 0) {
    throw "FaceFusion CUDA kontrolü başarısız. Ekrandaki kurulum kayıtlarını incele."
}

Write-Host ""
Write-Host "Kurulum tamamlandı. FaceFusion klasörü: $FaceFusionDir" -ForegroundColor Green
Write-Host "Model ağırlıkları ilk yüz değiştirme işleminde otomatik indirilebilir." -ForegroundColor Yellow
Write-Host "Başlatmak için Retake deposundaki tools\start_facefusion_retake.ps1 dosyasını çalıştır." -ForegroundColor Cyan
