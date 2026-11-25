<#
.SYNOPSIS
  Retake icin NVIDIA API anahtarini bu bilgisayara kaydeder (tek seferlik).
.DESCRIPTION
  Anahtar yalnizca Windows kullanici ortam degiskeni NVIDIA_API_KEY olarak
  saklanir; dosyaya, depoya veya loglara yazilmaz. Ardindan sentetik (kisisel
  olmayan) bir test goruntusuyle NVIDIA gorsel hakemi denenir ve arka planda
  calisan Retake motoru yeniden baslatilir.
  Anahtar zaten tanimliysa yalnizca test ve yeniden baslatma yapilir.
.EXAMPLE
  .\tools\set_nvidia_key.ps1
.EXAMPLE
  .\tools\set_nvidia_key.ps1 -Replace
#>
param([switch]$Replace)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$existing = [Environment]::GetEnvironmentVariable("NVIDIA_API_KEY", "User")
if (-not $existing) { $existing = [Environment]::GetEnvironmentVariable("NVIDIA_API_KEY", "Machine") }
if ($existing -and -not $Replace) {
    Write-Host "NVIDIA_API_KEY bu bilgisayarda zaten tanımlı; değiştirmek için -Replace kullan." -ForegroundColor Green
    $env:NVIDIA_API_KEY = $existing
} else {
    Write-Host "build.nvidia.com > API Key ile aldığın anahtarı (nvapi-...) yapıştır. Ekranda görünmez." -ForegroundColor Cyan
    $secure = Read-Host -AsSecureString "NVIDIA API anahtarı"
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer).Trim() }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
    if (-not $key) { throw "Anahtar boş." }
    if (-not $key.StartsWith("nvapi-")) {
        Write-Host "Uyarı: anahtar genelde 'nvapi-' ile başlar; yine de kaydediliyor." -ForegroundColor Yellow
    }
    [Environment]::SetEnvironmentVariable("NVIDIA_API_KEY", $key, "User")
    $env:NVIDIA_API_KEY = $key
    Remove-Variable key
    Write-Host "Kaydedildi (Windows kullanıcı ortam değişkeni NVIDIA_API_KEY)." -ForegroundColor Green
}

Write-Host "NVIDIA görsel hakemi sentetik bir test görüntüsüyle deneniyor (kişisel fotoğraf gönderilmez)..." -ForegroundColor Cyan
conda run --no-capture-output -n facefusion python (Join-Path $Root "tools\nvidia_judge.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "Test başarısız. Retake hakemsiz çalışmaya devam eder; yukarıdaki mesajı Claude'a gönder." -ForegroundColor Yellow
}

if (Get-Command Get-ScheduledTask -ErrorAction SilentlyContinue) {
    $sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $task = Get-ScheduledTask -TaskName ("Retake Local Engine-" + $sid) -ErrorAction SilentlyContinue
    if ($task) {
        if ($task.State -eq "Running") { Stop-ScheduledTask -TaskName $task.TaskName; Start-Sleep -Seconds 3 }
        Start-ScheduledTask -TaskName $task.TaskName
        Write-Host "Arka plandaki Retake motoru yeniden başlatıldı; hakem artık açık." -ForegroundColor Green
        exit 0
    }
}
Write-Host "Motoru elle çalıştırıyorsan pencereyi kapatıp yeniden başlat: .\tools\start_facefusion_retake.ps1" -ForegroundColor Yellow
