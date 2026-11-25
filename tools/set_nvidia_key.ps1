<#
.SYNOPSIS
  Retake NVIDIA gorsel hakemini test eder; anahtar yoksa guvenle kaydeder.
.DESCRIPTION
  Motor NVIDIA_API_KEY anahtarini su sirayla arar: ortam degiskeni, Retake
  klasorundeki .env, functions\.env, .dev.vars (hepsi git disinda) ve
  RETAKE_ENV_FILE ile gosterilen dosya. Anahtar bulunursa yalnizca sentetik
  (kisisel olmayan) bir goruntuyle test edilir. Bulunamazsa gizli girisle
  sorulur ve Windows kullanici ortam degiskeni olarak saklanir. Anahtar
  ekrana, depoya veya loglara yazilmaz. Sonra arka plandaki motor yeniden
  baslatilir.
.EXAMPLE
  .\tools\set_nvidia_key.ps1
.EXAMPLE
  .\tools\set_nvidia_key.ps1 -Replace
#>
param([switch]$Replace)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Judge = Join-Path $Root "tools\nvidia_judge.py"

function Test-Judge {
    Write-Host "NVIDIA görsel hakemi sentetik bir test görüntüsüyle deneniyor (kişisel fotoğraf gönderilmez)..." -ForegroundColor Cyan
    Push-Location $Root
    try { conda run --no-capture-output -n facefusion python $Judge } finally { Pop-Location }
    return $LASTEXITCODE
}

$result = 2
if (-not $Replace) { $result = Test-Judge }
if ($result -eq 2) {
    Write-Host "build.nvidia.com > API Key ile aldığın anahtarı (nvapi-...) yapıştır. Ekranda görünmez." -ForegroundColor Cyan
    $secure = Read-Host -AsSecureString "NVIDIA API anahtarı"
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer).Trim() }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
    if (-not $key) { throw "Anahtar boş." }
    [Environment]::SetEnvironmentVariable("NVIDIA_API_KEY", $key, "User")
    $env:NVIDIA_API_KEY = $key
    Remove-Variable key
    Write-Host "Kaydedildi (Windows kullanıcı ortam değişkeni NVIDIA_API_KEY)." -ForegroundColor Green
    $result = Test-Judge
}
if ($result -ne 0) {
    Write-Host "Test başarısız. Retake hakemsiz çalışmaya devam eder; yukarıdaki mesajı Claude'a gönder." -ForegroundColor Yellow
}

if (Get-Command Get-ScheduledTask -ErrorAction SilentlyContinue) {
    $sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $task = Get-ScheduledTask -TaskName ("Retake Local Engine-" + $sid) -ErrorAction SilentlyContinue
    if ($task) {
        if ($task.State -eq "Running") { Stop-ScheduledTask -TaskName $task.TaskName; Start-Sleep -Seconds 3 }
        Start-ScheduledTask -TaskName $task.TaskName
        Write-Host "Arka plandaki Retake motoru yeniden başlatıldı." -ForegroundColor Green
        exit $result
    }
}
Write-Host "Motoru elle çalıştırıyorsan pencereyi kapatıp yeniden başlat: .\tools\start_facefusion_retake.ps1" -ForegroundColor Yellow
exit $result
