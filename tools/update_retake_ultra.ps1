<#
.SYNOPSIS
  Bring Retake's local FaceFusion bridge and Firebase web hosting up to date.
.DESCRIPTION
  Run this from an up-to-date Retake worktree. Downloads main,
  publishes Firebase hosting if authenticated, then *restarts only* the
  facefusion_local_server.py process owned by the Retake app.
  Existing pairing mode is preserved when detectable.
  This script does not install CUDA, FaceFusion or personal LoRA weights.
.EXAMPLE
  Set-ExecutionPolicy -Scope Process Bypass
  .\tools\update_retake_ultra.ps1
.EXAMPLE
  .\tools\update_retake_ultra.ps1 -SkipDeploy
#>
param(
    [switch]$SkipPull,
    [switch]$SkipDeploy,
    [switch]$EnablePairing
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $Root
try {
    if (-not (Test-Path ".git")) {
        throw "Bu dosyayi Retake GitHub deposu icinden calistir: $Root"
    }
    if (-not $SkipPull) {
        Write-Host "Retake GitHub main guncellemesi..." -ForegroundColor Cyan
        git pull --ff-only origin main
        if ($LASTEXITCODE -ne 0) {
            throw "Git pull basarisiz. Yerel degisikliklerinizi kontrol edin; otomatik overwrite yapilmadi."
        }
    }

    if (-not $SkipDeploy) {
        if (Get-Command firebase -ErrorAction SilentlyContinue) {
            Write-Host "Firebase Hosting retake06-ua yayinlaniyor..." -ForegroundColor Cyan
            firebase deploy --only "hosting:retake06" --project "projelerim-8fecd"
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Firebase yayini basarisiz. firebase login ile oturum acin, komutu tekrar calistirin."
            } else {
                Write-Host "Firebase Hosting yayini tamamlandi." -ForegroundColor Green
            }
        } else {
            Write-Warning "Firebase CLI bulunamadi; web yayini ATLANDI. npm install -g firebase-tools; firebase login"
        }
    }

    # Stop just the Python command with the Retake bridge, not arbitrary
    # applications using Python, CUDA or port 8765.
    $bridges = @(
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -and $_.CommandLine -match 'facefusion_local_server\.py'
        }
    )
    $pairFromPrevious = $false
    foreach ($bridge in $bridges) {
        if ($bridge.CommandLine -match '--enable-pairing') {
            $pairFromPrevious = $true
        }
        Write-Host ("Onceki Retake motoru durduruluyor (PID " + $bridge.ProcessId + ")...") -ForegroundColor Yellow
        Stop-Process -Id $bridge.ProcessId -ErrorAction Stop
    }
    if ($bridges.Count -gt 0) {
        Start-Sleep -Seconds 2
    }

    $StartScript = Join-Path $PSScriptRoot "start_facefusion_retake.ps1"
    if (-not (Test-Path $StartScript)) {
        throw "Baslatma betigi bulunamadi: $StartScript"
    }
    Write-Host "Retake Ultra-420 guncel FaceFusion servisi baslatiliyor..." -ForegroundColor Green
    Write-Host "Bu PowerShell penceresini kapatmayin. Ayrica http://127.0.0.1:8765/health kontrol edin." -ForegroundColor Yellow
    if ($EnablePairing -or $pairFromPrevious) {
        & $StartScript -EnablePairing
    } else {
        & $StartScript
    }
} finally {
    Pop-Location
}
