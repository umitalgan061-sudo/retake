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
.EXAMPLE
  .\tools\update_retake_ultra.ps1 -SkipPull -SkipDeploy -Force
  (-Force stops the managed engine even while it is processing a photo.)
#>
param(
    [switch]$SkipPull,
    [switch]$SkipDeploy,
    [switch]$EnablePairing,
    [switch]$Force,
    [int]$WaitSeconds = 150
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$managedTask = $null
$managedStopped = $false
if (Get-Command Get-ScheduledTask -ErrorAction SilentlyContinue) {
    $sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $managedTask = Get-ScheduledTask -TaskName ("Retake Local Engine-" + $sid) -ErrorAction SilentlyContinue
}
if ($managedTask -and $managedTask.State -eq "Running") {
    $headers = @{}
    $tokenFile = Join-Path $HOME ".retake\pairing-token"
    if (Test-Path $tokenFile) { $headers["X-Retake-Token"] = (Get-Content $tokenFile -Raw).Trim() }
    # A just-restarted engine needs a while (CUDA, models) before /health
    # answers: wait for it instead of failing at once.
    $deadline = (Get-Date).AddSeconds([Math]::Max(0, $WaitSeconds))
    $health = $null
    $announced = ""
    while ($true) {
        try { $health = Invoke-RestMethod "http://127.0.0.1:8765/health" -Headers $headers -TimeoutSec 5 } catch { $health = $null }
        if ($health -and $health.background_mode -eq $true -and $health.busy -eq $false) { break }
        if ($Force -and $health -and $health.background_mode -eq $true) { break }
        if ((Get-Date) -gt $deadline) { break }
        $state = if (-not $health) { "starting" } elseif ($health.busy) { "busy" } else { "manual" }
        if ($state -ne $announced) {
            switch ($state) {
                "starting" { Write-Host "Motor henuz yanit vermiyor (yeni baslatildiysa modeller yukleniyor); en cok $WaitSeconds sn bekleniyor..." -ForegroundColor Yellow }
                "busy" { Write-Host "Motor su an bir fotografi isliyor; bitmesi bekleniyor..." -ForegroundColor Yellow }
                default { Write-Host "Elle acilmis bir motor penceresi calisiyor gibi gorunuyor; bekleniyor..." -ForegroundColor Yellow }
            }
            $announced = $state
        }
        Start-Sleep -Seconds 5
    }
    if (-not $health) {
        throw "Motor $WaitSeconds sn icinde yanit vermedi. Durum icin: .\tools\install_retake_autostart.ps1 -Action Status  (motor penceresindeki son satirlari da kontrol edin)"
    }
    if ($health.background_mode -ne $true) {
        throw "Elle acilmis bir Retake motoru penceresi calisiyor. O pencereyi kapatip bu komutu tekrar calistirin."
    }
    if ($health.busy -ne $false -and -not $Force) {
        throw "Motor hala bir fotografi isliyor. Bitince tekrar calistirin; isi yarida kesmek icin -Force ekleyin."
    }
    if ($health.busy -ne $false) {
        Write-Warning "-Force: islenen fotograf yarida kesiliyor."
    }
    Stop-ScheduledTask -TaskName $managedTask.TaskName
    $managedStopped = $true
}
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

    if ($managedTask) {
        # Do not start a competing manual server; the registered task keeps its options.
        Start-ScheduledTask -TaskName $managedTask.TaskName
        $managedStopped = $false
        Write-Host "Managed Retake engine restarted. Check: .\tools\install_retake_autostart.ps1 -Action Status" -ForegroundColor Green
        return
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
    if ($managedStopped) { Start-ScheduledTask -TaskName $managedTask.TaskName -ErrorAction Continue }
}
