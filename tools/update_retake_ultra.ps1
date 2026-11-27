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
if ($managedTask) {
    # A registered task may still point at an older worktree. Do not stop
    # it or publish a newer frontend while claiming that engine was updated.
    $taskRoots = @($managedTask.Actions | ForEach-Object { $_.WorkingDirectory } | Where-Object { $_ })
    if ($taskRoots.Count -ne 1 -or [System.IO.Path]::GetFullPath($taskRoots[0]).TrimEnd("\") -ne $Root.TrimEnd("\")) {
        $registered = if ($taskRoots.Count) { ($taskRoots -join ", ") } else { "(none)" }
        Write-Host ("Retake startup task runs another folder: {0}" -f $registered) -ForegroundColor Yellow
        Write-Host ("This folder: {0}" -f $Root) -ForegroundColor Yellow
        Write-Host "Move the task here (settings are kept), then run this update again:" -ForegroundColor Cyan
        Write-Host "  .\tools\install_retake_autostart.ps1"
        throw "Retake startup task points to a different worktree; no restart or publish was performed."
    }
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
# Preflight manual work before changing code or deploying the site.
$bridges = @()
if (-not $managedTask) {
    $bridges = @(
        Get-CimInstance Win32_Process -ErrorAction Stop |
        Where-Object {
            $_.CommandLine -and $_.CommandLine -match 'facefusion_local_server\.py' -and $_.Name -match '^python(?:w)?\.exe$'
        }
    )
    if ($bridges.Count -gt 0) {
        $serverPath = Join-Path $PSScriptRoot "facefusion_local_server.py"
        foreach ($bridge in $bridges) {
            if ($bridge.CommandLine.IndexOf($serverPath, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
                throw "Another or unidentified Retake worktree is running; no process was stopped."
            }
        }
        if ($bridges.Count -ne 1) { throw "Multiple Retake engines found; no process was stopped." }
        $manualHeaders = @{}
        $tokenPath = Join-Path $HOME ".retake\pairing-token"
        if (Test-Path $tokenPath) { $manualHeaders["X-Retake-Token"] = (Get-Content $tokenPath -Raw).Trim() }
        try { $manualHealth = Invoke-RestMethod "http://127.0.0.1:8765/health" -Headers $manualHeaders -TimeoutSec 5 }
        catch { throw "Manual engine state cannot be verified. Close it explicitly before restarting." }
        if ($manualHealth.background_mode -eq $true) { throw "Unexpected managed engine; no process was stopped." }
        if ($manualHealth.busy -ne $false -and -not $Force) { throw "Manual engine is processing a photo or busy state is unknown. Retry when idle." }
        if ($manualHealth.busy -ne $false) { Write-Warning "-Force: manual photo processing may be interrupted." }
    }
}
function Publish-VerifiedRetakeWeb {
    if (-not $SkipDeploy) {
        if (-not (Get-Command firebase -ErrorAction SilentlyContinue)) {
            throw "Firebase CLI bulunamadi; web yayini yapilmadi."
        }
        Write-Host "Firebase Hosting retake06-ua yayinlaniyor..." -ForegroundColor Cyan
        firebase deploy --only "hosting:retake06" --project "projelerim-8fecd"
        if ($LASTEXITCODE -ne 0) { throw "Firebase yayini basarisiz; web yayini dogrulanmadi." }
        # Verify the actual delivered bytes, not the CLI success message.
        # Never send local pairing headers or secrets to the public host.
        $temp = [System.IO.Path]::GetTempFileName()
        try {
            foreach ($file in @("app.js", "src/diagnostics.js", "sw.js")) {
                $localHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $Root $file)).Hash
                $verified = $false
                for ($attempt = 0; $attempt -lt 3; $attempt++) {
                    $nonce = [Guid]::NewGuid().ToString("N")
                    try {
                        Invoke-WebRequest -UseBasicParsing -Uri ("https://retake06-ua.web.app/" + $file + "?verify=" + $nonce) -Headers @{"Cache-Control"="no-cache"} -TimeoutSec 20 -OutFile $temp
                        if ((Get-FileHash -Algorithm SHA256 -LiteralPath $temp).Hash -eq $localHash) { $verified = $true; break }
                    } catch { }
                    if ($attempt -lt 2) { Start-Sleep -Seconds 3 }
                }
                if (-not $verified) { throw "Firebase komutu basarili fakat yayindaki $file yerel dosyayla eslesmiyor." }
            }
            Write-Host "Firebase yayini dogrulandi: app.js, diagnostics.js, sw.js (SHA256)." -ForegroundColor Green
        } finally { Remove-Item -LiteralPath $temp -Force -ErrorAction SilentlyContinue }
    }
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

    if ($managedTask) {
        # Do not start a competing manual server; the registered task keeps its options.
        Start-ScheduledTask -TaskName $managedTask.TaskName
        $managedStopped = $false
        $expectedLine = Select-String -LiteralPath (Join-Path $PSScriptRoot "facefusion_local_server.py") -Pattern '^QUALITY_PIPELINE_REVISION = "([^"]+)"$' | Select-Object -First 1
        if ($expectedLine) {
            $expected = $expectedLine.Matches[0].Groups[1].Value
            $verifyHeaders = @{}
            $pairingPath = Join-Path $HOME ".retake\pairing-token"
            if (Test-Path $pairingPath) { $verifyHeaders["X-Retake-Token"] = (Get-Content $pairingPath -Raw).Trim() }
            $verifyUntil = (Get-Date).AddSeconds([Math]::Max(0, $WaitSeconds))
            $verified = $false
            do {
                try {
                    $current = Invoke-RestMethod "http://127.0.0.1:8765/health" -Headers $verifyHeaders -TimeoutSec 5
                    if ($current.ok -eq $true -and $current.background_mode -eq $true -and $current.quality_revision -eq $expected) { $verified = $true; break }
                } catch { }
                if ((Get-Date) -ge $verifyUntil) { break }
                Start-Sleep -Seconds 3
            } while ($true)
            if (-not $verified) { throw "Managed engine restart requested, but quality revision $expected was not verified. Run .\tools\install_retake_autostart.ps1 -Action Status." }
            Write-Host "Managed Retake engine verified: $expected" -ForegroundColor Green
        } else {
            throw "Managed engine restart requested, but the local quality revision declaration was not found. Verification failed."
        }
        Publish-VerifiedRetakeWeb
        return
    }

    Publish-VerifiedRetakeWeb

    # Stop just the Python command with the Retake bridge, not arbitrary
    # applications using Python, CUDA or port 8765.
    $pairFromPrevious = $false
    foreach ($bridge in $bridges) {
        if ($bridge.CommandLine -match '--enable-pairing') {
            $pairFromPrevious = $true
        }
        Write-Host ("Onceki Retake motoru durduruluyor (PID " + $bridge.ProcessId + ")...") -ForegroundColor Yellow
        $current = Invoke-RestMethod "http://127.0.0.1:8765/health" -Headers $manualHeaders -TimeoutSec 5
        if ($current.background_mode -eq $true -or ($current.busy -ne $false -and -not $Force)) { throw "Manual engine state changed; no process was stopped." }
        $same = Get-CimInstance Win32_Process -Filter ("ProcessId=" + $bridge.ProcessId) -ErrorAction Stop
        if (-not $same -or $same.CreationDate -ne $bridge.CreationDate -or $same.CommandLine -ne $bridge.CommandLine) { throw "Manual engine changed during update; no process was stopped." }
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
