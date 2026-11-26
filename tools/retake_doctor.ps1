<#
.SYNOPSIS
  Retake tani raporu: motor, acilis gorevi, GPU, disk, paketler, NVIDIA jurisi,
  yayindaki site ve motor olay kaydini tek komutla denetler.
.DESCRIPTION
  Rapor fotograf, dosya adi, eslestirme kodu, Hafize jetonu, kullanici adi,
  bilgisayar adi veya yerel yol icermez; panoya kopyalanir, sohbete
  yapistirilabilir. Hicbir ayari degistirmez, hicbir sey indirmez/kurmaz.
.EXAMPLE
  .\tools\retake_doctor.ps1
.EXAMPLE
  .\tools\retake_doctor.ps1 -ProbeJury
  (Hafize jurisine iki sentetik renk karti gonderir; fotograf gonderilmez.)
.EXAMPLE
  .\tools\retake_doctor.ps1 -Offline
  (GitHub, Hafize ve siteye baglanmadan yalniz bu bilgisayari denetler.)
#>
param(
    [switch]$ProbeJury,
    [switch]$Offline,
    [switch]$Json,
    [switch]$NoClipboard
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RetakeDir = Join-Path $HOME ".retake"
$Doctor = Join-Path $PSScriptRoot "retake_doctor.py"
if (-not (Test-Path -LiteralPath $Doctor)) { throw "retake_doctor.py bulunamadi: once git pull --ff-only origin main" }

# Windows-only facts, collected here and passed as JSON. Only booleans,
# states and numbers: no command lines, paths or user names.
$facts = [ordered]@{ powershell = [string]$PSVersionTable.PSVersion }
try {
    if (Get-Command Get-ScheduledTask -ErrorAction SilentlyContinue) {
        $sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
        $task = Get-ScheduledTask -TaskName ("Retake Local Engine-" + $sid) -ErrorAction SilentlyContinue
        if ($task) {
            $info = Get-ScheduledTaskInfo -TaskName $task.TaskName -ErrorAction SilentlyContinue
            $dirs = @($task.Actions | ForEach-Object { $_.WorkingDirectory } | Where-Object { $_ })
            $sameRoot = ($dirs.Count -eq 1) -and ([System.IO.Path]::GetFullPath($dirs[0]).TrimEnd("\") -eq $Root.TrimEnd("\"))
            $last = $null
            if ($info) { $last = [int64]$info.LastTaskResult }
            $facts.task = [ordered]@{ exists = $true; state = [string]$task.State; last_result = $last; working_dir_matches = [bool]$sameRoot }
        } else {
            $facts.task = [ordered]@{ exists = $false }
        }
    }
} catch { }
try {
    if (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) {
        $listener = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($listener) {
            $owner = Get-CimInstance Win32_Process -Filter ("ProcessId=" + [int]$listener.OwningProcess) -ErrorAction SilentlyContinue
            $isRetake = [bool]($owner -and $owner.CommandLine -and ($owner.CommandLine -match 'facefusion_local_server\.py|retake_background\.py'))
            $facts.port = [ordered]@{ listening = $true; owner_is_retake = $isRetake }
        } else {
            $facts.port = [ordered]@{ listening = $false }
        }
    }
} catch { }

# Prefer the engine's own Python (its packages are what matters).
$python = $null
$useConda = $false
$configPath = Join-Path $RetakeDir "background-config.json"
if (Test-Path -LiteralPath $configPath) {
    try {
        $cfg = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
        if ($cfg.python -and (Test-Path -LiteralPath ([string]$cfg.python))) { $python = [string]$cfg.python }
    } catch { }
}
if (-not $python -and (Get-Command conda -ErrorAction SilentlyContinue)) { $useConda = $true }
if (-not $python -and -not $useConda) {
    foreach ($name in @("python", "py")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) { $python = $command.Source; break }
    }
}
if (-not $python -and -not $useConda) { throw "Python bulunamadi. Conda PowerShell penceresinden calistirin." }

$stamp = [guid]::NewGuid().ToString("N")
$factsPath = Join-Path ([System.IO.Path]::GetTempPath()) ("retake-doctor-" + $stamp + ".json")
$reportPath = Join-Path ([System.IO.Path]::GetTempPath()) ("retake-doctor-" + $stamp + ".txt")
$facts | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $factsPath -Encoding UTF8
$doctorArgs = @($Doctor, "--root", $Root, "--windows-facts", $factsPath, "--output", $reportPath)
if ($Offline) { $doctorArgs += "--offline" }
if ($ProbeJury) { $doctorArgs += "--probe-jury" }
if ($Json) { $doctorArgs += "--json" }
$env:PYTHONIOENCODING = "utf-8"
try {
    Write-Host "Retake tani raporu hazirlaniyor (Hafize uyaniyorsa 1-2 dk surebilir)..." -ForegroundColor Cyan
    if ($useConda) {
        & conda run --no-capture-output -n facefusion python @doctorArgs | Out-Null
    } else {
        & $python @doctorArgs | Out-Null
    }
    $exitCode = $LASTEXITCODE
    if (-not (Test-Path -LiteralPath $reportPath)) { throw "Tani araci rapor uretemedi (cikis kodu $exitCode)." }
    $text = [System.IO.File]::ReadAllText($reportPath, [System.Text.Encoding]::UTF8)
    try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
    Write-Host $text
    if (-not $NoClipboard) {
        try {
            Set-Clipboard -Value $text
            Write-Host "Rapor panoya kopyalandi; sohbete yapistirabilirsiniz." -ForegroundColor Green
        } catch {
            Write-Host "Panoya kopyalanamadi; yukaridaki metni secip kopyalayin." -ForegroundColor Yellow
        }
    }
} finally {
    Remove-Item -LiteralPath $factsPath, $reportPath -ErrorAction SilentlyContinue
}
