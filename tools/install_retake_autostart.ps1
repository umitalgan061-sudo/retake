# Register a per-user, invisible Retake engine. No elevation or cloud service.
param(
    [ValidateSet("Install","Status","Remove","Start")][string]$Action = "Install",
    [ValidateSet("512x512","768x768","1024x1024")][string]$FaceSwapperPixelBoost = "1024x1024",
    [switch]$EnablePairing,
    [switch]$EnableFaceEnhancer,
    [switch]$DisableRealism,
    [ValidateRange(-1,1.2)][double]$ShadingStrength = -1,
    [string]$RealismDir = "C:\AI\RetakeRealism",
    [string]$FaceFusionDir = "C:\AI\FaceFusion",
    [string]$NimEditors = $env:RETAKE_NIM_EDITORS
)
$ErrorActionPreference = "Stop"
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
$taskName = "Retake Local Engine-" + $identity.User.Value
$directory = Join-Path $HOME ".retake"
$configPath = Join-Path $directory "background-config.json"
$statusPath = Join-Path $directory "background-status.json"
$runner = Join-Path $PSScriptRoot "retake_background.py"
function Get-EngineHealth {
    $headers = @{}
    $tokenFile = Join-Path $directory "pairing-token"
    if (Test-Path $tokenFile) {
        $headers["X-Retake-Token"] = (Get-Content $tokenFile -Raw).Trim()
    }
    try { Invoke-RestMethod -Uri "http://127.0.0.1:8765/health" -Headers $headers -TimeoutSec 3 }
    catch { $null }
}
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($Action -eq "Status") {
    if ($task) {
        $info = Get-ScheduledTaskInfo -TaskName $taskName
        [pscustomobject]@{ Task=$taskName; State=[string]$task.State; LastTaskResult=$info.LastTaskResult }
    } else { Write-Host "Startup task not installed." }
    if (Test-Path $statusPath) { Get-Content $statusPath -Raw }
    $health = Get-EngineHealth
    if ($health) { $health | ConvertTo-Json -Depth 6 } else { Write-Host "Engine health unavailable." }
    return
}
if ($Action -eq "Start") {
    if (-not $task) { throw "Install the startup task first." }
    Start-ScheduledTask -TaskName $taskName
    return
}
if ($Action -eq "Remove") {
    if ($task) {
        $health = Get-EngineHealth
        if ($task.State -eq "Running" -and (-not $health -or $health.busy -ne $false)) {
            throw "Cannot confirm the engine is idle. Do not remove while a photo may be processing."
        }
        if ($task.State -eq "Running") { Stop-ScheduledTask -TaskName $taskName }
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-Host "Startup task removed. Model weights and reference files were preserved."
    }
    return
}
if (-not (Test-Path $runner)) { throw "Retake background launcher missing." }
$old = $null
if (Test-Path $configPath) { try { $old = Get-Content $configPath -Raw | ConvertFrom-Json } catch { $old = $null } }
# Re-registering (for example from another Retake folder) keeps the existing
# FaceFusion and realism folders unless new ones are given explicitly.
if (-not $PSBoundParameters.ContainsKey("FaceFusionDir") -and $old -and $old.facefusion_dir -and (Test-Path (Join-Path ([string]$old.facefusion_dir) "facefusion.py"))) {
    $FaceFusionDir = [string]$old.facefusion_dir
}
if (-not $PSBoundParameters.ContainsKey("RealismDir") -and $old -and $old.realism_dir -and (Test-Path -LiteralPath ([string]$old.realism_dir))) {
    $RealismDir = [string]$old.realism_dir
}
if (-not (Test-Path (Join-Path $FaceFusionDir "facefusion.py"))) { throw "FaceFusion is not installed at $FaceFusionDir" }
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) { throw "Open your existing Conda PowerShell and run this script there." }
# Resolve the exact working Python and activated PATH once; Task Scheduler
# does not load the user's Conda PowerShell profile.
$probe = @(& conda run --no-capture-output -n facefusion python -c "import json,os,sys,onnxruntime as ort; print(json.dumps(dict(python=sys.executable,path=os.environ['PATH'],cuda='CUDAExecutionProvider' in ort.get_available_providers())))")
if ($LASTEXITCODE -ne 0 -or -not $probe.Count) { throw "FaceFusion Conda environment probe failed." }
$envInfo = $probe[-1] | ConvertFrom-Json
if (-not $envInfo.cuda) { throw "CUDA provider unavailable in the FaceFusion environment." }
$python = [string]$envInfo.python
$pythonw = Join-Path (Split-Path $python -Parent) "pythonw.exe"
if (-not (Test-Path $pythonw)) { throw "pythonw.exe not found in FaceFusion environment." }
$health = Get-EngineHealth
if (-not $PSBoundParameters.ContainsKey("EnablePairing")) {
    $EnablePairing = [bool]($old.pairing -or $health.paired)
    if (-not $EnablePairing) {
        # Preserve an existing paired manual bridge even if its health is
        # inaccessible without the token. Do not print the token or command.
        $bridge = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
            $_.CommandLine -and $_.CommandLine -match 'facefusion_local_server\.py' -and $_.CommandLine -match '--enable-pairing'
        })
        $EnablePairing = [bool]$bridge.Count
    }
}
if (-not $PSBoundParameters.ContainsKey("FaceSwapperPixelBoost")) {
    if ($health.pixel_boost -in @("512x512","768x768","1024x1024")) { $FaceSwapperPixelBoost = $health.pixel_boost }
    elseif ($old.pixel_boost) { $FaceSwapperPixelBoost = $old.pixel_boost }
}
if (-not $PSBoundParameters.ContainsKey("EnableFaceEnhancer")) { $EnableFaceEnhancer = [bool]($health.face_enhancer_enabled -or $old.enhancer) }
if (-not $PSBoundParameters.ContainsKey("DisableRealism") -and $old) { $DisableRealism = [bool]$old.disable_realism }
if (-not $PSBoundParameters.ContainsKey("ShadingStrength") -and $old -and $null -ne $old.shading) { $ShadingStrength = $old.shading }
if (-not $PSBoundParameters.ContainsKey("NimEditors") -and -not $NimEditors -and $old) { $NimEditors = $old.editors }
# No credentials or photo data are written. Existing pairing token remains
# in the bridge's own local file. Do not auto-enable an untested editor.
New-Item -ItemType Directory -Path $directory -Force | Out-Null
$cfg = [ordered]@{
    python=$python; activated_path=[string]$envInfo.path
    facefusion_dir=$FaceFusionDir; pixel_boost=$FaceSwapperPixelBoost
    pairing=[bool]$EnablePairing; enhancer=[bool]$EnableFaceEnhancer
    disable_realism=[bool]$DisableRealism; realism_dir=$RealismDir
    shading=$(if ($ShadingStrength -ge 0) { $ShadingStrength } else { $null })
    editors=[string]$NimEditors
}
$repoRoot = [System.IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent)).TrimEnd("\")
$taskRoot = $null
$moved = $false
if ($task) {
    $taskRoots = @($task.Actions | ForEach-Object { $_.WorkingDirectory } | Where-Object { $_ })
    if ($taskRoots.Count -eq 1) { $taskRoot = [System.IO.Path]::GetFullPath([string]$taskRoots[0]).TrimEnd("\") }
}
if ($task -and $task.State -eq "Running") {
    if ($taskRoot -and $taskRoot -eq $repoRoot) {
        Write-Host "Startup task is already running from this folder. No changes were applied; use Status to verify." -ForegroundColor Green
        return
    }
    # The task runs another Retake folder: move it here, but never interrupt a photo.
    if (-not $health -or $health.busy -ne $false) {
        throw "The startup task runs another Retake folder, but the engine is busy or not answering. Run this again when no photo is being processed."
    }
    Write-Host ("Startup task runs another Retake folder ({0}); moving it to {1}. Settings are kept." -f $(if ($taskRoot) { $taskRoot } else { "unknown" }), $repoRoot) -ForegroundColor Yellow
    Stop-ScheduledTask -TaskName $taskName
    $moved = $true
    Start-Sleep -Seconds 3
}
# Fixed, explicit local script path, no PowerShell profile or visible console.
$cfg | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $configPath -Encoding UTF8
$taskAction = New-ScheduledTaskAction -Execute $pythonw -Argument ('"' + $runner + '" --config "' + $configPath + '"') -WorkingDirectory $repoRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $identity.Name
$principal = New-ScheduledTaskPrincipal -UserId $identity.Name -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -RestartCount 10 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
Register-ScheduledTask -TaskName $taskName -Action $taskAction -Trigger $trigger -Principal $principal -Settings $settings -Description "Retake local FaceFusion engine (loopback only, no paid services)" -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-Host "Retake startup task registered for your Windows login." -ForegroundColor Green
if ($moved) {
    Write-Host "The engine now runs this folder. It needs about a minute to load; verify with: .\tools\retake_doctor.ps1" -ForegroundColor Green
} elseif ($health) {
    Write-Host "The existing engine was NOT stopped. When its current photo is finished, close its old PowerShell window. The hidden engine will then take over automatically."
} else {
    Write-Host "Run this script with -Action Status to confirm live engine health."
}
Write-Host "The PC must remain powered on. No cloud GPU was installed."
