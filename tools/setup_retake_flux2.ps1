<#
.SYNOPSIS
  Optional separate local FLUX.2 Klein 4B environment. No cloud inference.
.DESCRIPTION
  Installs packages only when -Install is explicitly supplied; downloads the
  official Apache-2.0 model only with -Download. Always performs a real CUDA
  synthetic image edit before starting the loopback editor. Does not touch the
  FaceFusion Python environment or automatically enable an untested model.
#>
param(
    [switch]$Install,
    [switch]$Download,
    [switch]$ProbeOnly,
    [switch]$EnableManaged,
    [string]$PythonExe = "python",
    [string]$EnvironmentDir = "$HOME\retake-flux2-venv",
    [string]$ModelDir = "$HOME\retake-models\FLUX.2-klein-4B",
    [ValidateSet("cu126","cu128","cu130")][string]$CudaWheel = "cu128",
    [ValidateSet("model","sequential")][string]$Offload = "model",
    [ValidateSet(512,768,1024)][int]$MaxEdge = 768,
    [ValidateRange(1024,65535)][int]$Port = 8004
)
$ErrorActionPreference = "Stop"
if ($EnableManaged -and (-not $ProbeOnly -or $Offload -ne "sequential" -or $MaxEdge -ne 512)) { throw "Use -EnableManaged -ProbeOnly -Offload sequential -MaxEdge 512." }
if ($Port -eq 8765) { throw "Use a separate port, not the FaceFusion bridge." }
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { throw "NVIDIA driver/nvidia-smi not found." }
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
if ($LASTEXITCODE -ne 0) { throw "NVIDIA GPU query failed." }
$Py = Join-Path $EnvironmentDir "Scripts\python.exe"
function Invoke-Checked {
    param([string]$Exe,[string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Exe" }
}
if ($Install) {
    if (-not (Test-Path $Py)) { Invoke-Checked $PythonExe @("-m","venv",$EnvironmentDir) }
    Invoke-Checked $Py @("-m","pip","install","--upgrade","pip")
    Invoke-Checked $Py @("-m","pip","install","torch","--index-url","https://download.pytorch.org/whl/$CudaWheel")
    Invoke-Checked $Py @("-m","pip","install","--upgrade","diffusers","transformers","accelerate","safetensors","Pillow","huggingface_hub")
    Invoke-Checked $Py @("-c","import torch; from diffusers import Flux2KleinPipeline; assert torch.cuda.is_available(), 'CUDA not available'; assert torch.cuda.is_bf16_supported(), 'BF16 not supported'; print(torch.__version__,torch.cuda.get_device_name(0))")
}
if (-not (Test-Path $Py)) { throw "Missing separate environment. Run with -Install first." }
if ($Download) {
    # Only download weights/configs from the fixed official repository; do not
    # upload images, run remote Python, or request paid API credits.
    Invoke-Checked $Py @("-c","from huggingface_hub import snapshot_download; import sys; snapshot_download(repo_id='black-forest-labs/FLUX.2-klein-4B',local_dir=sys.argv[1],allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.tiktoken','*.jinja','LICENSE','README.md']); print('Local model download completed')",$ModelDir)
}
if (-not (Test-Path (Join-Path $ModelDir "model_index.json"))) { throw "Model not installed. Run with -Download after reviewing disk space and license." }
$Runner = Join-Path $PSScriptRoot "local_flux2_editor.py"
$RunnerArgs = @($Runner,"--model-dir",$ModelDir,"--port","$Port","--max-edge","$MaxEdge","--offload",$Offload)
if ($ProbeOnly) { $RunnerArgs += "--probe-only" }
if ($EnableManaged) { $RunnerArgs += "--enable-managed" }
Write-Host "Running the real GPU synthetic edit. Keep the editor window open only after the probe succeeds." -ForegroundColor Cyan
if (-not $ProbeOnly) {
    Write-Host "After READY, use a separate Retake PowerShell window: " -ForegroundColor Yellow
    Write-Host ('$env:RETAKE_NIM_EDITORS = "retake-local/flux2-klein-4b=' + $Port + '"')
    Write-Host '.\tools\start_facefusion_retake.ps1'
}
Invoke-Checked $Py $RunnerArgs
