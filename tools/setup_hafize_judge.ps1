param(
    # Mevcut jetonu yenile (eski özet Hafize'de geçersiz olur; yeni özeti GitHub Secret'a yeniden yaz).
    [switch]$Rotate,
    # NVIDIA jürisini bu bilgisayarda kapat: yerel jeton dosyasını siler.
    [switch]$Disable,
    # Yalnız Hafize'nin herkese açık sağlık bilgisini kontrol et (görsel gönderilmez).
    [switch]$Check,
    [string]$HafizeUrl = "https://hafize-ua.web.app"
)

# Retake NVIDIA görsel jürisi kurulumu.
# Ham proje jetonu yalnız bu bilgisayarda (%USERPROFILE%\.retake\hafize-project-token) kalır.
# Hafize'ye yalnız SHA-256 özeti verilir: {"retake":"<64 hex>"} -> HAFIZE_PROJECT_AI_TOKEN_HASHES.
$ErrorActionPreference = "Stop"

$RetakeDir = Join-Path $env:USERPROFILE ".retake"
$TokenPath = Join-Path $RetakeDir "hafize-project-token"

if ($Disable) {
    if (Test-Path $TokenPath) {
        Remove-Item -LiteralPath $TokenPath -Force
        Write-Host "NVIDIA jürisi kapatıldı: yerel jeton silindi. Hafize'deki özeti de kaldırabilirsin." -ForegroundColor Yellow
    } else {
        Write-Host "Yerel jeton zaten yok; NVIDIA jürisi kapalı." -ForegroundColor Yellow
    }
    return
}

if ($Check) {
    $health = Invoke-RestMethod -Uri ($HafizeUrl.TrimEnd("/") + "/api/health") -Method Get -TimeoutSec 60
    $gateway = $health.projectAiGateway
    if (-not $gateway) { throw "Hafize sağlık yanıtında projectAiGateway yok: backend eski sürümde olabilir." }
    $judgeReady = [bool]($gateway.enabled -and $gateway.visionJudge -and ($gateway.visionJudge.projects -contains "retake"))
    Write-Host ("Proje geçidi etkin: {0} (yapılandırılmış proje: {1})" -f $gateway.enabled, $gateway.configuredProjects)
    if ($gateway.visionJudge) { Write-Host ("Jüri modelleri: " + ($gateway.visionJudge.models -join ", ")) }
    Write-Host ("Yerel jeton: " + $(if (Test-Path $TokenPath) { "var" } else { "yok" }))
    if ($judgeReady -and (Test-Path $TokenPath)) {
        Write-Host "Hazır görünüyor. Retake motorunu yeniden başlatınca /health içinde nvidia_judge=configured olmalı." -ForegroundColor Green
    } else {
        Write-Host "Henüz hazır değil: özeti Hafize'ye ekleyip backend'i yeniden dağıt, sonra tekrar kontrol et." -ForegroundColor Yellow
    }
    return
}

New-Item -ItemType Directory -Force -Path $RetakeDir | Out-Null

if ((Test-Path $TokenPath) -and -not $Rotate) {
    $Token = (Get-Content -LiteralPath $TokenPath -Raw).Trim()
    Write-Host "Mevcut yerel jeton kullanılıyor (yenilemek için -Rotate)." -ForegroundColor Cyan
} else {
    $bytes = New-Object byte[] 36
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
    # base64url, 48 karakter: Hafize'nin 43–128 karakter kuralına uyar.
    $Token = [Convert]::ToBase64String($bytes).TrimEnd("=").Replace("+", "-").Replace("/", "_")
    [System.IO.File]::WriteAllText($TokenPath, $Token, (New-Object System.Text.UTF8Encoding $false))
    # Dosyayı yalnız bu Windows kullanıcısının okuyabileceği hale getir.
    & icacls $TokenPath /inheritance:r /grant:r ("{0}:(R,W)" -f $env:USERNAME) | Out-Null
    Write-Host "Yeni proje jetonu üretildi ve yalnız bu bilgisayarda saklandı: $TokenPath" -ForegroundColor Green
}

if ($Token -notmatch '^[A-Za-z0-9_-]{43,128}$') { throw "Yerel jeton biçimi geçersiz. -Rotate ile yeniden üret." }

$sha = [System.Security.Cryptography.SHA256]::Create()
try {
    $digestBytes = $sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($Token))
} finally { $sha.Dispose() }
$Digest = -join ($digestBytes | ForEach-Object { $_.ToString("x2") })
$SecretValue = '{"retake":"' + $Digest + '"}'

Write-Host ""
Write-Host "1) Hafize deposunda GitHub > Settings > Secrets and variables > Actions > New repository secret:" -ForegroundColor Cyan
Write-Host "   Ad:    HAFIZE_PROJECT_AI_TOKEN_HASHES"
Write-Host "   Değer: $SecretValue"
Write-Host "   (Bu yalnız özet; ham jeton değildir. Başka proje özetleri varsa aynı JSON nesnesine ekle.)"
Write-Host "2) Hafize'de 'Hafize Cloud Run backend — deploy and connect' iş akışını yeniden çalıştır." -ForegroundColor Cyan
Write-Host "   Render kullanıyorsan aynı değeri servis ortam değişkeni olarak ekle."
Write-Host "3) Retake motorunu yeniden başlat; sitede 'NVIDIA görsel jürisi' kutusu görünür (varsayılan kapalı)." -ForegroundColor Cyan
Write-Host "   Kontrol: .\tools\setup_hafize_judge.ps1 -Check"
try { Set-Clipboard -Value $SecretValue; Write-Host "Özet değer panoya kopyalandı." -ForegroundColor Green } catch { }
