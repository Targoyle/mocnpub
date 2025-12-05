# mocnpub ベンチマークスクリプト
# 指定した秒数だけ実行して、keys/sec を取得する
#
# 使い方:
#   .\scripts\benchmark.ps1
#   .\scripts\benchmark.ps1 -Seconds 120 -BatchSize 1146880
#   .\scripts\benchmark.ps1 -KeysPerThread 2048  # 再ビルドして実行
#
# KeysPerThread を変更すると、環境変数 MAX_KEYS_PER_THREAD を設定して
# cargo build --release を実行してから benchmark を開始します。

param(
    [int]$Seconds = 120,
    [int]$BatchSize = 1146880,
    [int]$ThreadsPerBlock = 128,
    [int]$KeysPerThread = 1408,
    [string]$Prefix = "00000000",
    [switch]$SkipBuild = $false
)

$projectRoot = Join-Path $PSScriptRoot ".."
$projectRoot = [System.IO.Path]::GetFullPath($projectRoot)
$exePath = Join-Path $projectRoot "target\release\mocnpub-main.exe"

Write-Host "=== mocnpub Benchmark ===" -ForegroundColor Cyan
Write-Host "Parameters:" -ForegroundColor Yellow
Write-Host "  BatchSize:        $BatchSize"
Write-Host "  ThreadsPerBlock:  $ThreadsPerBlock"
Write-Host "  KeysPerThread:    $KeysPerThread (build-time)"
Write-Host "  Duration:         $Seconds sec"
Write-Host "  Prefix:           $Prefix"
Write-Host ""

# ビルド（-SkipBuild が指定されていない場合）
if (-not $SkipBuild) {
    Write-Host "Building with MAX_KEYS_PER_THREAD=$KeysPerThread..." -ForegroundColor Yellow
    $env:MAX_KEYS_PER_THREAD = $KeysPerThread
    Push-Location $projectRoot
    try {
        cargo build --release
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error: cargo build failed" -ForegroundColor Red
            exit 1
        }
    } finally {
        Pop-Location
        Remove-Item Env:\MAX_KEYS_PER_THREAD -ErrorAction SilentlyContinue
    }
    Write-Host "Build completed!" -ForegroundColor Green
    Write-Host ""
}

if (-not (Test-Path $exePath)) {
    Write-Host "Error: $exePath not found." -ForegroundColor Red
    exit 1
}

# プロセスを起動
$procArgs = @(
    "--gpu",
    "--prefix", $Prefix,
    "--limit", "0",
    "--batch-size", $BatchSize,
    "--threads-per-block", $ThreadsPerBlock
)

$pinfo = New-Object System.Diagnostics.ProcessStartInfo
$pinfo.FileName = $exePath
$pinfo.Arguments = $procArgs -join " "
$pinfo.RedirectStandardOutput = $true
$pinfo.RedirectStandardError = $true
$pinfo.UseShellExecute = $false
$pinfo.CreateNoWindow = $true

$process = New-Object System.Diagnostics.Process
$process.StartInfo = $pinfo

Write-Host "Starting benchmark..." -ForegroundColor Green
$process.Start() | Out-Null

# 指定秒数待つ
Start-Sleep -Seconds $Seconds

# プロセスを終了
$process.Kill()
$process.WaitForExit()

# 出力を取得
$stdout = $process.StandardOutput.ReadToEnd()
$stderr = $process.StandardError.ReadToEnd()

# 最後の keys/sec を抽出
$lines = $stdout -split "`n"
$lastKeysPerSec = $null

foreach ($line in $lines) {
    if ($line -match "(\d+\.?\d*)\s+keys/sec") {
        $lastKeysPerSec = $matches[1]
    }
}

Write-Host ""
Write-Host "=== Result ===" -ForegroundColor Cyan
if ($lastKeysPerSec) {
    $keysPerSecFloat = [double]$lastKeysPerSec
    $keysPerSecB = $keysPerSecFloat / 1e9
    Write-Host "Keys/sec: $lastKeysPerSec ($([math]::Round($keysPerSecB, 3))B)" -ForegroundColor Green

    # CSV 形式で出力（コピペ用）
    Write-Host ""
    Write-Host "CSV: $BatchSize,$ThreadsPerBlock,$KeysPerThread,$lastKeysPerSec" -ForegroundColor Yellow
} else {
    Write-Host "Could not extract keys/sec from output" -ForegroundColor Red
    Write-Host "Output:" -ForegroundColor Yellow
    Write-Host $stdout
}
