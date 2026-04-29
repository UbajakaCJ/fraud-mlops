param(
    [string]$BaseUrl  = "http://localhost",
    [string]$Username = "admin",
    [string]$Password = "admin",
    [string]$FixturesPath = ".\tests\fixtures"
)

# -------------------------------------------------------
#  FRAUD DETECTION API -- BATCH LOAD TEST
#  Tests /predict/batch with 1,000 / 5,000 / 10,000 txns
#  MongoDB MLOps Platform | Bluechip Technologies
# -------------------------------------------------------

$BatchUrl = "$BaseUrl/predict/batch"
$TxnUrl   = "$BaseUrl/transactions/"

# -------------------------------------------------------
#  AUTHENTICATE
# -------------------------------------------------------
Write-Host ""
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "   FRAUD DETECTION API -- BATCH LOAD TEST" -ForegroundColor Cyan
Write-Host "   MongoDB MLOps Platform | Bluechip Technologies" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  Authenticating against $BaseUrl/auth/token ..." -ForegroundColor Gray

try {
    $AuthBody = @{ username = $Username; password = $Password } | ConvertTo-Json
    $AuthResp = Invoke-RestMethod -Uri "$BaseUrl/auth/token" -Method POST `
        -ContentType "application/json" -Body $AuthBody -ErrorAction Stop
    $Token   = $AuthResp.access_token
    $Headers = @{ "Authorization" = "Bearer $Token" }
    Write-Host "  Auth OK. Token acquired." -ForegroundColor Green
} catch {
    Write-Host "  AUTH FAILED: $_" -ForegroundColor Red
    exit 1
}

# -------------------------------------------------------
#  HELPER: Run one batch and return result object
# -------------------------------------------------------
function Invoke-BatchTest {
    param(
        [string]$FixtureFile,
        [int]$BatchSize,
        [hashtable]$Headers,
        [string]$Url
    )

    Write-Host ""
    Write-Host "  ── Batch $BatchSize ──────────────────────────────────" -ForegroundColor Yellow

    # Load fixture
    if (-not (Test-Path $FixtureFile)) {
        Write-Host "  ERROR: Fixture not found at $FixtureFile" -ForegroundColor Red
        return $null
    }

    $Body = Get-Content $FixtureFile -Raw
    $BodySize = [math]::Round((($Body | Measure-Object -Character).Characters) / 1KB, 1)
    Write-Host ("  Fixture     : {0} ({1} KB)" -f (Split-Path $FixtureFile -Leaf), $BodySize)
    Write-Host ("  Sending {0} transactions to {1} ..." -f $BatchSize, $Url) -ForegroundColor Gray

    $Start = Get-Date
    try {
        $Resp    = Invoke-RestMethod -Uri $Url -Method POST `
            -ContentType "application/json" `
            -Headers $Headers `
            -Body $Body `
            -ErrorAction Stop
        $Elapsed = ((Get-Date) - $Start).TotalMilliseconds

        $TPS     = if ($Elapsed -gt 0) { [math]::Round($BatchSize / ($Elapsed / 1000), 1) } else { 0 }
        $PerTxn  = [math]::Round($Elapsed / $BatchSize, 3)

        Write-Host ("  Status      : SUCCESS (HTTP 200)") -ForegroundColor Green
        Write-Host ("  Total time  : {0}ms" -f [math]::Round($Elapsed, 2))
        Write-Host ("  Per-txn     : {0}ms" -f $PerTxn)
        Write-Host ("  Throughput  : {0} txns/sec" -f $TPS)
        Write-Host ("  Fraud count : {0} / {1} ({2}%)" -f $Resp.fraud_count, $Resp.total, $Resp.fraud_rate_pct)
        Write-Host ("  Model used  : {0}" -f $Resp.model_name)
        Write-Host ("  API latency : {0}ms (reported by API)" -f $Resp.latency_ms)

        return [PSCustomObject]@{
            BatchSize    = $BatchSize
            Success      = $true
            WallTimeMs   = [math]::Round($Elapsed, 2)
            PerTxnMs     = $PerTxn
            TPS          = $TPS
            FraudCount   = $Resp.fraud_count
            FraudRatePct = $Resp.fraud_rate_pct
            ModelName    = $Resp.model_name
            ApiLatencyMs = $Resp.latency_ms
            Error        = $null
        }
    } catch {
        $Elapsed = ((Get-Date) - $Start).TotalMilliseconds
        $ErrMsg  = $_.Exception.Message
        Write-Host ("  Status      : FAILED after {0}ms" -f [math]::Round($Elapsed, 2)) -ForegroundColor Red
        Write-Host ("  Error       : $ErrMsg") -ForegroundColor Red

        return [PSCustomObject]@{
            BatchSize    = $BatchSize
            Success      = $false
            WallTimeMs   = [math]::Round($Elapsed, 2)
            PerTxnMs     = 0
            TPS          = 0
            FraudCount   = 0
            FraudRatePct = 0
            ModelName    = "N/A"
            ApiLatencyMs = 0
            Error        = $ErrMsg
        }
    }
}

# -------------------------------------------------------
#  RUN ALL THREE BATCH SIZES
# -------------------------------------------------------
$Batches = @(
    @{ Size = 1000;  File = "$FixturesPath\batch_1000.json"  },
    @{ Size = 5000;  File = "$FixturesPath\batch_5000.json"  },
    @{ Size = 10000; File = "$FixturesPath\batch_10000.json" }
)

Write-Host ""
Write-Host "  Target URL  : $BatchUrl"
Write-Host "  Fixtures    : $FixturesPath"
Write-Host "=======================================================" -ForegroundColor Cyan

$Results = @()
foreach ($b in $Batches) {
    $r = Invoke-BatchTest -FixtureFile $b.File -BatchSize $b.Size -Headers $Headers -Url $BatchUrl
    if ($r) { $Results += $r }
}

# -------------------------------------------------------
#  MONGODB AUDIT CHECK
# -------------------------------------------------------
Write-Host ""
Write-Host "  Checking MongoDB audit log..." -ForegroundColor Gray
try {
    $TxnResp    = Invoke-RestMethod -Uri $TxnUrl -Headers $Headers -ErrorAction Stop
    $MongoTotal = $TxnResp.total
    Write-Host ("  MongoDB transaction_logs total docs: {0}" -f $MongoTotal) -ForegroundColor Green
} catch {
    $MongoTotal = "N/A"
    Write-Host "  Could not query MongoDB audit log." -ForegroundColor Yellow
}

# -------------------------------------------------------
#  SUMMARY TABLE
# -------------------------------------------------------
Write-Host ""
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "   BATCH TEST SUMMARY" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ("  {0,-10} {1,-10} {2,-12} {3,-12} {4,-12} {5,-10}" -f `
    "Batch", "Status", "Wall(ms)", "Per-txn(ms)", "TPS", "FraudRate%")
Write-Host ("  {0,-10} {1,-10} {2,-12} {3,-12} {4,-12} {5,-10}" -f `
    "─────", "──────", "────────", "───────────", "───", "──────────")

foreach ($r in $Results) {
    $Status = if ($r.Success) { "OK" } else { "FAILED" }
    $Color  = if ($r.Success) { "Green" } else { "Red" }
    Write-Host ("  {0,-10} {1,-10} {2,-12} {3,-12} {4,-12} {5,-10}" -f `
        $r.BatchSize, $Status, $r.WallTimeMs, $r.PerTxnMs, $r.TPS, $r.FraudRatePct) `
        -ForegroundColor $Color
}

Write-Host ""
Write-Host ("  MongoDB total logged : {0} transactions" -f $MongoTotal)
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# -------------------------------------------------------
#  SCALING ANALYSIS
# -------------------------------------------------------
$Successful = $Results | Where-Object { $_.Success }
if ($Successful.Count -ge 2) {
    Write-Host "  SCALING ANALYSIS" -ForegroundColor Yellow
    $Base = $Successful[0]
    foreach ($r in $Successful) {
        $ScaleFactor    = [math]::Round($r.BatchSize / $Base.BatchSize, 1)
        $TimeMultiplier = if ($Base.WallTimeMs -gt 0) { [math]::Round($r.WallTimeMs / $Base.WallTimeMs, 2) } else { 0 }
        $Efficiency     = if ($ScaleFactor -gt 0) { [math]::Round($TimeMultiplier / $ScaleFactor * 100, 1) } else { 0 }
        Write-Host ("  Batch {0,6}: {1,2}x data -> {2,5}x time  (scaling efficiency: {3}%)" -f `
            $r.BatchSize, $ScaleFactor, $TimeMultiplier, $Efficiency)
    }
    Write-Host "  (Sub-linear scaling = good vectorisation in the model pipeline)" -ForegroundColor Gray
    Write-Host "=======================================================" -ForegroundColor Cyan
}
