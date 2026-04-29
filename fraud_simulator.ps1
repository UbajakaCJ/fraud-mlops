param(
    [string]$BaseUrl       = "http://localhost",
    [string]$Username      = "admin",
    [string]$Password      = "admin",
    [string]$FixtureFile   = ".\tests\fixtures\batch_1000.json",
    [int]$TransactionsPerSec = 5,       # simulated feed speed
    [int]$MaxTransactions    = 50,      # how many to process (0 = all)
    [switch]$AlertsOnly                 # show only fraud alerts
)

# -------------------------------------------------------
#  REAL-TIME FRAUD DETECTION SIMULATOR
#  Streams transactions from fixture file one-by-one,
#  simulating a live payment processing feed.
#  Bluechip Technologies | MongoDB MLOps Platform
# -------------------------------------------------------

# ANSI colour helpers
function Green  { param($t) Write-Host $t -ForegroundColor Green  -NoNewline }
function Red    { param($t) Write-Host $t -ForegroundColor Red    -NoNewline }
function Yellow { param($t) Write-Host $t -ForegroundColor Yellow -NoNewline }
function Cyan   { param($t) Write-Host $t -ForegroundColor Cyan   -NoNewline }
function Gray   { param($t) Write-Host $t -ForegroundColor Gray   -NoNewline }
function White  { param($t) Write-Host $t -ForegroundColor White  -NoNewline }
function NL     { Write-Host "" }

$SleepMs = [math]::Round(1000 / $TransactionsPerSec)

# -------------------------------------------------------
#  BANNER
# -------------------------------------------------------
Clear-Host
Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   🛡  REAL-TIME FRAUD DETECTION SIMULATOR            ║" -ForegroundColor Cyan
Write-Host "║   Bluechip Technologies — AI Services                ║" -ForegroundColor Cyan
Write-Host "║   MongoDB MLOps Platform                             ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# -------------------------------------------------------
#  AUTHENTICATE
# -------------------------------------------------------
Gray "  Authenticating ..."; NL
try {
    $AuthBody = @{ username = $Username; password = $Password } | ConvertTo-Json
    $AuthResp = Invoke-RestMethod -Uri "$BaseUrl/auth/token" -Method POST `
        -ContentType "application/json" -Body $AuthBody -ErrorAction Stop
    $Token   = $AuthResp.access_token
    $Headers = @{ "Authorization" = "Bearer $Token" }
    Green "  ✓ Auth OK"; NL
} catch {
    Red "  ✗ AUTH FAILED: $_"; NL
    exit 1
}

# -------------------------------------------------------
#  LOAD TRANSACTIONS
# -------------------------------------------------------
if (-not (Test-Path $FixtureFile)) {
    Red "  ✗ Fixture not found: $FixtureFile"; NL
    exit 1
}
$Data         = Get-Content $FixtureFile -Raw | ConvertFrom-Json
$Transactions = $Data.transactions
$Total        = if ($MaxTransactions -gt 0) { [math]::Min($MaxTransactions, $Transactions.Count) } else { $Transactions.Count }

Write-Host ""
Write-Host "  Feed source : $(Split-Path $FixtureFile -Leaf)" -ForegroundColor Gray
Write-Host "  Processing  : $Total transactions at $TransactionsPerSec txns/sec" -ForegroundColor Gray
Write-Host "  Endpoint    : $BaseUrl/predict/" -ForegroundColor Gray
if ($AlertsOnly) {
    Write-Host "  Mode        : ALERTS ONLY (fraud transactions)" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "  Starting feed in 2 seconds..." -ForegroundColor Gray
Start-Sleep -Seconds 2

# -------------------------------------------------------
#  HEADER ROW
# -------------------------------------------------------
Write-Host "─────────────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host ("  {0,-18} {1,-10} {2,-10} {3,-8} {4,-10} {5,-8} {6}" -f `
    "Transaction ID", "Amount", "Risk", "Fraud?", "Prob", "Lat(ms)", "Status") -ForegroundColor DarkGray
Write-Host "─────────────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray

# -------------------------------------------------------
#  COUNTERS
# -------------------------------------------------------
$ProcessedCount = 0
$FraudCount     = 0
$LegitCount     = 0
$ErrorCount     = 0
$TotalLatency   = 0.0
$MaxLatency     = 0.0
$MinLatency     = [double]::MaxValue
$CriticalAlerts = 0
$HighAlerts     = 0
$StartTime      = Get-Date

# -------------------------------------------------------
#  MAIN FEED LOOP
# -------------------------------------------------------
for ($i = 0; $i -lt $Total; $i++) {
    $Txn    = $Transactions[$i]
    $TxnId  = "TXN-{0:D6}" -f ($i + 1)
    $Amount = $Txn.Amount

    # Build request body
    $Body = @{
        transaction_id = $TxnId
        transaction    = $Txn
    } | ConvertTo-Json -Depth 5

    $Start = Get-Date
    try {
        $Resp   = Invoke-RestMethod -Uri "$BaseUrl/predict/" -Method POST `
            -ContentType "application/json" -Headers $Headers -Body $Body -ErrorAction Stop
        $LatMs  = [math]::Round(((Get-Date) - $Start).TotalMilliseconds, 2)

        $ProcessedCount++
        $TotalLatency += $LatMs
        if ($LatMs -gt $MaxLatency) { $MaxLatency = $LatMs }
        if ($LatMs -lt $MinLatency) { $MinLatency = $LatMs }

        $IsFraud   = $Resp.is_fraud
        $Prob      = $Resp.fraud_probability
        $Risk      = $Resp.risk_level
        $ProbPct   = "{0:P1}" -f $Prob

        if ($IsFraud) {
            $FraudCount++
            if ($Risk -eq "CRITICAL") { $CriticalAlerts++ }
            if ($Risk -eq "HIGH")     { $HighAlerts++ }
        } else {
            $LegitCount++
        }

        # Skip legit if alerts-only mode
        if ($AlertsOnly -and -not $IsFraud) {
            Start-Sleep -Milliseconds $SleepMs
            continue
        }

        # ── Print transaction row ──
        White ("  {0,-18}" -f $TxnId)

        # Amount
        $AmtStr = "₦{0:N2}" -f $Amount
        Gray (" {0,-10}" -f $AmtStr)

        # Risk level with colour
        switch ($Risk) {
            "CRITICAL" { Red    (" {0,-10}" -f $Risk) }
            "HIGH"     { Yellow (" {0,-10}" -f $Risk) }
            "MEDIUM"   { Cyan   (" {0,-10}" -f $Risk) }
            "LOW"      { Green  (" {0,-10}" -f $Risk) }
            default    { Gray   (" {0,-10}" -f $Risk) }
        }

        # Fraud flag
        if ($IsFraud) {
            Red (" {0,-8}" -f "🚨 YES")
        } else {
            Green (" {0,-8}" -f "✓  NO")
        }

        # Probability
        Gray (" {0,-10}" -f $ProbPct)

        # Latency
        Gray (" {0,-8}" -f $LatMs)

        # Alert banner for fraud
        if ($IsFraud) {
            switch ($Risk) {
                "CRITICAL" { Red    " ◄ BLOCK TRANSACTION" }
                "HIGH"     { Yellow " ◄ FLAG FOR REVIEW"   }
                default    { Yellow " ◄ MONITOR"           }
            }
        } else {
            Green " ✓ APPROVED"
        }
        NL

        # Extra alert line for critical fraud
        if ($IsFraud -and $Risk -eq "CRITICAL") {
            Red ("  ⚠  CRITICAL FRAUD ALERT — {0} — Probability: {1}" -f $TxnId, $ProbPct)
            NL
        }

    } catch {
        $ErrorCount++
        $LatMs = [math]::Round(((Get-Date) - $Start).TotalMilliseconds, 2)
        Red ("  {0,-18} ERROR: {1}" -f $TxnId, $_.Exception.Message)
        NL
    }

    Start-Sleep -Milliseconds $SleepMs
}

# -------------------------------------------------------
#  SESSION SUMMARY
# -------------------------------------------------------
$ElapsedSec  = [math]::Round(((Get-Date) - $StartTime).TotalSeconds, 1)
$AvgLatency  = if ($ProcessedCount -gt 0) { [math]::Round($TotalLatency / $ProcessedCount, 2) } else { 0 }
$FraudRate   = if ($ProcessedCount -gt 0) { [math]::Round($FraudCount / $ProcessedCount * 100, 1) } else { 0 }
$ActualTPS   = if ($ElapsedSec -gt 0)     { [math]::Round($ProcessedCount / $ElapsedSec, 1) }    else { 0 }

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   SESSION SUMMARY" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ("  Processed       : {0} transactions" -f $ProcessedCount)
Green ("  Legitimate      : {0}" -f $LegitCount); NL
Red   ("  Fraud detected  : {0} ({1}%)" -f $FraudCount, $FraudRate); NL
if ($ErrorCount -gt 0) {
    Yellow ("  Errors          : {0}" -f $ErrorCount); NL
}
Write-Host ""
Write-Host "  ALERTS" -ForegroundColor Yellow
Red    ("  CRITICAL        : {0}  ◄ Transactions blocked" -f $CriticalAlerts); NL
Yellow ("  HIGH            : {0}  ◄ Flagged for review"   -f $HighAlerts);     NL
Write-Host ""
Write-Host "  PERFORMANCE" -ForegroundColor Yellow
Write-Host ("  Avg latency     : {0}ms" -f $AvgLatency)
Write-Host ("  Min latency     : {0}ms" -f [math]::Round($MinLatency, 2))
Write-Host ("  Max latency     : {0}ms" -f $MaxLatency)
Write-Host ("  Actual TPS      : {0}" -f $ActualTPS)
Write-Host ("  Session time    : {0}s" -f $ElapsedSec)

# MongoDB audit check
try {
    $TxnResp = Invoke-RestMethod -Uri "$BaseUrl/transactions/?size=1" -Headers $Headers -ErrorAction Stop
    Write-Host ""
    Write-Host "  MONGODB AUDIT" -ForegroundColor Yellow
    Green ("  Total logged    : {0} transactions" -f $TxnResp.total); NL
} catch {}

Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
