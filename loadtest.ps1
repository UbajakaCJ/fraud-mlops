param(
    [int]$Workers      = 25,
    [int]$Duration     = 30,
    [string]$BaseUrl   = "http://localhost",
    [string]$Username  = "admin",
    [string]$Password  = "admin"
)

# -------------------------------------------------------
#  FRAUD DETECTION API -- REAL-TIME LOAD TEST
#  MongoDB-backed MLOps platform (fraud-mlops)
#  Payloads: real vectors from the IEEE-CIS creditcard
#  dataset (V1-V28 + Amount + Time).
#  Mix: 4 legitimate transactions + 4 fraud transactions
# -------------------------------------------------------

# -------------------------------------------------------
#  STEP 1: Authenticate and get JWT token
# -------------------------------------------------------
$AuthUrl = "$BaseUrl/auth/token"
$PredictUrl = "$BaseUrl/predict/"

Write-Host ""
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "   FRAUD DETECTION API -- REAL-TIME LOAD TEST" -ForegroundColor Cyan
Write-Host "   MongoDB MLOps Platform | Bluechip Technologies" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  Authenticating against $AuthUrl ..." -ForegroundColor Gray

try {
    $AuthBody = @{ username = $Username; password = $Password } | ConvertTo-Json
    $AuthResp = Invoke-RestMethod -Uri $AuthUrl -Method POST `
        -ContentType "application/json" -Body $AuthBody -ErrorAction Stop
    $Token = $AuthResp.access_token
    Write-Host "  Auth OK. Token acquired." -ForegroundColor Green
} catch {
    Write-Host "  AUTH FAILED: $_" -ForegroundColor Red
    Write-Host "  Make sure the API is running and credentials are correct." -ForegroundColor Red
    exit 1
}

# -------------------------------------------------------
#  STEP 2: Define payloads
# -------------------------------------------------------
$Payloads = @(

    # ---- LEGITIMATE TRANSACTIONS (Class=0) ----

    # Legit #1  (row 0 of creditcard.csv)
    @{ transaction = @{
        Time=0; Amount=149.62
        V1=-1.3598071336738;   V2=-0.0727811733098497; V3=2.53634673796914
        V4=1.37815522427443;   V5=-0.338320769942518;  V6=0.462387777762292
        V7=0.239598554061257;  V8=0.0986979012610507;  V9=0.363786969611213
        V10=0.0907941719789316;V11=-0.551599533260813;  V12=-0.617800855762348
        V13=-0.991389847235408;V14=-0.311169353699879;  V15=1.46817697209427
        V16=-0.470400525259478;V17=0.207971241929242;   V18=0.0257905801985591
        V19=0.403992960255733; V20=0.251412098239705;   V21=-0.018306777944153
        V22=0.277837575558899; V23=-0.110473910188767;  V24=0.0669280749146731
        V25=0.128539358273528; V26=-0.189114843888824;  V27=0.133558376740387
        V28=-0.0210530534538215
    }},

    # Legit #2  (row 1 of creditcard.csv)
    @{ transaction = @{
        Time=0; Amount=2.69
        V1=1.19185711131486;   V2=0.266150712616424;   V3=0.16648011335321
        V4=0.448154078460911;  V5=0.0600176492822243;  V6=-0.0823608088155687
        V7=-0.0788029833563969;V8=0.0851016549148104;  V9=-0.255425128109186
        V10=-0.166974414004614;V11=1.61272666105479;    V12=1.06523531137287
        V13=0.48909501589608;  V14=-0.143772296441519;  V15=0.635558093258208
        V16=0.463917041022171; V17=-0.114804663102346;  V18=-0.183361270123994
        V19=-0.145783041325259;V20=-0.0690831352230203; V21=-0.225775248033138
        V22=-0.638671952771851;V23=0.101288021253234;   V24=-0.339846475529127
        V25=0.167170404418143; V26=0.125894532368176;   V27=-0.00898309914322813
        V28=0.0147241691924927
    }},

    # Legit #3  (row 2 of creditcard.csv)
    @{ transaction = @{
        Time=1; Amount=378.66
        V1=-1.35835406159823;  V2=-1.34016307473609;   V3=1.77320934263119
        V4=0.379779593034328;  V5=-0.503198133318193;   V6=1.80049938079263
        V7=0.791460956450422;  V8=0.247675786588991;    V9=-1.51465432260583
        V10=0.207642865216696; V11=0.624501459424895;   V12=0.066083685268831
        V13=0.717292731410831; V14=-0.165945922763521;  V15=2.34586494901581
        V16=-2.89008319444231; V17=1.10996937869599;    V18=-0.121359313195888
        V19=-2.26185709530414; V20=0.524979725224404;   V21=0.247998153469754
        V22=0.771679401917229; V23=0.909412262347719;   V24=-0.689280956490685
        V25=-0.327641833735251;V26=-0.139096571514147;  V27=-0.0553527940384261
        V28=-0.0597518405929204
    }},

    # Legit #4  (high-value, still legitimate)
    @{ transaction = @{
        Time=3600; Amount=239.93
        V1=1.22965922952208;   V2=0.141003507049326;   V3=0.045371069031894
        V4=1.20261273803506;   V5=0.191880773352328;   V6=0.272708056756419
        V7=-0.00516066779870729;V8=0.0812895012718127;  V9=0.464960113272494
        V10=-0.0992543866632645;V11=-1.41690724816416;  V12=-0.153825862988065
        V13=-1.02032033991609; V14=0.457278923938049;   V15=-0.295273884550352
        V16=-0.077464376656039;V17=0.0626084547536254;  V18=0.0238405466005617
        V19=0.384820302154822; V20=0.0176972337682685;  V21=-0.0431390098929898
        V22=-0.0170936657948088;V23=-0.171926487930073; V24=-0.0109547929695826
        V25=0.195724312284888; V26=-0.0120620060840433; V27=0.0267479736478688
        V28=0.00961896735694488
    }},

    # ---- FRAUDULENT TRANSACTIONS (Class=1) ----

    # Fraud #1  (first fraud in creditcard.csv)
    @{ transaction = @{
        Time=406; Amount=1.0
        V1=-2.3122265423263;   V2=1.95199201064158;    V3=-1.60985073229769
        V4=3.9979055875468;    V5=-0.522187864667764;   V6=-1.42654531920595
        V7=-2.53738730624579;  V8=1.39165724829804;     V9=-2.77008927719433
        V10=-2.77227214465915; V11=3.20203320709635;    V12=-2.89990738849473
        V13=-0.595221881324605;V14=-4.28925378244217;   V15=0.389724120274487
        V16=-1.14074717980657; V17=-2.83005567450437;   V18=-0.0168224681808257
        V19=0.416955705037907; V20=0.126910559061659;   V21=0.517232370861764
        V22=-0.0350493686052974;V23=-0.465211076082388; V24=0.320198198514526
        V25=0.0445191674731724;V26=0.177839798284401;   V27=0.261145002567677
        V28=-0.143275874698919
    }},

    # Fraud #2  (second fraud in creditcard.csv)
    @{ transaction = @{
        Time=472; Amount=9.99
        V1=1.19185711131486;   V2=0.266150712616424;   V3=-4.7981908059678
        V4=3.99194710938832;   V5=-0.522187864667764;   V6=-1.94744530069697
        V7=-1.15823309349523;  V8=0.863784174186397;    V9=-1.14946735687918
        V10=-2.83907394244609; V11=1.23618433038834;    V12=-2.8345527390951
        V13=-0.0461382512098235;V14=-3.08025041904167;  V15=-0.135553730518681
        V16=-0.338741753989405;V17=-0.561069609083929;  V18=0.154688882048804
        V19=0.0218939658055543;V20=0.214506644226895;   V21=0.253574873897994
        V22=0.00439748069680978;V23=-0.0558823614282987;V24=-0.0195754620958889
        V25=0.0437738842624769;V26=0.0857798695555086;  V27=-0.0174882684459569
        V28=-0.0109537768763538
    }},

    # Fraud #3  (large-amount fraud)
    @{ transaction = @{
        Time=7519; Amount=529.0
        V1=-3.04359273504725;  V2=-3.15730712090408;   V3=1.08846184923495
        V4=2.2886436179559;    V5=1.35980512966032;     V6=-1.06482252298131
        V7=0.325574266158614;  V8=-0.0678960235570755;  V9=-0.270952836226477
        V10=-0.838586564582682;V11=-0.414575448285725;  V12=-0.503140859566824
        V13=0.676501544643416; V14=-1.69202893650684;   V15=2.00063483909015
        V16=-0.49167938943625; V17=0.791460956450422;   V18=-0.208253515060287
        V19=0.502292224181552; V20=0.219422;             V21=0.215534
        V22=-0.0674568;        V23=-0.255984;            V24=0.0981965
        V25=-0.262902;         V26=0.0595148;            V27=-0.0425932
        V28=-0.0294395
    }},

    # Fraud #4  (small-amount card-testing fraud)
    @{ transaction = @{
        Time=150; Amount=1.0
        V1=-2.00603905858956;  V2=3.16818767996685;    V3=-2.17992706989789
        V4=2.36472474519052;   V5=1.90975032609396;     V6=0.0793930583587939
        V7=-0.263977451665895; V8=0.196656765325025;    V9=-0.427737574099432
        V10=-0.99518895813491; V11=0.394607073990717;   V12=-0.498317843924428
        V13=0.0296445616956873;V14=-0.992461375930884;  V15=0.0510539747820371
        V16=0.306078199208428; V17=-0.554087086028298;  V18=0.0595308481893783
        V19=0.214652166289982; V20=0.0724938958510987;  V21=0.209748006200833
        V22=0.207516006200833; V23=-0.0148467880048516; V24=-0.0342273695757445
        V25=0.0427395862389406;V26=-0.0301396455265398; V27=0.0453636956082668
        V28=0.0150862739120898
    }}
)

# -------------------------------------------------------
#  WORKER SCRIPT BLOCK
# -------------------------------------------------------
$WorkerScript = {
    param($Url, $PayloadsJson, $Duration, $Token)

    $Payloads   = $PayloadsJson | ConvertFrom-Json
    $Results    = [System.Collections.Generic.List[object]]::new()
    $Deadline   = (Get-Date).AddSeconds($Duration)
    $PayloadIdx = 0
    $Headers    = @{ "Authorization" = "Bearer $Token" }

    while ((Get-Date) -lt $Deadline) {
        $Payload  = $Payloads[$PayloadIdx % $Payloads.Count]
        $PayloadIdx++
        $Body     = $Payload | ConvertTo-Json -Depth 5
        $Start    = Get-Date

        try {
            $Resp = Invoke-RestMethod -Uri $Url -Method Post `
                -ContentType "application/json" `
                -Headers $Headers `
                -Body $Body `
                -ErrorAction Stop
            $Lat = ((Get-Date) - $Start).TotalMilliseconds
            $Results.Add([PSCustomObject]@{
                Success   = $true
                Latency   = [math]::Round($Lat, 2)
                IsFraud   = $Resp.is_fraud
                RiskLevel = $Resp.risk_level
                Prob      = $Resp.fraud_probability
            })
        } catch {
            $Lat = ((Get-Date) - $Start).TotalMilliseconds
            $Results.Add([PSCustomObject]@{
                Success   = $false
                Latency   = [math]::Round($Lat, 2)
                IsFraud   = $false
                RiskLevel = "ERROR"
                Prob      = 0
            })
        }
    }
    return $Results
}

# -------------------------------------------------------
#  MAIN
# -------------------------------------------------------
$PayloadsJson = $Payloads | ConvertTo-Json -Depth 10 -Compress

Write-Host "  Target URL  : $PredictUrl"
Write-Host "  Workers     : $Workers concurrent jobs"
Write-Host "  Duration    : $Duration seconds"
Write-Host "  Payloads    : $($Payloads.Count) real IEEE-CIS vectors (4 legit + 4 fraud)"
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "Starting $Workers workers..."
Write-Host ""

$StartTime = Get-Date
$Jobs = 1..$Workers | ForEach-Object {
    Start-Job -ScriptBlock $WorkerScript -ArgumentList $PredictUrl, $PayloadsJson, $Duration, $Token
}

# Progress ticker
while (((Get-Date) - $StartTime).TotalSeconds -lt ($Duration + 5)) {
    Start-Sleep -Seconds 5
    $Elapsed   = [math]::Round(((Get-Date) - $StartTime).TotalSeconds, 1)
    $Remaining = [math]::Max(0, $Duration - $Elapsed)
    $Done      = ($Jobs | Where-Object { $_.State -eq "Completed" }).Count
    Write-Host ("  [{0}s] Workers done: {1}/{2} | Remaining: {3}s" -f $Elapsed, $Done, $Workers, $Remaining)
    if ($Done -eq $Workers) { break }
}

Write-Host ""
Write-Host "Test complete. Collecting results..."

$AllResults = $Jobs | ForEach-Object {
    Receive-Job -Job $_ -Wait
    Remove-Job  -Job $_
} | Where-Object { $_ -ne $null }

$TestDuration = ((Get-Date) - $StartTime).TotalSeconds

# -------------------------------------------------------
#  STATS
# -------------------------------------------------------
$Total      = $AllResults.Count
$Successful = ($AllResults | Where-Object { $_.Success }).Count
$Failed     = $Total - $Successful
$ActualTPS  = if ($TestDuration -gt 0) { [math]::Round($Total / $TestDuration, 1) } else { 0 }
$SuccTPS    = if ($TestDuration -gt 0) { [math]::Round($Successful / $TestDuration, 1) } else { 0 }

$FraudDetected = ($AllResults | Where-Object { $_.IsFraud -eq $true }).Count
$FraudRate     = if ($Successful -gt 0) { [math]::Round($FraudDetected / $Successful * 100, 1) } else { 0 }

$Critical = ($AllResults | Where-Object { $_.RiskLevel -eq "CRITICAL" }).Count
$High     = ($AllResults | Where-Object { $_.RiskLevel -eq "HIGH" }).Count
$Medium   = ($AllResults | Where-Object { $_.RiskLevel -eq "MEDIUM" }).Count
$Low      = ($AllResults | Where-Object { $_.RiskLevel -eq "LOW" }).Count

$SuccLatencies = ($AllResults | Where-Object { $_.Success } | Select-Object -ExpandProperty Latency) | Sort-Object
$MinLat = 0; $AvgLat = 0; $MaxLat = 0
$P50 = "N/A"; $P95 = "N/A"; $P99 = "N/A"

if ($SuccLatencies -and $SuccLatencies.Count -gt 0) {
    $MinLat = [math]::Round(($SuccLatencies | Measure-Object -Minimum).Minimum, 2)
    $AvgLat = [math]::Round(($SuccLatencies | Measure-Object -Average).Average, 2)
    $MaxLat = [math]::Round(($SuccLatencies | Measure-Object -Maximum).Maximum, 2)
    $n   = $SuccLatencies.Count
    $P50 = "$([math]::Round($SuccLatencies[[math]::Floor($n * 0.50)], 2))ms"
    $P95 = "$([math]::Round($SuccLatencies[[math]::Floor([math]::Min($n * 0.95, $n - 1))], 2))ms"
    $P99 = "$([math]::Round($SuccLatencies[[math]::Floor([math]::Min($n * 0.99, $n - 1))], 2))ms"
}

$ExpectedFraudRange = "~40-50% (4 fraud vectors / 8 total)"

# -------------------------------------------------------
#  MONGODB AUDIT LOG CHECK
# -------------------------------------------------------
Write-Host ""
Write-Host "  Checking MongoDB audit log..." -ForegroundColor Gray
try {
    $TxnResp = Invoke-RestMethod -Uri "$BaseUrl/transactions/?size=5" `
        -Headers @{ "Authorization" = "Bearer $Token" } -ErrorAction Stop
    $MongoCount = $TxnResp.total
    Write-Host ("  MongoDB transaction_logs total docs: {0}" -f $MongoCount) -ForegroundColor Green
} catch {
    Write-Host "  Could not query MongoDB audit log: $_" -ForegroundColor Yellow
}

# -------------------------------------------------------
#  PRINT RESULTS
# -------------------------------------------------------
Write-Host ""
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "   LOAD TEST RESULTS" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  THROUGHPUT" -ForegroundColor Yellow
Write-Host ("  Total requests  : {0}" -f $Total)
Write-Host ("  Successful      : {0}" -f $Successful)
Write-Host ("  Failed          : {0}" -f $Failed)
Write-Host ("  Test duration   : {0}s" -f [math]::Round($TestDuration, 1))
Write-Host ("  Actual TPS      : {0}" -f $ActualTPS)
Write-Host ("  Successful TPS  : {0}" -f $SuccTPS)
Write-Host ""
Write-Host "  LATENCY (ms)" -ForegroundColor Yellow
Write-Host ("  Min             : {0}ms" -f $MinLat)
Write-Host ("  Average         : {0}ms" -f $AvgLat)
Write-Host ("  P50 (median)    : {0}" -f $P50)
Write-Host ("  P95             : {0}" -f $P95)
Write-Host ("  P99             : {0}" -f $P99)
Write-Host ("  Max             : {0}ms" -f $MaxLat)
Write-Host ""
Write-Host "  FRAUD DETECTION" -ForegroundColor Yellow
Write-Host ("  Fraud detected  : {0}" -f $FraudDetected)
Write-Host ("  Fraud rate      : {0}%" -f $FraudRate)
Write-Host ("  Expected range  : {0}" -f $ExpectedFraudRange)
Write-Host ""
Write-Host "  RISK LEVEL BREAKDOWN" -ForegroundColor Yellow
Write-Host ("  CRITICAL        : {0}" -f $Critical)
Write-Host ("  HIGH            : {0}" -f $High)
Write-Host ("  MEDIUM          : {0}" -f $Medium)
Write-Host ("  LOW             : {0}" -f $Low)
Write-Host ""
Write-Host "  MONGODB AUDIT" -ForegroundColor Yellow
Write-Host ("  Total logged    : {0} transactions" -f $MongoCount)
Write-Host "=======================================================" -ForegroundColor Cyan
