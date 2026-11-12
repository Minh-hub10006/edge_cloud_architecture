# Mobile uplink/latency trace for the DRL v3 simulator (docs/PLAN_DRL_V3_VI.md section 13), laptop on a phone hotspot.
#
#   powershell -ExecutionPolicy Bypass -File scripts\net_measure.ps1 -Label 4g_sang -Minutes 30
#
# Writes results\net\<Label>_speedtest.csv (Ookla CLI, nearest server: upload/download Mbit/s, ping, jitter, about
# one test per 30 s) and results\net\<Label>_ping.csv (one ping per second to 8.8.8.8, measured while the speed tests
# load the link, as a camera uploading would). Needs speedtest.exe (Ookla CLI) in tools\speedtest\.
param([string]$Label = "4g", [int]$Minutes = 30, [string]$PingHost = "8.8.8.8")

$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root "tools\speedtest\speedtest.exe"
if (-not (Test-Path $exe)) { throw "missing $exe (download the Ookla CLI first)" }
$out = Join-Path $root "results\net"
New-Item -ItemType Directory -Force $out | Out-Null
$speedCsv = Join-Path $out "${Label}_speedtest.csv"
$pingCsv = Join-Path $out "${Label}_ping.csv"
$end = (Get-Date).AddMinutes($Minutes)

# Ping job: one line per second, "utc,ms" (ms empty on loss).
$job = Start-Job -ArgumentList $pingCsv, $PingHost, $end -ScriptBlock {
    param($file, $target, $until)
    "utc,ping_ms" | Out-File -Encoding utf8 $file
    while ((Get-Date) -lt $until) {
        $t = (Get-Date).ToUniversalTime().ToString("o")
        $r = Test-Connection -ComputerName $target -Count 1 -ErrorAction SilentlyContinue
        "$t,$(if ($r) { $r.ResponseTime } else { '' })" | Out-File -Append -Encoding utf8 $file
        Start-Sleep -Milliseconds (1000 - [Math]::Min(999, $(if ($r) { $r.ResponseTime } else { 999 })))
    }
}

$n = 0
while ((Get-Date) -lt $end) {
    $lines = @(& $exe --accept-license --accept-gdpr --format=csv $(if ($n -eq 0) { "--output-header" }) 2>$null)
    if ($lines.Count -gt 0) {
        if ($n -eq 0) { "`"utc`",$($lines[0])" | Out-File -Encoding utf8 $speedCsv; $lines = $lines[1..($lines.Count - 1)] }
        foreach ($l in $lines) { "$((Get-Date).ToUniversalTime().ToString('o')),$l" | Out-File -Append -Encoding utf8 $speedCsv; $n++ }
    }
    Write-Host "$(Get-Date -Format HH:mm:ss) speedtest $n done"
    Start-Sleep -Seconds 10
}
Wait-Job $job | Out-Null
Remove-Job $job
Write-Host "DONE $n speed tests -> $speedCsv ; ping -> $pingCsv"
