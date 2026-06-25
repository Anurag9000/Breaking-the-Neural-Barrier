$ErrorActionPreference = "SilentlyContinue"

$LogFile = Join-Path (Get-Location) "awake_guard.log"

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

Add-Type @"
using System;
using System.Runtime.InteropServices;

public class Awake {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

$ES_CONTINUOUS        = 0x80000000
$ES_SYSTEM_REQUIRED   = 0x00000001
$ES_AWAYMODE_REQUIRED = 0x00000040

Log "STRONG AWAKE GUARD STARTED"
Log "Display may turn off. System sleep/modern-standby should be blocked."

while ($true) {
    [Awake]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQUIRED) | Out-Null

    powercfg /change standby-timeout-ac 0 | Out-Null
    powercfg /change hibernate-timeout-ac 0 | Out-Null
    powercfg /change disk-timeout-ac 0 | Out-Null

    Start-Sleep -Seconds 20
}
