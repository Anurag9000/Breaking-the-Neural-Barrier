$ErrorActionPreference = "SilentlyContinue"

$RepoRoot = "D:\Projects\Breaking-the-Neural-Barrier"
$LogFile = Join-Path $RepoRoot "awake_guard.log"

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

[uint32]$ES_CONTINUOUS        = 0x80000000
[uint32]$ES_SYSTEM_REQUIRED   = 0x00000001
[uint32]$ES_AWAYMODE_REQUIRED = 0x00000040

[uint32]$flags = $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQUIRED

Log "STRONG AWAKE GUARD STARTED"
Log "Flags: $flags"
Log "Display may turn off. System sleep/modern-standby should be blocked."

while ($true) {
    $ret = [Awake]::SetThreadExecutionState($flags)

    powercfg /change standby-timeout-ac 0 | Out-Null
    powercfg /change hibernate-timeout-ac 0 | Out-Null
    powercfg /change disk-timeout-ac 0 | Out-Null

    Log "Awake lock refreshed. Return=$ret"

    Start-Sleep -Seconds 20
}
