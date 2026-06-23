param(
    [string]$StopFile = "",
    [int]$HotTempC = 90,
    [int]$CoolTempC = 85,
    [int]$HotCpuMax = 50,
    [int]$NormalCpuMax = 100,
    [int]$PollSeconds = 5
)

$ErrorActionPreference = "SilentlyContinue"

$LogFile = Join-Path (Get-Location) "thermal_guard.log"

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

$SUB_PROCESSOR = "54533251-82be-4824-96c1-47b60b740d00"
$PROC_MIN      = "893dee8e-2bef-41e0-89c6-b55d0929964c"
$PROC_MAX      = "bc5038f7-23e0-4960-96da-33abaf5935ec"
$COOLING       = "94d3a615-a899-4ac5-ae2b-e4d8f634367f"

function Get-ActiveSchemeGuid {
    $line = powercfg /getactivescheme
    return [regex]::Match($line, "[0-9a-fA-F-]{36}").Value
}

function Set-HighPerformancePlan {
    $highPerfBase = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"

    $hpLine = powercfg /l | Where-Object { $_ -match "High performance" } | Select-Object -First 1

    if (-not $hpLine) {
        $created = powercfg -duplicatescheme $highPerfBase
        $m = [regex]::Match(($created | Out-String), "[0-9a-fA-F-]{36}")
        if ($m.Success) {
            $hp = $m.Value
        } else {
            $hp = $highPerfBase
        }
    } else {
        $hp = [regex]::Match($hpLine, "[0-9a-fA-F-]{36}").Value
    }

    powercfg /s $hp | Out-Null
    return $hp
}

function Apply-PerformanceNoSleep($scheme) {
    # Screen off after 5 min, but system does not sleep
    powercfg /change monitor-timeout-ac 5 | Out-Null
    powercfg /change standby-timeout-ac 0 | Out-Null
    powercfg /change hibernate-timeout-ac 0 | Out-Null
    powercfg /change disk-timeout-ac 0 | Out-Null

    # CPU: sane performance mode, not forced idle 100%
    powercfg /setacvalueindex $scheme $SUB_PROCESSOR $PROC_MIN 5 | Out-Null
    powercfg /setacvalueindex $scheme $SUB_PROCESSOR $PROC_MAX $NormalCpuMax | Out-Null

    # Active cooling = fan-first policy where firmware supports it
    powercfg /setacvalueindex $scheme $SUB_PROCESSOR $COOLING 1 | Out-Null

    powercfg /s $scheme | Out-Null
}

function Set-CpuMax($scheme, $percent) {
    powercfg /setacvalueindex $scheme $SUB_PROCESSOR $PROC_MAX $percent | Out-Null
    powercfg /s $scheme | Out-Null
}

function Get-CpuTempC {
    try {
        $sensors = Get-CimInstance -Namespace root\LibreHardwareMonitor -ClassName Sensor -ErrorAction Stop |
            Where-Object {
                $_.SensorType -eq "Temperature" -and
                (
                    $_.Name -like "*CPU Package*" -or
                    $_.Name -like "*CPU Core*" -or
                    $_.Name -like "*Tctl*" -or
                    $_.Name -like "*Tdie*"
                )
            }

        if ($sensors) {
            return [math]::Round(($sensors | Measure-Object Value -Maximum).Maximum, 1)
        }
    } catch {}

    try {
        $sensors = Get-CimInstance -Namespace root\OpenHardwareMonitor -ClassName Sensor -ErrorAction Stop |
            Where-Object {
                $_.SensorType -eq "Temperature" -and
                (
                    $_.Name -like "*CPU Package*" -or
                    $_.Name -like "*CPU Core*" -or
                    $_.Name -like "*Tctl*" -or
                    $_.Name -like "*Tdie*"
                )
            }

        if ($sensors) {
            return [math]::Round(($sensors | Measure-Object Value -Maximum).Maximum, 1)
        }
    } catch {}

    try {
        $temps = Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction Stop |
            ForEach-Object { ($_.CurrentTemperature / 10) - 273.15 }

        if ($temps) {
            return [math]::Round(($temps | Measure-Object -Maximum).Maximum, 1)
        }
    } catch {}

    return $null
}

Add-Type @"
using System;
using System.Runtime.InteropServices;

public class Awake {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

$ES_CONTINUOUS = 0x80000000
$ES_SYSTEM_REQUIRED = 0x00000001

$scheme = Set-HighPerformancePlan
Apply-PerformanceNoSleep $scheme

$throttled = $false

Log "THERMAL GUARD STARTED"
Log "Power plan set to High Performance: $scheme"
Log "Screen may turn off, but sleep/hibernate are disabled on AC."
Log "Active cooling policy enabled continuously."
Log "CPU > ${HotTempC}C => cap to ${HotCpuMax}%"
Log "CPU < ${CoolTempC}C => restore to ${NormalCpuMax}%"

try {
    while ($true) {
        if ($StopFile -and (Test-Path $StopFile)) {
            Log "Stop file detected. Exiting thermal guard."
            break
        }

        [Awake]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null

        Apply-PerformanceNoSleep $scheme

        $temp = Get-CpuTempC

        if ($null -eq $temp) {
            Log "CPU temp unavailable. Active cooling + no-sleep still active. For accurate temp, run LibreHardwareMonitor."
        }
        elseif (-not $throttled -and $temp -ge $HotTempC) {
            Set-CpuMax $scheme $HotCpuMax
            $throttled = $true
            Log "HOT: CPU ${temp}C -> CPU capped to ${HotCpuMax}%"
        }
        elseif ($throttled -and $temp -le $CoolTempC) {
            Set-CpuMax $scheme $NormalCpuMax
            $throttled = $false
            Log "COOLED: CPU ${temp}C -> CPU restored to ${NormalCpuMax}%"
        }
        else {
            if ($throttled) {
                Log "CPU ${temp}C | THROTTLED ${HotCpuMax}% | active cooling"
            } else {
                Log "CPU ${temp}C | NORMAL ${NormalCpuMax}% | active cooling"
            }
        }

        Start-Sleep -Seconds $PollSeconds
    }
}
finally {
    Set-CpuMax $scheme $NormalCpuMax
    [Awake]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
    Log "THERMAL GUARD STOPPED. CPU max restored to ${NormalCpuMax}%."
}
