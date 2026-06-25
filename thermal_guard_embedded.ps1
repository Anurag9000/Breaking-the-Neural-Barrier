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

function Ensure-HighPerformance {
    $hpBase = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
    $line = powercfg /l | Where-Object { $_ -match "High performance" } | Select-Object -First 1

    if ($line) {
        $hp = [regex]::Match($line, "[0-9a-fA-F-]{36}").Value
    } else {
        $created = powercfg -duplicatescheme $hpBase
        $hp = [regex]::Match(($created | Out-String), "[0-9a-fA-F-]{36}").Value
        if (-not $hp) { $hp = $hpBase }
    }

    powercfg /s $hp | Out-Null
    return $hp
}

function Apply-RunPowerPolicy($scheme) {
    # Display off after 5 min; system awake
    powercfg /change monitor-timeout-ac 5 | Out-Null
    powercfg /change standby-timeout-ac 0 | Out-Null
    powercfg /change hibernate-timeout-ac 0 | Out-Null
    powercfg /change disk-timeout-ac 0 | Out-Null

    # Sleep/hibernate/hybrid/unattended sleep disabled
    powercfg /setacvalueindex $scheme 238c9fa8-0aad-41ed-83f4-97be242c8f20 29f6c1db-86da-48c5-9fdb-f2b67b1f44da 0 | Out-Null
    powercfg /setacvalueindex $scheme 238c9fa8-0aad-41ed-83f4-97be242c8f20 9d7815a6-7ee4-497e-8888-515a05f02364 0 | Out-Null
    powercfg /setacvalueindex $scheme 238c9fa8-0aad-41ed-83f4-97be242c8f20 94ac6d29-73ce-41a6-809f-6363ba21b47e 0 | Out-Null
    powercfg /setacvalueindex $scheme 238c9fa8-0aad-41ed-83f4-97be242c8f20 7bc4a2f9-d8fc-4469-b07b-33eb785aaca0 0 | Out-Null
    powercfg /setacvalueindex $scheme 238c9fa8-0aad-41ed-83f4-97be242c8f20 25dfa149-5dd1-4736-b5ab-e8a37b5b8187 1 | Out-Null

    # Lid close on charger = do nothing
    powercfg /setacvalueindex $scheme 4f971e89-eebd-4455-a8de-9e59040e7347 5ca83367-6e45-459f-a27b-476b1d01c9360 0 | Out-Null

    # CPU sane performance + active cooling
    powercfg /setacvalueindex $scheme $SUB_PROCESSOR $PROC_MIN 5 | Out-Null
    powercfg /setacvalueindex $scheme $SUB_PROCESSOR $PROC_MAX $NormalCpuMax | Out-Null
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
                ($_.Name -like "*CPU Package*" -or $_.Name -like "*CPU Core*" -or $_.Name -like "*Tctl*" -or $_.Name -like "*Tdie*")
            }
        if ($sensors) { return [math]::Round(($sensors | Measure-Object Value -Maximum).Maximum, 1) }
    } catch {}

    try {
        $sensors = Get-CimInstance -Namespace root\OpenHardwareMonitor -ClassName Sensor -ErrorAction Stop |
            Where-Object {
                $_.SensorType -eq "Temperature" -and
                ($_.Name -like "*CPU Package*" -or $_.Name -like "*CPU Core*" -or $_.Name -like "*Tctl*" -or $_.Name -like "*Tdie*")
            }
        if ($sensors) { return [math]::Round(($sensors | Measure-Object Value -Maximum).Maximum, 1) }
    } catch {}

    try {
        $temps = Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction Stop |
            ForEach-Object { ($_.CurrentTemperature / 10) - 273.15 }
        if ($temps) { return [math]::Round(($temps | Measure-Object -Maximum).Maximum, 1) }
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

$ES_CONTINUOUS        = 0x80000000
$ES_SYSTEM_REQUIRED   = 0x00000001
$ES_AWAYMODE_REQUIRED = 0x00000040

$scheme = Ensure-HighPerformance
Apply-RunPowerPolicy $scheme

$throttled = $false

Log "THERMAL GUARD STARTED"
Log "High Performance active: $scheme"
Log "No sleep/hibernate/disk sleep. Display may turn off."
Log "Active cooling/fan-first policy repeatedly applied."
Log "CPU >= ${HotTempC}C -> CPU max ${HotCpuMax}%"
Log "CPU <= ${CoolTempC}C -> CPU max ${NormalCpuMax}%"

try {
    while ($true) {
        if ($StopFile -and (Test-Path $StopFile)) {
            Log "Stop file detected. Exiting thermal guard."
            break
        }

        [Awake]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQUIRED) | Out-Null

        Apply-RunPowerPolicy $scheme

        $temp = Get-CpuTempC

        if ($null -eq $temp) {
            Log "CPU temp unavailable. Active cooling/no-sleep still active. Run LibreHardwareMonitor for accurate temp throttling."
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
