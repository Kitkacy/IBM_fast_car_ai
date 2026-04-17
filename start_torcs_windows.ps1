param(
    [string]$TorcsExe = "C:\Users\szymo\source\repos\torcs\torcs\wtorcs.exe",
    [string]$AutostartScript = "",
    [string]$VisionFlag = "novision",
    [string]$LogFile = "",
    [string]$LaunchMode = "background"
)

if ([string]::IsNullOrWhiteSpace($LogFile)) {
    $LogFile = Join-Path $PSScriptRoot "torcs_launcher.log"
}

$torcsPath = [System.IO.Path]::GetFullPath($TorcsExe)
$torcsDir = Split-Path -Path $torcsPath -Parent
$visionEnabled = $false
if ($VisionFlag -and $VisionFlag.ToLowerInvariant() -eq "vision") {
    $visionEnabled = $true
}
$launchModeNormalized = if ($LaunchMode) { $LaunchMode.ToLowerInvariant() } else { "background" }

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class WinApi {
    [DllImport("user32.dll")]
    public static extern bool ShowWindowAsync(IntPtr hWnd, int nCmdShow);
}
"@ -ErrorAction SilentlyContinue

function Write-LauncherLog {
    param([string]$Message)
    Add-Content -Path $LogFile -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff')] $Message"
}

Write-LauncherLog "begin launcher"
Write-LauncherLog "TORCS_EXE=$torcsPath"
Write-LauncherLog "VISION_FLAG=$VisionFlag"
Write-LauncherLog "LAUNCH_MODE=$launchModeNormalized"

# Always try to kill before launching; ignore errors when process is not running.
taskkill /IM wtorcs.exe /F /T *> $null
taskkill /IM torcs.exe /F /T *> $null
Start-Sleep -Milliseconds 700

$args = @("-nofuel", "-nodamage", "-nolaptime")
if ($visionEnabled) {
    $args += "-vision"
}

Start-Process -FilePath $torcsPath -WorkingDirectory $torcsDir -ArgumentList $args | Out-Null
Write-LauncherLog "launch command dispatched"

Start-Sleep -Milliseconds 1100

if ($AutostartScript -and (Test-Path -LiteralPath $AutostartScript)) {
    Write-LauncherLog "running autostart script: $AutostartScript"
    & cmd /c "$AutostartScript" *> $null
    Write-LauncherLog "autostart script finished"
}
else {
    Write-LauncherLog "autostart script missing or empty"
}

if ($launchModeNormalized -eq "foreground") {
    # Minimize after autostart so input automation can still focus/control the window.
    Start-Sleep -Milliseconds 200
    $p = Get-Process wtorcs, torcs -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowHandle -ne 0 } |
        Sort-Object StartTime -Descending |
        Select-Object -First 1
    if ($p) {
        [void][WinApi]::ShowWindowAsync($p.MainWindowHandle, 6)
        Write-LauncherLog "window minimized for background mode"
    }
    else {
        Write-LauncherLog "no TORCS window found to minimize"
    }
}

Write-LauncherLog "launcher completed"
