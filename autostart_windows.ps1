Add-Type -AssemblyName System.Windows.Forms

Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Threading;

public static class WinApi {
    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern bool ShowWindowAsync(IntPtr hWnd, int nCmdShow);

    [DllImport("user32.dll", SetLastError = true)]
    public static extern uint SendInput(uint nInputs, INPUT[] pInputs, int cbSize);

    [DllImport("user32.dll")]
    public static extern IntPtr GetForegroundWindow();

    [DllImport("user32.dll")]
    public static extern bool IsIconic(IntPtr hWnd);

    public struct INPUT {
        public uint type;
        public INPUTUNION u;
    }

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit)]
    public struct INPUTUNION {
        [System.Runtime.InteropServices.FieldOffset(0)]
        public KEYBDINPUT ki;
    }

    public struct KEYBDINPUT {
        public ushort wVk;
        public ushort wScan;
        public uint dwFlags;
        public uint time;
        public IntPtr dwExtraInfo;
    }

    public static void SendKey(ushort vk) {
        var down = new INPUT { type = 1 };
        down.u.ki.wVk = vk;
        down.u.ki.dwFlags = 0;

        var up = new INPUT { type = 1 };
        up.u.ki.wVk = vk;
        up.u.ki.dwFlags = 2; // KEYEVENTF_KEYUP

        SendInput(1, new INPUT[] { down }, System.Runtime.InteropServices.Marshal.SizeOf(typeof(INPUT)));
        Thread.Sleep(50);
        SendInput(1, new INPUT[] { up }, System.Runtime.InteropServices.Marshal.SizeOf(typeof(INPUT)));
    }
}
"@

function Get-TorcsWindowProcess {
    $candidates = @(Get-Process -Name wtorcs, torcs -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne [IntPtr]::Zero })
    if (-not $candidates -or $candidates.Count -eq 0) { return $null }
    return ($candidates | Sort-Object StartTime -Descending | Select-Object -First 1)
}

function Focus-TorcsWindow {
    param([int]$Attempts = 50, [int]$SleepMs = 200)
    for ($i = 0; $i -lt $Attempts; $i++) {
        $proc = Get-TorcsWindowProcess
        if ($null -ne $proc) {
            if ([WinApi]::IsIconic($proc.MainWindowHandle)) {
                [void][WinApi]::ShowWindowAsync($proc.MainWindowHandle, 9)
            }
            Start-Sleep -Milliseconds 100
            [void][WinApi]::SetForegroundWindow($proc.MainWindowHandle)
            Start-Sleep -Milliseconds 300
            $fg = [WinApi]::GetForegroundWindow()
            if ($fg -eq $proc.MainWindowHandle) {
                Write-Host "[autostart] TORCS window focused (handle=$($proc.MainWindowHandle))"
                return $true
            }
            [void][WinApi]::SetForegroundWindow($proc.MainWindowHandle)
            Start-Sleep -Milliseconds 200
        } else {
            Start-Sleep -Milliseconds $SleepMs
        }
    }
    Write-Host "[autostart] Failed to focus TORCS window"
    return $false
}

if (-not (Focus-TorcsWindow)) {
    exit 1
}

Start-Sleep -Milliseconds 500

# Key sequence: Enter, Enter, Up, Up, Enter, Enter
# VK codes: Enter=0x0D, Up=0x26
$sequence = @(0x0D, 0x0D, 0x26, 0x26, 0x0D, 0x0D)
$sendKeysFallback = @{ 13 = "{ENTER}"; 38 = "{UP}" }
foreach ($vk in $sequence) {
    # Keep focus sticky before each key for game window menus.
    [void](Focus-TorcsWindow -Attempts 3 -SleepMs 30)
    [WinApi]::SendKey([ushort]$vk)

    # Fallback for cases where DirectInput path misses menu navigation.
    if ($sendKeysFallback.ContainsKey([int]$vk)) {
        [System.Windows.Forms.SendKeys]::SendWait($sendKeysFallback[[int]$vk])
    }
    Start-Sleep -Milliseconds 30
}
Write-Host "[autostart] Key sequence sent"