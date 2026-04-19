#!/bin/bash
# macOS AppleScript to send keystrokes to Wine/TORCS window
osascript <<EOF
tell application "System Events"
    key code 36         -- Return
    delay 0.1
    key code 36         -- Return
    delay 0.1
    key code 126        -- Up arrow
    delay 0.1
    key code 126        -- Up arrow
    delay 0.1
    key code 36         -- Return
    delay 0.1
    key code 36         -- Return
end tell
EOF
