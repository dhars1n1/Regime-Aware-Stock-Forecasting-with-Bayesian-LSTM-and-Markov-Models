@echo off
echo ============================================================================
echo ENABLING WINDOWS LONG PATH SUPPORT (Administrator Required)
echo ============================================================================

echo.
echo 1. Enabling Long Path Support in Registry...
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f

if %errorlevel% == 0 (
    echo    ✅ Long Path support enabled successfully!
    echo    ⚠️  Please RESTART your computer for changes to take effect
) else (
    echo    ❌ Failed to enable Long Path support
    echo    💡 Make sure you're running as Administrator
)

echo.
echo 2. Next Steps:
echo    - Restart your computer
echo    - Run: .\venv\Scripts\Activate.ps1
echo    - Run: pip install tensorflow-cpu==2.16.1 --no-cache-dir
echo.
pause