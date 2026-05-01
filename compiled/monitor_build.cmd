@echo off
setlocal
echo ======================================================
echo GTA-PS Native Build Monitor
echo ======================================================
echo Press Ctrl+C to stop monitoring. The build will continue in background.
echo.
:loop
cls
echo [Build Log: code\compiled\build_benchmarks.log]
echo Last 15 lines:
echo ------------------------------------------------------
powershell -Command "Get-Content code\compiled\build_benchmarks.log -Tail 15"
echo ------------------------------------------------------
echo Status: Building (Size of artifacts: )
dir /s code\compiled\artifacts\time_benchmark.build | find "File(s)"
timeout /t 5 > nul
goto loop
