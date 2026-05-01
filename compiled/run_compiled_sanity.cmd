@echo off
setlocal

:: Capture all user arguments once at the start
set "USER_ARGS=%*"

:: Path to the artifacts directory relative to this script
set "ARTIFACTS_DIR=%~dp0artifacts"

echo ======================================================
echo GTA-PS Compiled Sanity Suite (Benchmarks Only)
echo ======================================================

:: 1. Time Benchmark
echo [1/2] Time Benchmark (Compiled)
set "TIME_EXE=%ARTIFACTS_DIR%\time_benchmark.dist\time_benchmark.exe"
if exist "%TIME_EXE%" (
    "%TIME_EXE%" %USER_ARGS%
) else if exist "%ARTIFACTS_DIR%\time_benchmark.build" (
    echo [STATUS] Native compilation is currently IN PROGRESS.
    echo          Large math libraries are being transformed to C.
    echo          Run 'code\compiled\monitor_build.cmd' in another window to see progress.
) else (
    echo WARNING: Compiled time_benchmark not found.
)
echo.

:: 2. Convergence Benchmark
echo [2/2] Convergence Benchmark (Compiled)
set "CONV_EXE=%ARTIFACTS_DIR%\convergence_benchmark.dist\convergence_benchmark.exe"
if exist "%CONV_EXE%" (
    "%CONV_EXE%" %USER_ARGS%
) else if exist "%ARTIFACTS_DIR%\convergence_benchmark.build" (
    echo [STATUS] Native compilation is currently IN PROGRESS.
) else (
    echo WARNING: Compiled convergence_benchmark not found.
)
echo.

echo Done.
endlocal
