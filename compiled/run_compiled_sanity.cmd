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
if exist "%ARTIFACTS_DIR%\time_benchmark.dist\time_benchmark.exe" (
    "%ARTIFACTS_DIR%\time_benchmark.dist\time_benchmark.exe" %USER_ARGS%
) else (
    echo ERROR: time_benchmark not found
)
echo.

:: 2. Convergence Benchmark
echo [2/2] Convergence Benchmark (Compiled)
if exist "%ARTIFACTS_DIR%\convergence_benchmark.dist\convergence_benchmark.exe" (
    "%ARTIFACTS_DIR%\convergence_benchmark.dist\convergence_benchmark.exe" %USER_ARGS%
) else (
    echo ERROR: convergence_benchmark not found
)
echo.

:: 2. Convergence Benchmark
echo [2/2] Convergence Benchmark (Compiled)
if exist "%ARTIFACTS_DIR%\convergence_benchmark.dist\convergence_benchmark.exe" (
    "%ARTIFACTS_DIR%\convergence_benchmark.dist\convergence_benchmark.exe" %USER_ARGS%
) else (
    echo ERROR: convergence_benchmark not found
)
echo.

echo Done.
endlocal
