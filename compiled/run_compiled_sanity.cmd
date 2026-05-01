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
"%ARTIFACTS_DIR%\time_benchmark.dist\time_benchmark.exe" %USER_ARGS%
echo.

:: 2. Convergence Benchmark
echo [2/2] Convergence Benchmark (Compiled)
"%ARTIFACTS_DIR%\convergence_benchmark.dist\convergence_benchmark.exe" %USER_ARGS%
echo.

echo Done.
endlocal
