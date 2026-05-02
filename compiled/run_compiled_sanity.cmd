@echo off
setlocal enabledelayedexpansion

set "USER_ARGS=%*"
set "CODE_ROOT=%~dp0.."
set "PYTHONPATH=%CODE_ROOT%\src;%CODE_ROOT%\threads;%CODE_ROOT%\misc"
set "ARTIFACTS_DIR=%~dp0artifacts"

if not exist "!ARTIFACTS_DIR!" (
    echo ERROR: artifacts directory not found at !ARTIFACTS_DIR!
    exit /b 1
)

echo ======================================================
echo GTA-PS Compiled Sanity Suite
echo ======================================================
echo GTA-PS Compiled Sanity Suite
echo ======================================================

:: 1. Time Benchmark
echo [1/2] Time Benchmark (Compiled)
"!ARTIFACTS_DIR!\time_benchmark.dist\time_benchmark.exe" !USER_ARGS!
echo.

:: 2. Convergence Benchmark
echo [2/2] Convergence Benchmark (Compiled)
"!ARTIFACTS_DIR!\convergence_benchmark.dist\convergence_benchmark.exe" !USER_ARGS!
echo.

:: 2. Convergence Benchmark
echo [2/2] Convergence Benchmark (Compiled)
if exist "%ARTIFACTS_DIR%\convergence_benchmark.dist\convergence_benchmark.exe" (
    "%ARTIFACTS_DIR%\convergence_benchmark.dist\convergence_benchmark.exe" %USER_ARGS%
) else if exist "%ARTIFACTS_DIR%\convergence_benchmark.dist\convergence_benchmark" (
    "%ARTIFACTS_DIR%\convergence_benchmark.dist\convergence_benchmark" %USER_ARGS%
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
