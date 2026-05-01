@echo off
setlocal enabledelayedexpansion

:: Capture all user arguments once at the start
set "USER_ARGS=%*"

:: Set PYTHONPATH for executables that need it
set "CODE_ROOT=%~dp0.."
set "PYTHONPATH=%CODE_ROOT%\src;%CODE_ROOT%\threads;%CODE_ROOT%\misc"

:: Find and validate artifacts directory
set "ARTIFACTS_DIR="

:: Strategy 1: Check if ARTIFACTS_DIR is set as environment variable
if defined ARTIFACTS_DIR_ENV (
    set "ARTIFACTS_DIR=!ARTIFACTS_DIR_ENV!"
)

:: Strategy 2: Check relative to script location
if not defined ARTIFACTS_DIR (
    if exist "%~dp0artifacts" (
        set "ARTIFACTS_DIR=%~dp0artifacts"
    )
)

:: Strategy 3: Check in parent directories from current location
if not defined ARTIFACTS_DIR (
    for /d %%D in (artifacts compiled\artifacts ..\artifacts ..\..\artifacts) do (
        if exist "%%~D" (
            set "ARTIFACTS_DIR=%%~D"
            goto :found_artifacts
        )
    )
)

:: Strategy 4: Search from script's parent directory
if not defined ARTIFACTS_DIR (
    if exist "%~dp0..\artifacts" (
        set "ARTIFACTS_DIR=%~dp0..\artifacts"
    )
)

:found_artifacts
if not defined ARTIFACTS_DIR (
    echo ERROR: Cannot find artifacts directory
    echo.
    echo Searched in:
    echo   - Script directory: %~dp0artifacts
    echo   - Current directory: artifacts, compiled\artifacts, ..\artifacts, ..\..\artifacts
    echo   - Parent of script: %~dp0..\artifacts
    echo.
    echo Solution: Set ARTIFACTS_DIR_ENV environment variable to point to the artifacts folder
    echo   Example: set ARTIFACTS_DIR_ENV=D:\path\to\artifacts
    exit /b 1
)

echo ======================================================
echo GTA-PS Compiled Sanity Suite (Benchmarks Only)
echo ======================================================
echo Using artifacts from: !ARTIFACTS_DIR!
echo.
echo GTA-PS Compiled Sanity Suite
echo ======================================================
echo GTA-PS Compiled Sanity Suite
echo ======================================================


:: 1. Time Benchmark
echo [1/2] Time Benchmark (Compiled)
if exist "!ARTIFACTS_DIR!\time_benchmark.dist\time_benchmark.exe" (
    "!ARTIFACTS_DIR!\time_benchmark.dist\time_benchmark.exe" !USER_ARGS!
) else if exist "!ARTIFACTS_DIR!\time_benchmark.dist\time_benchmark" (
    "!ARTIFACTS_DIR!\time_benchmark.dist\time_benchmark" !USER_ARGS!
) else (
    echo ERROR: time_benchmark not found in !ARTIFACTS_DIR!\time_benchmark.dist
    exit /b 1
)
echo.

:: 2. Convergence Benchmark
echo [2/2] Convergence Benchmark (Compiled)
if exist "!ARTIFACTS_DIR!\convergence_benchmark.dist\convergence_benchmark.exe" (
    "!ARTIFACTS_DIR!\convergence_benchmark.dist\convergence_benchmark.exe" !USER_ARGS!
) else if exist "!ARTIFACTS_DIR!\convergence_benchmark.dist\convergence_benchmark" (
    "!ARTIFACTS_DIR!\convergence_benchmark.dist\convergence_benchmark" !USER_ARGS!
) else (
    echo ERROR: convergence_benchmark not found in !ARTIFACTS_DIR!\convergence_benchmark.dist
    exit /b 1
)
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
