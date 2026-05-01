@echo off
setlocal enabledelayedexpansion

:: If first argument looks like a path, use it as executables directory
if not "%~1"=="" (
    if exist "%~1" (
        set "EXES_DIR=%~1"
        set "USER_ARGS=%~2 %~3 %~4 %~5 %~6 %~7 %~8 %~9"
        goto :found_exes
    )
)

:: Otherwise capture all args as user arguments
set "USER_ARGS=%*"

:: Set PYTHONPATH for executables that need it
set "CODE_ROOT=%~dp0.."
set "PYTHONPATH=%CODE_ROOT%\src;%CODE_ROOT%\threads;%CODE_ROOT%\misc"

:: Find executables directory
set "EXES_DIR="

:: Strategy 1: Check if EXES_DIR_ENV is set
if defined EXES_DIR_ENV (
    if exist "!EXES_DIR_ENV!" (
        set "EXES_DIR=!EXES_DIR_ENV!"
    )
)

:: Strategy 2: Check for artifacts structure (compiled\artifacts\*.dist\*.exe)
if not defined EXES_DIR (
    if exist "%~dp0artifacts" (
        set "EXES_DIR=%~dp0artifacts"
    )
)

:: Strategy 3: Check parent directories
if not defined EXES_DIR (
    if exist "%~dp0..\artifacts" (
        set "EXES_DIR=%~dp0..\artifacts"
    )
)

if not defined EXES_DIR (
    if exist "%~dp0..\..\artifacts" (
        set "EXES_DIR=%~dp0..\..\artifacts"
    )
)

:found_exes
if not defined EXES_DIR (
    echo ERROR: Cannot find executables
    echo.
    echo Usage:
    echo   run_compiled_sanity.cmd [path_to_exes] [args...]
    echo.
    echo Examples:
    echo   run_compiled_sanity.cmd
    echo   run_compiled_sanity.cmd D:\executables
    echo   run_compiled_sanity.cmd "C:\Program Files\GTA-PS\exes"
    echo.
    echo Or set environment variable:
    echo   set EXES_DIR_ENV=D:\path\to\exes
    exit /b 1
)

echo ======================================================
echo GTA-PS Compiled Sanity Suite (Benchmarks Only)
echo ======================================================
echo Using executables from: !EXES_DIR!
echo.
echo GTA-PS Compiled Sanity Suite
echo ======================================================
echo GTA-PS Compiled Sanity Suite
echo ======================================================


:: 1. Time Benchmark
echo [1/2] Time Benchmark (Compiled)
if exist "!EXES_DIR!\time_benchmark.exe" (
    "!EXES_DIR!\time_benchmark.exe" !USER_ARGS!
) else if exist "!EXES_DIR!\time_benchmark" (
    "!EXES_DIR!\time_benchmark" !USER_ARGS!
) else if exist "!EXES_DIR!\time_benchmark.dist\time_benchmark.exe" (
    "!EXES_DIR!\time_benchmark.dist\time_benchmark.exe" !USER_ARGS!
) else if exist "!EXES_DIR!\time_benchmark.dist\time_benchmark" (
    "!EXES_DIR!\time_benchmark.dist\time_benchmark" !USER_ARGS!
) else (
    echo ERROR: time_benchmark not found in !EXES_DIR!
    exit /b 1
)
echo.

:: 2. Convergence Benchmark
echo [2/2] Convergence Benchmark (Compiled)
if exist "!EXES_DIR!\convergence_benchmark.exe" (
    "!EXES_DIR!\convergence_benchmark.exe" !USER_ARGS!
) else if exist "!EXES_DIR!\convergence_benchmark" (
    "!EXES_DIR!\convergence_benchmark" !USER_ARGS!
) else if exist "!EXES_DIR!\convergence_benchmark.dist\convergence_benchmark.exe" (
    "!EXES_DIR!\convergence_benchmark.dist\convergence_benchmark.exe" !USER_ARGS!
) else if exist "!EXES_DIR!\convergence_benchmark.dist\convergence_benchmark" (
    "!EXES_DIR!\convergence_benchmark.dist\convergence_benchmark" !USER_ARGS!
) else (
    echo ERROR: convergence_benchmark not found in !EXES_DIR!
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
