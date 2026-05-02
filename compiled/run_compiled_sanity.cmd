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

:: Parse --start-from argument
set "START_FROM=1"
set "NEW_USER_ARGS="
for %%a in (!USER_ARGS!) do (
    echo %%a | findstr /r "^--start-from$" >nul 2>&1
    if !errorlevel!==0 (
        set "NEXT_IS_START=1"
    ) else (
        if defined NEXT_IS_START (
            set "START_FROM=%%a"
            set "NEXT_IS_START="
        ) else (
            set "NEW_USER_ARGS=!NEW_USER_ARGS! %%a"
        )
    )
)
set "USER_ARGS=!NEW_USER_ARGS!"
set "INDEX=0"

:: 1. Time Benchmark
set /a "INDEX+=1"
if !INDEX! LSS !START_FROM! (
    echo [!INDEX!/2] Time Benchmark - skipped ^(start-from=!START_FROM!^)
    goto :skip_time
)
echo [!INDEX!/2] Time Benchmark (Compiled)
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
:skip_time

:: 2. Convergence Benchmark
set /a "INDEX+=1"
if !INDEX! LSS !START_FROM! (
    echo [!INDEX!/2] Convergence Benchmark - skipped ^(start-from=!START_FROM!^)
    goto :skip_convergence
)
echo [!INDEX!/2] Convergence Benchmark (Compiled)
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
:skip_convergence

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
