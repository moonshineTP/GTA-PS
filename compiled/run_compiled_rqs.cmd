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
    echo   run_compiled_rqs.cmd [path_to_exes] [args...]
    echo.
    echo Examples:
    echo   run_compiled_rqs.cmd
    echo   run_compiled_rqs.cmd D:\executables
    echo   run_compiled_rqs.cmd "C:\Program Files\GTA-PS\exes"
    echo.
    echo Or set environment variable:
    echo   set EXES_DIR_ENV=D:\path\to\exes
    exit /b 1
)

echo ======================================================
echo GTA-PS Compiled RQs Suite
echo ======================================================
echo Using executables from: !EXES_DIR!
echo.
echo GTA-PS Compiled RQs Suite
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

call :RunRQ "RQ1 Proactiveness" "rq1_proactiveness.exe"
call :RunRQ "RQ1 DP Timeline" "rq1_dp_timeline.exe"
call :RunRQ "RQ1 Dyna Timeline" "rq1_dyna_timeline.exe"
call :RunRQ "RQ2 Allocation" "rq2_allocation.exe"
call :RunRQ "RQ2 Beta" "rq2_beta.exe"
call :RunRQ "RQ3 Tradeoff" "rq3_tradeoff.exe"
call :RunRQ "RQ4 Asymmetry" "rq4_asymmetry.exe"
call :RunRQ "RQ5 Bottlenecks" "rq5_bottlenecks.exe"
call :RunRQ "RQ6 Curl Divergence" "rq6_curl_divergence.exe"

echo Done.
endlocal
exit /b 0

:RunRQ
set "LABEL=%~1"
set "EXE_NAME=%~2"
set /a "INDEX+=1"

if !INDEX! LSS !START_FROM! (
    echo [!LABEL!] - skipped ^(start-from=!START_FROM!^)
    exit /b 0
)

echo [!INDEX!/9] !LABEL!

:: Try to run executable directly from EXES_DIR
if exist "!EXES_DIR!\!EXE_NAME!" (
    "!EXES_DIR!\!EXE_NAME!" !USER_ARGS!
    exit /b 0
)

:: Try with .exe extension
if exist "!EXES_DIR!\!EXE_NAME!.exe" (
    "!EXES_DIR!\!EXE_NAME!.exe" !USER_ARGS!
    exit /b 0
)

:: Try artifacts structure: EXES_DIR\name.dist\name.exe
set "DIST_NAME=!EXE_NAME:.exe=.dist!"
if exist "!EXES_DIR!\!DIST_NAME!\!EXE_NAME!" (
    "!EXES_DIR!\!DIST_NAME!\!EXE_NAME!" !USER_ARGS!
    exit /b 0
)

:: Try artifacts structure without .exe
if exist "!EXES_DIR!\!DIST_NAME!\!DIST_NAME!" (
    "!EXES_DIR!\!DIST_NAME!\!DIST_NAME!" !USER_ARGS!
    exit /b 0
)

echo ERROR: !EXE_NAME! not found
echo Searched in:
echo   - !EXES_DIR!\!EXE_NAME!
echo   - !EXES_DIR!\!EXE_NAME!.exe
echo   - !EXES_DIR!\!DIST_NAME!\!EXE_NAME!
echo.
exit /b 1
