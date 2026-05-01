@echo off
setlocal enabledelayedexpansion

:: Capture all user arguments once at the start
set "USER_ARGS=%*"

:: Set PYTHONPATH for executables that need it
set "CODE_ROOT=%~dp0.."
set "PYTHONPATH=%CODE_ROOT%\src;%CODE_ROOT%\threads;%CODE_ROOT%\misc"

:: Find and validate artifacts directory
set "ARTIFACTS_DIR="

:: Strategy 1: Check if ARTIFACTS_DIR_ENV is set as environment variable
if defined ARTIFACTS_DIR_ENV (
    if exist "!ARTIFACTS_DIR_ENV!" (
        set "ARTIFACTS_DIR=!ARTIFACTS_DIR_ENV!"
    )
)

:: Strategy 2: Check relative to script location (code\compiled\artifacts)
if not defined ARTIFACTS_DIR (
    if exist "%~dp0artifacts" (
        set "ARTIFACTS_DIR=%~dp0artifacts"
    )
)

:: Strategy 3: Check in code directory (code\artifacts)
if not defined ARTIFACTS_DIR (
    if exist "%~dp0..\artifacts" (
        set "ARTIFACTS_DIR=%~dp0..\artifacts"
    )
)

:: Strategy 4: Check in parent workspace (parent\artifacts)
if not defined ARTIFACTS_DIR (
    if exist "%~dp0..\..\artifacts" (
        set "ARTIFACTS_DIR=%~dp0..\..\artifacts"
    )
)

:: Strategy 5: Check further up the tree
if not defined ARTIFACTS_DIR (
    if exist "%~dp0..\..\..\artifacts" (
        set "ARTIFACTS_DIR=%~dp0..\..\..\artifacts"
    )
)

if not defined ARTIFACTS_DIR (
    echo ERROR: Cannot find artifacts directory
    echo.
    echo Searched in:
    echo   - %~dp0artifacts
    echo   - %~dp0..\artifacts
    echo   - %~dp0..\..\artifacts
    echo   - %~dp0..\..\..\artifacts
    echo.
    echo Solution: Set ARTIFACTS_DIR_ENV environment variable or ensure artifacts folder exists
    echo   Example: set ARTIFACTS_DIR_ENV=D:\workspace\artifacts
    exit /b 1
)

echo ======================================================
echo GTA-PS Compiled RQs Suite
echo ======================================================
echo Using artifacts from: !ARTIFACTS_DIR!
echo.
echo GTA-PS Compiled RQs Suite
echo ======================================================


:: List of RQ executables to run
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
set "DIST_NAME=%EXE_NAME:.exe=.dist%"

echo [%LABEL%]
if exist "!ARTIFACTS_DIR!\!DIST_NAME!\!EXE_NAME!" (
    "!ARTIFACTS_DIR!\!DIST_NAME!\!EXE_NAME!" !USER_ARGS!
) else if exist "!ARTIFACTS_DIR!\!DIST_NAME!\!EXE_NAME:exe=!" (
    "!ARTIFACTS_DIR!\!DIST_NAME!\!EXE_NAME:exe=!" !USER_ARGS!
) else (
    echo ERROR: !EXE_NAME! not found
    echo Expected: !ARTIFACTS_DIR!\!DIST_NAME!\!EXE_NAME! or !ARTIFACTS_DIR!\!DIST_NAME!\!EXE_NAME:exe=!
    exit /b 1
)
echo.
exit /b 0
