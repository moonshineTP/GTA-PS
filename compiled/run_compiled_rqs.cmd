@echo off
setlocal

:: Capture all user arguments once at the start
set "USER_ARGS=%*"

:: Path to the artifacts directory relative to this script
set "ARTIFACTS_DIR=%~dp0artifacts"

echo ======================================================
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
if exist "%ARTIFACTS_DIR%\%DIST_NAME%\%EXE_NAME%" (
    "%ARTIFACTS_DIR%\%DIST_NAME%\%EXE_NAME%" %USER_ARGS%
) else (
    echo ERROR: %EXE_NAME% not found in %ARTIFACTS_DIR%\%DIST_NAME%
)
echo.
exit /b 0
