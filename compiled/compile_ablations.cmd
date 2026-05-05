@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "CODE_ROOT=%SCRIPT_DIR%.."
set "ARTIFACTS_DIR=%SCRIPT_DIR%artifacts"
set "PYTHON=%CODE_ROOT%\.venv\Scripts\python.exe"
set "PYTHONPATH=%CODE_ROOT%\src;%CODE_ROOT%\threads;%CODE_ROOT%\misc"

echo Starting Ablations Compilation (Windows)...
echo.

set "ABLS=misc\mixing_threshold_ablation.py misc\shift_rate_ablation.py misc\boltzmann_b_ablation.py misc\gamma_ablation.py misc\periodicity_ablation.py misc\reward_support_ablation.py misc\cumulative_ablation.py"

for %%f in (%ABLS%) do (
    echo --- Compiling %%f ---
    "%PYTHON%" -m nuitka --standalone --output-dir="%ARTIFACTS_DIR%" --assume-yes-for-downloads --jobs=5 --include-package=gta_ps "%CODE_ROOT%\%%f"
)

echo.
echo All Ablations compiled to %ARTIFACTS_DIR%
endlocal
