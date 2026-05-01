# GTA-PS Native Compilation Pipeline

This directory contains the production-ready pipeline for converting Python research scripts into high-performance native Windows executables using Nuitka.

## 🚀 Performance Benefit
**Verified Speedup:** ~2.6x faster than standard Python execution.
*   **Startup:** Near-instant (eliminates heavy NumPy/SciPy import latency).
*   **Execution:** Core planning loops are optimized to machine code.
*   **Scalability:** Recommended for large-scale runs (e.g., 1000+ seeds).

## 🛠 Workflow Architecture

The pipeline is decoupled into **Compilation** (building the binaries) and **Orchestration** (running the groups).

| Group | Compilation Script | Orchestration (Run) Script |
| :--- | :--- | :--- |
| **Sanity** | `compile_sanity.cmd` | `run_compiled_sanity.cmd` |
| **RQs** | `compile_rqs.cmd` | `run_compiled_rqs.cmd` |
| **Ablations** | `compile_ablations.cmd` | `run_compiled_ablations.cmd` |

## 📖 Usage Instructions

### 1. Compilation (One-time per script)
Run the compilation script for the desired group. This will build standalone `.exe` files into the `artifacts/` directory.
```cmd
# Example: Compile all Research Question scripts
code\compiled\compile_rqs.cmd
```
*Note: The first run takes 10-15 minutes as it compiles the math stack. Subsequent scripts are faster due to internal caching (`ccache`).*

### 2. Execution
Use the orchestration scripts to run the compiled binaries with custom parameters.
```cmd
# Example: Run 1000 seeds natively
code\compiled\run_compiled_rqs.cmd --seeds 1000 --workers 5
```

## ⚙️ Build Configuration
*   **Python Version:** Optimized for **Python 3.13+**.
*   **Safety Limit:** All scripts are hard-coded to a **5-worker limit** to prevent system freezes while keeping 100% CPU utilization efficient.
*   **Plugins:** Explicitly enables the `matplotlib` plugin in headless mode for stable plotting.
*   **Backend:** Uses the local Windows C++ compiler (Visual Studio or MinGW).

## 📁 Directory Structure
*   `artifacts/`: Contains the `.dist` folders and final `.exe` binaries.
*   `monitor_build.cmd`: Utility to watch live Nuitka logs and build sizes.
*   `build_all_native.cmd`: Master script to compile every script in the project.
