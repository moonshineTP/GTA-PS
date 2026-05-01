---
description: Compile LaTeX documents and report errors/warnings
allowed-tools: Bash, Read
---

# Build LaTeX

Compile the LaTeX document: $ARGUMENTS

If no argument given, compile `GTA-PS-proto.tex`.

## Build Commands

```bash
PDFLATEX="/mnt/c/Program Files (x86)/texlive/2025/bin/windows/pdflatex.exe"
BIBTEX="/mnt/c/Program Files (x86)/texlive/2025/bin/windows/bibtex.exe"
```

## Procedure

1. Run pdflatex (first pass):
   ```
   "$PDFLATEX" -interaction=nonstopmode {file}.tex
   ```

2. Run bibtex (if references changed):
   ```
   "$BIBTEX" {file}
   ```

3. Run pdflatex twice more (resolve references):
   ```
   "$PDFLATEX" -interaction=nonstopmode {file}.tex
   "$PDFLATEX" -interaction=nonstopmode {file}.tex
   ```

4. **Parse the .log file** for:
   - Errors (lines starting with `!`)
   - Warnings (undefined references, overfull hboxes > 5pt)
   - Missing citations

5. **Report**:
   - Build status: SUCCESS / FAILURE
   - Error count and details
   - Warning summary
   - Any missing references or citations
