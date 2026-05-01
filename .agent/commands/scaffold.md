---
description: Scaffold a new section, proof, algorithm, or experiment design for the paper
allowed-tools: Read, Edit, Write, Bash, WebSearch
---

# Scaffold

You are a research scaffolding agent for the GTA-PS paper.

## Task
Scaffold: $ARGUMENTS

Examples:
- `proof of Proposition 3` — generate proof skeleton with key steps
- `section on experimental design` — draft section structure with subsections
- `algorithm for topology-aware update` — write pseudocode in algorithm environment
- `convergence analysis` — outline theorem statement and proof strategy
- `related work on spectral RL` — draft related work paragraphs

## Procedure

1. **Read existing content** — Read `GTA-PS-proto.tex` and `GTA-PS-memo.tex` to understand current state, notation, and what's already written.

2. **Scaffold based on type**:

### For Proofs:
```latex
\begin{proof}
% Step 1: [description of what to show]
% Key tool: [which lemma/inequality/technique]

% Step 2: ...

% Step 3: ...

% Conclusion: combining Steps 1-3 yields the result.
\end{proof}
```
- Fill in as much rigorous math as possible
- Mark gaps explicitly with `% TODO: ...`
- Use existing macros (`\calS`, `\bbE`, `\norm{}`, `\Pt`, `\Lt`, etc.)

### For Sections:
- Write full LaTeX with `\section`, `\subsection`
- Include topic sentences for each paragraph
- Add `\citep{}` placeholders for needed references
- Write transitions between subsections

### For Algorithms:
```latex
\begin{algorithm}
\caption{...}
\begin{algorithmic}[1]
\Require ...
\Ensure ...
\State ...
\end{algorithmic}
\end{algorithm}
```

### For Experiment Designs:
- Environments/domains to test
- Baselines to compare against
- Metrics and evaluation protocol
- Expected outcomes and what they would demonstrate

3. **Output**: Present the scaffolded LaTeX, and ask if user wants it inserted into the tex file.

## Notation Reference
- State space: `\calS`, Action space: `\calA`
- Expectation: `\bbE`, Reals: `\bbR`
- Policy-induced kernel: `\Pt` or `P_t`
- Normalized Laplacian: `\Lt` or `L_t`
- Norms: `\norm{x}`, Absolute: `\abs{x}`
- Priority: `\prio`
