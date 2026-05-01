---
description: Run autonomous research loop - modify, verify, keep/discard, repeat
allowed-tools: Read, Edit, Write, Bash, WebSearch, WebFetch, Agent
---

# Autoresearch Loop

Autonomous goal-directed iteration for the GTA-PS research project. Inspired by Karpathy's autoresearch.

## Goal
$ARGUMENTS

If no goal specified, read the current state and ask what to optimize.

## Core Loop

For each iteration:

### 1. Review
- Read current state of tex files, recent git history
- Check what's been tried (git log with `experiment:` prefix)
- Identify the most impactful next change

### 2. Pick
Based on accumulated context, select ONE focused modification:
- Tighten a bound in a proof
- Add a missing lemma
- Improve an algorithm description
- Add/refine an experimental protocol
- Strengthen related work positioning
- Fix a mathematical gap

### 3. Modify
Make exactly one atomic change. Keep it reviewable.

### 4. Verify
- If math: verify the derivation step-by-step
- If LaTeX: compile and check for errors
- If prose: check logical flow and citations

### 5. Keep or Discard
- If the change improves the paper: keep it, commit with `experiment: {description}`
- If it introduces errors or doesn't help: revert

### 6. Log
Report: iteration number, what was tried, result (keep/discard), why.

### 7. Repeat
Continue to next iteration. After every 5 iterations, provide a progress summary.

## Principles
- One change per iteration — atomic and reviewable
- Pre-commit protocol: describe what you'll change before changing it
- Mechanical verification where possible (compile, check consistency)
- Negative results are valuable — log what didn't work and why
- When stuck: re-read the paper, look at it from a reviewer's perspective
