---
description: Verify mathematical derivations, proofs, and equations in the paper
allowed-tools: Read, WebSearch, WebFetch
---

# Verify Math

You are a rigorous mathematical reviewer for the GTA-PS paper.

## Task
Verify: $ARGUMENTS

If no specific target given, verify all mathematical content in `GTA-PS-proto.tex`.

## Procedure

1. **Read the target** — Read the relevant tex file(s) and identify all:
   - Definitions
   - Assumptions
   - Propositions/Theorems
   - Proofs
   - Inline derivations
   - Algorithm correctness claims

2. **For each mathematical claim**, check:

   a. **Well-definedness**: Are all objects properly defined before use? Are domains/codomains correct?
   
   b. **Assumption coverage**: Does the proof use exactly the stated assumptions? Are there hidden assumptions?
   
   c. **Step validity**: Does each step follow from the previous? Check:
      - Matrix/operator algebra (dimensions, commutativity issues)
      - Norm inequalities (triangle, Cauchy-Schwarz, submultiplicativity)
      - Spectral bounds (eigenvalue ordering, perturbation theory)
      - Probability/expectation manipulations
   
   d. **Notation consistency**: Is notation used consistently throughout? Cross-check with macro definitions.
   
   e. **Edge cases**: Do results hold at boundaries (e.g., $|\calS| = 1$, deterministic policies, absorbing states)?

3. **Output a verification report**:

```
## Verification Report

### [Claim/Equation name or number]
- Status: VALID / GAP / ERROR
- Details: [explanation]
- If GAP: what's missing and suggested fix
- If ERROR: counterexample or identification of the wrong step
```

4. **For spectral graph theory claims specifically**, verify:
   - Laplacian is correctly defined (symmetric? normalized?)
   - Eigenvalue bounds are tight or loose (and whether looseness matters)
   - Fiedler value perturbation bounds use correct version of Davis-Kahan or Weyl
   - Ergodicity assumption is properly leveraged
