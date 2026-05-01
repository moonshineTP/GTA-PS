---
description: Read and summarize a PDF paper, extracting key insights relevant to GTA-PS
allowed-tools: Read, WebFetch, WebSearch, Write, Bash
---

# Read Paper

You are a research paper reader for the GTA-PS project.

## Task
Read and analyze the paper: $ARGUMENTS

The argument can be:
- A local file path to a .pdf
- An arxiv URL (e.g., https://arxiv.org/abs/2301.xxxxx)
- A paper title to search for

## Procedure

### If given a local PDF path:
1. Use the Read tool to read the PDF file (use `pages` parameter for large PDFs — read in chunks of 15 pages)
2. Extract and summarize

### If given an arxiv URL:
1. Fetch the abstract page via WebFetch
2. Extract title, authors, abstract
3. If the user wants the full paper, fetch the PDF URL

### If given a paper title:
1. WebSearch for `{title} arxiv OR semanticscholar`
2. Fetch the found page

## Output Structure

### Quick Summary (5 lines max)
- **Title**: 
- **Authors**: 
- **Year/Venue**: 
- **Core contribution**: 
- **Method**: 

### Detailed Analysis
1. **Problem addressed**: What gap does this paper fill?
2. **Key idea**: The main technical insight in 3-4 sentences
3. **Method details**: Algorithms, architectures, or theoretical results
4. **Experimental setup**: Domains, baselines, metrics
5. **Main results**: What did they show?
6. **Limitations**: What doesn't work or isn't addressed?

### Relevance to GTA-PS
- How does this relate to our topology-augmented prioritized sweeping approach?
- What can we cite, build on, or differentiate from?
- Suggested BibTeX entry (verified, not hallucinated)

### Key Equations/Definitions
Extract the 3-5 most important equations or definitions, rendered in LaTeX notation using our macros where applicable (`\calS`, `\calA`, `\bbE`, `\norm{}`, etc.)
