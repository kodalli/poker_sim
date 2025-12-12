---
description: Log model training experiment results
allowed-tools: Read, Edit, Glob, Grep
argument-hint: [version-name]
---

## Task
Add a new experiment entry to the research log for model version **$1**.

## Current Log
@docs/experiment-log.md

## Entry Format
Use this structure for the new entry:

### $1 - [TODAY'S DATE in YYYY-MM-DD format]

**Summary**: [Brief description of changes/hypothesis tested]

**Results**:
| Opponent     | Win% | BB/hand |
|--------------|------|---------|
| random       |      |         |
| call_station |      |         |
| tag          |      |         |
| lag          |      |         |
| rock         |      |         |
| trapper      |      |         |

**Analysis**: [What worked, what didn't, why]

**Lessons Learned**: [Key takeaways for future experiments]

## Instructions
1. Ask user for the experiment details (results, analysis)
2. Append the new entry to docs/experiment-log.md
3. Keep entries in reverse chronological order (newest first)
