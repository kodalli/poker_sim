---
allowed-tools: Bash(git add:*), Bash(git status:*), Bash(git commit:*), Bash(git diff:*), Bash(git log:*), Read, Edit, Glob, Grep, AskUserQuestion
description: Create a git commit following project conventions
---

## Context

Current status:
!`git status`

Recent commits (for style reference):
!`git log --oneline -5`

Staged changes:
!`git diff --cached`

Unstaged changes:
!`git diff`

## Commit Convention

Follow the format from @CLAUDE.md:

**Format:** `<type>(<scope>): <description>`

Valid types: `feat`, `fix`, `perf`, `revert`, `chore`, `docs`, `style`, `refactor`, `test`, `ci`

## Rules

1. Do NOT include Claude Code attribution or Co-Authored-By trailers
2. Use `-m "title" -m "body"` format for multi-line messages
3. Stage relevant files first if needed
4. Keep the description concise (under 72 chars)
5. Body should explain the "why", not the "what"
6. NEVER use `--no-verify` unless the user explicitly grants permission

## Pre-commit Hook Failure Handling

If the commit fails due to pre-commit hooks:

1. **Formatting/whitespace fixes** (ruff, biome, trailing whitespace, EOF):
   - Stage the auto-fixed files and retry

2. **Type errors (mypy)**:
   - Read the error output carefully
   - Fix the type issues in the source files
   - Stage fixes and retry

3. **Lint errors (ruff check, eslint)**:
   - Fix the reported issues in the source files
   - Stage fixes and retry

4. **Other failures**:
   - Analyze the error output
   - Fix the underlying issue
   - Stage fixes and retry

**Retry limit**: If commits continue to fail after 3 attempts, ask the user whether to:
- Continue trying to fix issues
- Skip hooks with `--no-verify` (requires explicit user permission)
- Abort the commit

## Task

Create a commit for the current changes. If changes aren't staged, stage the appropriate files first. Automatically fix any pre-commit hook failures and retry until successful.
