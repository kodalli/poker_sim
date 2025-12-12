# Claude Code Custom Commands

Custom commands let you create reusable prompts that can be invoked with a slash command (e.g., `/project:fix-issue`).

## Location

Commands are markdown files stored in:

| Location | Scope | Example |
|----------|-------|---------|
| `.claude/commands/` | Project-specific (shared via git) | `/project:fix-issue` |
| `~/.claude/commands/` | Personal (your machine only) | `/user:optimize` |

## Naming

The filename (minus `.md`) becomes the command name:

- `.claude/commands/fix-issue.md` → `/project:fix-issue`
- `~/.claude/commands/optimize.md` → `/user:optimize`

## Basic Structure

A simple command is just markdown content that becomes the prompt:

```markdown
Analyze the performance of this code and suggest three specific optimizations.
```

## Frontmatter Options

Commands support YAML frontmatter for advanced configuration:

```markdown
---
allowed-tools: Read, Grep, Glob
description: Run security vulnerability scan
model: claude-sonnet-4-5-20250929
argument-hint: [issue-number] [priority]
---

Your prompt here...
```

| Field | Purpose |
|-------|---------|
| `allowed-tools` | Restrict which tools Claude can use |
| `description` | Shows in command list |
| `model` | Override the model for this command |
| `argument-hint` | Document expected arguments |

## Arguments

Use `$ARGUMENTS` for the full argument string, or positional placeholders (`$1`, `$2`, etc.):

```markdown
---
argument-hint: [issue-number]
---

Fix issue #$1. Check the issue description and implement changes.
```

Invoke with: `/project:fix-issue 123`

## Dynamic Content

### File References

Use `@filename` to include file contents inline:

```markdown
---
description: Review configuration files
---

Review the following configuration files for issues:
- Package config: @package.json
- TypeScript config: @tsconfig.json

Check for security issues and misconfigurations.
```

### Shell Output

Use `` !`command` `` to embed command output:

```markdown
## Context
- Current status: !`git status`
- Current diff: !`git diff HEAD`

Create a git commit with an appropriate message.
```

## Full Example

A complete command combining all features:

```markdown
---
allowed-tools: Bash(git add:*), Bash(git status:*), Bash(git commit:*)
description: Create a git commit with auto-generated message
---

## Context

- Current status: !`git status`
- Current diff: !`git diff HEAD`

## Task

Create a git commit with an appropriate message based on the changes.
Follow the commit convention in @CLAUDE.md.
```

## Quick Start

1. Create the commands directory:
   ```bash
   mkdir -p .claude/commands
   ```

2. Create a command file:
   ```bash
   cat > .claude/commands/review.md << 'EOF'
   ---
   description: Review code for issues
   allowed-tools: Read, Grep, Glob
   ---

   Review the current changes for:
   - Security vulnerabilities
   - Performance issues
   - Code style violations

   Current diff: !`git diff`
   EOF
   ```

3. Use the command:
   ```
   /project:review
   ```
