# Contributing

## Commit Message Standard

This repository uses Conventional Commits to keep history readable on GitHub.

Format:

`<type>(optional-scope): <description>`

Examples:

- `feat(simulation): add adaptive timestep`
- `fix(renderer): clamp zoom to avoid jitter`
- `docs(readme): add control overview`

Allowed commit types:

- `feat`
- `fix`
- `docs`
- `style`
- `refactor`
- `perf`
- `test`
- `build`
- `ci`
- `chore`
- `revert`

Rules:

- Keep the commit title at 72 characters or less.
- Use imperative mood in the description ("add", "fix", "remove").
- Explain why in the body when needed.
- Reference issues in the footer, for example `Closes #42`.

## Hook Setup

Run this once after cloning to enforce commit format locally:

```powershell
./scripts/setup-git-hooks.ps1
```

The hook validates the first line of each commit message.