# .github/

GitHub Actions CI + repo metadata.

## Workflows

| File | Triggers | What it does |
|---|---|---|
| `workflows/ci.yml` | push to main, PRs to main | uv sync, ruff lint, pytest |

## Conventions

- All CI runs on Ubuntu latest with Python 3.12.
- CI must not require real NASA data — tests that touch it are skip-when-absent.
- The workflow installs `uv` via `astral-sh/setup-uv@v3`.
- `uv sync --frozen --all-extras` is the canonical install line.
