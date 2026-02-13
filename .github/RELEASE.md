# Release Process

## Creating a Release

1. Update `CHANGELOG.md`:
   - Rename `[Unreleased]` to `[X.Y.Z] - YYYY-MM-DD`
   - Add new empty `[Unreleased]` section with subheaders
   - Update comparison links at the bottom

2. Bump version and push:
```bash
python scripts/bump_version.py patch  # or: minor, major
git push && git push --tags
```

The script automatically stages `CHANGELOG.md` along with `pyproject.toml`.

Automation handles the rest: https://github.com/parameterlab/maseval/actions

---

## Troubleshooting

**Wrong version tagged?**

```bash
git tag -d v0.1.2
git push origin :refs/tags/v0.1.2
# Then fix pyproject.toml and retry
```

**Automation failed?**

```bash
uv build && uv publish
```

## Version Guide

- `patch`: Bug fixes (0.1.1 → 0.1.2)
- `minor`: New features (0.1.1 → 0.2.0)
- `major`: Breaking changes (0.1.1 → 1.0.0)
