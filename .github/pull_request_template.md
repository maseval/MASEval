## Description

<!-- Briefly describe what this PR changes and why -->

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Code quality improvement (refactoring, formatting, etc.)

## Checklist

### Contribution

- [ ] I have read the [CONTRIBUTING.md](CONTRIBUTING.md) guide.
- [ ] Commits follow "[How to write a good git commit message](http://chris.beams.io/posts/git-commit/)"

### Documentation

- [ ] Added/updated docstrings for new/modified functions as instructed [CONTRIBUTING.md](CONTRIBUTING.md)
- [ ] Updated relevant documentation in `docs/` (if applicable)
- [ ] Tag github issue with this PR (if applicable)

### Changelog

- [ ] Added entry to `CHANGELOG.md` under `[Unreleased]` section
  - Use **Added** section for new features
  - Use **Changed** section for modifications to existing functionality
  - Use **Fixed** section for bug fixes
  - Use **Removed** section for deprecated/removed features
- [ ] OR this is a documentation-only change (no changelog needed)

Example:
`- Support for multi-agent tracing (PR:#123)`

### Architecture (if applicable)

- [ ] Core/Interface separation: Changes in `maseval/core/` do NOT import from `maseval/interface/`
- [ ] Dependencies: New core dependencies added sparingly; framework integrations go to optional dependencies

## Additional Notes

<!-- Any additional context, screenshots, or information reviewers should know -->
