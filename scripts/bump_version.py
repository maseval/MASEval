#!/usr/bin/env python3
"""
Helper script to bump version and create git tag consistently.

Usage:
    python scripts/bump_version.py patch  # 0.1.1 -> 0.1.2
    python scripts/bump_version.py minor  # 0.1.1 -> 0.2.0
    python scripts/bump_version.py major  # 0.1.1 -> 1.0.0
"""

import argparse
import re
import subprocess
from pathlib import Path


def get_current_version(pyproject_path: Path) -> str:
    """Extract version from pyproject.toml."""
    content = pyproject_path.read_text()
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")
    return match.group(1)


def bump_version(version: str, part: str) -> str:
    """Bump version according to semver."""
    major, minor, patch = map(int, version.split("."))

    if part == "major":
        return f"{major + 1}.0.0"
    elif part == "minor":
        return f"{major}.{minor + 1}.0"
    elif part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid version part: {part}")


def update_pyproject(pyproject_path: Path, old_version: str, new_version: str):
    """Update version in pyproject.toml."""
    content = pyproject_path.read_text()
    new_content = re.sub(rf'^version\s*=\s*["\']({re.escape(old_version)})["\']', f'version = "{new_version}"', content, flags=re.MULTILINE)
    pyproject_path.write_text(new_content)


def main():
    parser = argparse.ArgumentParser(description="Bump version and create git tag")
    parser.add_argument("part", choices=["major", "minor", "patch"], help="Version part to bump")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    # Get paths
    repo_root = Path(__file__).parent.parent
    pyproject_path = repo_root / "pyproject.toml"

    # Get current version and calculate new version
    current_version = get_current_version(pyproject_path)
    new_version = bump_version(current_version, args.part)

    print(f"Current version: {current_version}")
    print(f"New version: {new_version}")

    if args.dry_run:
        print("\nDry run - no changes made")
        return

    # Update pyproject.toml
    print(f"\nUpdating {pyproject_path}...")
    update_pyproject(pyproject_path, current_version, new_version)

    # Commit and tag
    print("\nCreating git commit and tag...")
    changelog_path = repo_root / "CHANGELOG.md"
    subprocess.run(["git", "add", str(pyproject_path), str(changelog_path)], check=True)
    subprocess.run(["git", "commit", "-m", f"Bump version to {new_version}"], check=True)
    subprocess.run(["git", "tag", "-s", f"v{new_version}", "-m", f"Release v{new_version}"], check=True)

    print(f"\nDone! Version bumped to {new_version}")
    print("\nNext steps:")
    print("  1. Review the changes: git show")
    print("  2. Push to GitHub: git push && git push --tags")
    print("  3. GitHub Actions will handle PyPI release automatically")


if __name__ == "__main__":
    main()
