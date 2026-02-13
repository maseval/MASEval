"""Utilities for gathering system and environment configuration information."""

import os
import sys
import platform
import socket
from typing import Dict, Any, Optional
from datetime import datetime


def get_git_info(repo_path: Optional[str] = None) -> Dict[str, Any]:
    """Get git repository information for the current working directory or specified path.

    Args:
        repo_path: Optional path to the git repository. If None, uses current working directory.

    Returns:
        Dictionary containing git information:
        - commit_hash: Current commit SHA
        - branch: Current branch name
        - is_dirty: Whether there are uncommitted changes
        - remote_url: Remote origin URL (if available)
        - commit_message: Last commit message
        - commit_author: Last commit author
        - commit_date: Last commit date
        - error: Error message if git info cannot be retrieved
    """
    try:
        import git

        if repo_path is None:
            repo_path = os.getcwd()

        repo = git.Repo(repo_path, search_parent_directories=True)

        # Get current commit
        commit = repo.head.commit

        # Get remote URL (if available)
        remote_url = None
        try:
            remote_url = repo.remotes.origin.url if repo.remotes else None
        except Exception:
            pass

        return {
            "commit_hash": commit.hexsha,
            "commit_hash_short": commit.hexsha[:7],
            "branch": repo.active_branch.name,
            "is_dirty": repo.is_dirty(),
            "untracked_files": len(repo.untracked_files),
            "remote_url": remote_url,
            "commit_message": commit.message.strip(),
            "commit_author": str(commit.author),
            "commit_date": commit.committed_datetime.isoformat(),
            "repo_root": str(repo.working_dir),
        }
    except Exception as e:
        return {
            "error": f"Failed to get git info: {str(e)}",
            "error_type": type(e).__name__,
        }


def get_python_info() -> Dict[str, Any]:
    """Get Python interpreter information.

    Returns:
        Dictionary containing:
        - version: Python version string
        - version_info: Python version tuple
        - executable: Path to Python executable
        - implementation: Python implementation (CPython, PyPy, etc.)
        - compiler: Compiler used to build Python
    """
    return {
        "version": sys.version.split()[0],
        "version_full": sys.version,
        "version_info": {
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro,
            "releaselevel": sys.version_info.releaselevel,
            "serial": sys.version_info.serial,
        },
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "compiler": platform.python_compiler(),
    }


def get_system_info() -> Dict[str, Any]:
    """Get system and OS information.

    Returns:
        Dictionary containing:
        - hostname: Machine hostname
        - platform: Platform name
        - os: Operating system
        - os_version: OS version
        - architecture: Machine architecture
        - processor: Processor name
        - cpu_count: Number of CPUs
    """
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "architecture": platform.architecture()[0],
    }


def get_package_versions() -> Dict[str, str]:
    """Get versions of all installed packages using pip freeze.

    Returns:
        Dictionary mapping package names to version strings.
    """
    import subprocess

    packages = {}

    try:
        # Use pip freeze to get all installed packages
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse pip freeze output (format: "package==version" or "package @ path")
        for line in result.stdout.strip().split("\n"):
            if not line or line.startswith("#"):
                continue

            # Handle different pip freeze formats
            if "==" in line:
                package, version = line.split("==", 1)
                packages[package] = version
            elif " @ " in line:
                # Handle editable installs (e.g., "maseval @ file:///...")
                package = line.split(" @ ", 1)[0]
                packages[package] = "editable"

    except subprocess.CalledProcessError:
        # If pip freeze fails, return empty dict
        pass
    except Exception:
        # Handle any other unexpected errors
        pass

    return packages


def get_environment_variables(include_patterns: Optional[list[str]] = None) -> Dict[str, str]:
    """Get relevant environment variables, excluding sensitive values.

    Args:
        include_patterns: Optional list of substring patterns to match against environment variable names (case-insensitive).
            If None, defaults to common AI/ML framework variables including:
            - AI Services: OPENAI, ANTHROPIC, GOOGLE, GEMINI, HUGGINGFACE, HF_
            - Experiment Tracking: WANDB, LANGFUSE
            - ML Frameworks: CUDA, PYTORCH, TF_ (TensorFlow)
            - Python Environment: PATH, PYTHONPATH, VIRTUAL_ENV, CONDA

            Any variable containing these patterns will be included, except those containing sensitive keywords
            (KEY, TOKEN, SECRET, PASSWORD, CREDENTIAL, AUTH) which are always excluded for security.

    Returns:
        Dictionary of environment variable names and values. API keys, tokens, secrets, and passwords are completely excluded.
    """
    if include_patterns is None:
        include_patterns = [
            "OPENAI",
            "ANTHROPIC",
            "GOOGLE",
            "GEMINI",
            "HUGGINGFACE",
            "HF_",
            "WANDB",
            "LANGFUSE",
            "CUDA",
            "PYTORCH",
            "TF_",
            "PATH",
            "PYTHONPATH",
            "VIRTUAL_ENV",
            "CONDA",
        ]

    # Keywords that indicate sensitive data - these are completely skipped
    sensitive_keywords = ["KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH"]

    env_vars = {}
    for key, value in os.environ.items():
        # Skip any environment variable that looks sensitive
        if any(sensitive in key.upper() for sensitive in sensitive_keywords):
            continue

        # Check if key matches any pattern
        if any(pattern in key.upper() for pattern in include_patterns):
            env_vars[key] = value

    return env_vars


def gather_benchmark_config(
    repo_path: Optional[str] = None,
    include_env_vars: bool = True,
    include_packages: bool = True,
) -> Dict[str, Any]:
    """Gather comprehensive benchmark-level configuration.

    Args:
        repo_path: Optional path to git repository. If None, uses current working directory.
        include_env_vars: Whether to include environment variables.
        include_packages: Whether to include package versions.

    Returns:
        Dictionary containing all benchmark-level configuration:
        - timestamp: When config was gathered
        - git: Git repository information
        - python: Python interpreter information
        - system: System and OS information
        - packages: Installed package versions (if include_packages=True)
        - environment: Relevant environment variables (if include_env_vars=True)
    """
    config = {
        "timestamp": datetime.now().isoformat(),
        "git": get_git_info(repo_path),
        "python": get_python_info(),
        "system": get_system_info(),
    }

    if include_packages:
        config["packages"] = get_package_versions()

    if include_env_vars:
        config["environment"] = get_environment_variables()

    return config
