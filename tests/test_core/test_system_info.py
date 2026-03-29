"""Tests for maseval.core.utils.system_info module."""

from unittest.mock import patch
import os

import pytest

from maseval.core.utils.system_info import (
    gather_benchmark_config,
    get_environment_variables,
    get_git_info,
    get_package_versions,
    get_python_info,
    get_system_info,
)

pytestmark = pytest.mark.core


class TestGetGitInfo:
    def test_returns_commit_hash(self):
        info = get_git_info()
        # We're running inside a git repo, so this should succeed
        assert "commit_hash" in info
        assert "branch" in info

    def test_error_path_for_invalid_repo(self, tmp_path):
        info = get_git_info(repo_path=str(tmp_path / "nonexistent"))
        assert "error" in info
        assert "error_type" in info


class TestGetPythonInfo:
    def test_contains_expected_fields(self):
        info = get_python_info()
        assert "version" in info
        assert "executable" in info
        assert "implementation" in info
        assert "version_info" in info
        assert info["version_info"]["major"] >= 3


class TestGetSystemInfo:
    def test_contains_expected_fields(self):
        info = get_system_info()
        assert "hostname" in info
        assert "platform" in info
        assert "system" in info
        assert "machine" in info


class TestGetPackageVersions:
    def test_returns_dict(self):
        # This actually runs pip freeze, so just verify it returns a dict
        versions = get_package_versions()
        assert isinstance(versions, dict)

    def test_handles_subprocess_error(self):
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "pip")):
            versions = get_package_versions()
            assert versions == {}


class TestGetEnvironmentVariables:
    def test_excludes_sensitive_keys(self):
        with patch.dict(os.environ, {"MY_API_KEY": "secret", "CUDA_VISIBLE_DEVICES": "0"}, clear=False):
            env = get_environment_variables()
            assert "MY_API_KEY" not in env
            assert "CUDA_VISIBLE_DEVICES" in env

    def test_excludes_token_and_password(self):
        with patch.dict(os.environ, {"OPENAI_API_TOKEN": "tok", "DB_PASSWORD": "pw", "OPENAI_ORG": "org"}, clear=False):
            env = get_environment_variables()
            assert "OPENAI_API_TOKEN" not in env
            assert "DB_PASSWORD" not in env

    def test_custom_patterns(self):
        with patch.dict(os.environ, {"MY_CUSTOM_VAR": "val", "OTHER_VAR": "other"}, clear=False):
            env = get_environment_variables(include_patterns=["MY_CUSTOM"])
            assert "MY_CUSTOM_VAR" in env
            assert "OTHER_VAR" not in env


class TestGatherBenchmarkConfig:
    def test_includes_all_sections(self):
        config = gather_benchmark_config()
        assert "timestamp" in config
        assert "git" in config
        assert "python" in config
        assert "system" in config
        assert "packages" in config
        assert "environment" in config

    def test_excludes_packages(self):
        config = gather_benchmark_config(include_packages=False)
        assert "packages" not in config
        assert "python" in config

    def test_excludes_env_vars(self):
        config = gather_benchmark_config(include_env_vars=False)
        assert "environment" not in config
        assert "python" in config
