"""Unit tests for the CLI entry point."""

from pathlib import Path
import subprocess
import sys


def test_cli_help():
    """Verify dermaguard CLI outputs help information without error."""
    repo_root = Path(__file__).resolve().parent.parent
    res = subprocess.run(
        [sys.executable, "-m", "src.cli", "--help"],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert res.returncode == 0
    assert "assess-quality" in res.stdout
    assert "triage" in res.stdout
    assert "train" in res.stdout
    assert "validate" in res.stdout
