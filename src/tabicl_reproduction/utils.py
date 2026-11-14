import subprocess
from pathlib import Path


def git_commit() -> str:
    """Short hash of the checked-out commit, or "unknown" without git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"  # e.g. in a container without git
