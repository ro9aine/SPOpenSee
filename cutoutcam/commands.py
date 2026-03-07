from __future__ import annotations

import subprocess


def install_pre_commit() -> int:
    result = subprocess.run(["pre-commit", "install"], check=False)
    return int(result.returncode)
