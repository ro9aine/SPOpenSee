from __future__ import annotations

import re
import sys
from pathlib import Path


ALLOWED_TYPES = (
    "feat",
    "fix",
    "docs",
    "style",
    "refactor",
    "perf",
    "test",
    "build",
    "ci",
    "chore",
    "revert",
)

COMMIT_RE = re.compile(
    r"^(?P<type>" + "|".join(ALLOWED_TYPES) + r")(\([^)]+\))?!?:\s.+$"
)


def main() -> int:
    if len(sys.argv) != 2:
        print("Expected a single commit message file path.", file=sys.stderr)
        return 2

    message_path = Path(sys.argv[1])
    if not message_path.exists():
        print(f"Commit message file not found: {message_path}", file=sys.stderr)
        return 2

    first_line = message_path.read_text(encoding="utf-8").splitlines()[0].strip()
    if COMMIT_RE.match(first_line):
        return 0

    allowed = ", ".join(ALLOWED_TYPES)
    print(
        "Invalid commit message.\n"
        f"Expected format: <type>: <subject> or <type>(scope): <subject>\n"
        f"Allowed types: {allowed}\n"
        f"Got: {first_line!r}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
