#!/usr/bin/env python3
"""
Simple layout check for tests to avoid duplicates.

Rules enforced:
- No duplicate test filenames between tests root and subfolders (unit/, integration/, e2e/, regression/).

Exit non-zero on violation to fail CI/pre-commit.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    tests_root = Path(__file__).resolve().parents[1]

    root_files = {p.name for p in tests_root.glob("test_*.py")}
    subdirs = ["unit", "integration", "e2e", "regression"]

    duplicates: list[tuple[str, str]] = []
    for sub in subdirs:
        subdir = tests_root / sub
        if not subdir.exists():
            continue
        for p in subdir.glob("test_*.py"):
            if p.name in root_files:
                duplicates.append((p.name, sub))

    if duplicates:
        print("Duplicate test filenames detected between root and subfolders:")
        for name, sub in duplicates:
            print(f" - {name} present in tests/ and tests/{sub}/")
        print("\nAction: remove the root-level duplicate and keep the categorized copy.")
        return 1

    print("Test layout check passed: no duplicates detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

