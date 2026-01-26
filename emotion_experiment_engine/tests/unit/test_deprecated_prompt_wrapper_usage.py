"""Test file responsible for: ensuring deprecated prompt wrapper helper is unused.
Purpose: guard against regressions that reintroduce the legacy memory helper across
tests and documentation.
"""

import unittest
from pathlib import Path


TARGET_IDENTIFIER = "get_memory_" + "prompt_wrapper"


class TestDeprecatedPromptWrapperUsage(unittest.TestCase):
    """Fail when deprecated helper still appears in tests/docs."""

    def test_deprecated_helper_not_referenced(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        targets = [
            repo_root / "emotion_experiment_engine" / "tests",
            repo_root / "emotion_experiment_engine" / "README.md",
            repo_root / "emotion_experiment_engine" / "claude_doc",
            repo_root / "docs",
        ]

        found: list[str] = []

        for target in targets:
            if target.is_dir():
                for path in target.rglob("*"):
                    if not path.is_file():
                        continue
                    if path.suffix not in {".py", ".md", ".rst", ".feature", ""}:
                        continue
                    content = path.read_text(encoding="utf-8", errors="ignore")
                    if TARGET_IDENTIFIER in content:
                        found.append(str(path.relative_to(repo_root)))
            elif target.is_file():
                content = target.read_text(encoding="utf-8", errors="ignore")
                if TARGET_IDENTIFIER in content:
                    found.append(str(target.relative_to(repo_root)))

        self.assertFalse(
            found,
            msg=(
                "Deprecated helper `" + TARGET_IDENTIFIER + "` still referenced in: "
                + ", ".join(found)
            ),
        )


if __name__ == "__main__":
    unittest.main()
