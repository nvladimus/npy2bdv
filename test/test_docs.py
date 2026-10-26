# Runs every Python code block of the docs (docs/source/*.rst), so examples stay correct.
import os
import re
import tempfile
import unittest
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs" / "source"


def python_blocks(rst_text):
    """Code of each `.. code-block:: python` directive, dedented."""
    blocks = []
    for match in re.finditer(r"^( *)\.\. code-block:: python\n((?:\1 +\S.*\n|\s*\n)+)", rst_text, re.M):
        lines = match.group(2).splitlines()
        indent = min(len(l) - len(l.lstrip()) for l in lines if l.strip())
        blocks.append("\n".join(l[indent:] for l in lines))
    return blocks


class TestDocSnippets(unittest.TestCase):
    def test_snippets_run(self):
        pages = sorted(DOCS_DIR.glob("**/*.rst"))
        self.assertTrue(pages, f"No docs found in {DOCS_DIR}")
        cwd = os.getcwd()
        for page in pages:
            blocks = python_blocks(page.read_text(encoding="utf-8"))
            if not blocks:
                continue
            with self.subTest(page=page.name), tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                try:
                    namespace = {}  # blocks of one page run in sequence, sharing variables
                    for i, code in enumerate(blocks):
                        exec(compile(code, f"{page.name} block {i}", "exec"), namespace)
                finally:
                    os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
