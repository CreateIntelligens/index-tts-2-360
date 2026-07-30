#!/usr/bin/env python3
"""
Report every `from transformers... import X` in the vendored GPT files that the
installed transformers no longer provides, and whether the repo actually uses it.

The files under indextts/gpt/ are snapshots of transformers internals, so they
drift as transformers moves. Running this against a target environment tells us
up front which symbols need a fallback, instead of discovering them one crash at
a time.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGETS = [
    REPO / "indextts/gpt/transformers_generation_utils.py",
    REPO / "indextts/gpt/transformers_gpt2.py",
    REPO / "indextts/gpt/transformers_modeling_utils.py",
    REPO / "indextts/gpt/transformers_beam_search.py",
]


def main() -> int:
    import transformers

    print(f"transformers {transformers.__version__}  (python {sys.version.split()[0]})")
    missing_total = 0

    for path in TARGETS:
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        source = path.read_text(encoding="utf-8")
        misses = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not node.module.startswith("transformers"):
                continue
            try:
                module = importlib.import_module(node.module)
            except Exception as exc:  # module itself gone
                misses.append((node.module, "<entire module>", str(exc), 0))
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                if not hasattr(module, alias.name):
                    # Count real uses (occurrences beyond the import line itself).
                    uses = source.count(alias.name) - 1
                    misses.append((node.module, alias.name, "missing", uses))

        if misses:
            print(f"\n--- {path.relative_to(REPO)} ---")
            for module_name, symbol, note, uses in misses:
                print(f"  {module_name}.{symbol}  ({note}, {uses} use(s) in file)")
            missing_total += len(misses)

    print(f"\ntotal missing symbols: {missing_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
