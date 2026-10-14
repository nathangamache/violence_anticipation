"""Verify that edits to Python files only touched comments and docstrings.

Compares each file's AST (with all bare string statements removed) against a git revision.
Usage: python tools/check_comment_only_diff.py [--rev HEAD] file.py [file.py ...]
"""

import argparse
import ast
import subprocess
import sys


class _StripStrings(ast.NodeTransformer):
    def generic_visit(self, node):
        super().generic_visit(node)
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if isinstance(stmts, list) and stmts and isinstance(stmts[0], ast.stmt):
                kept = [s for s in stmts if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                                                 and isinstance(s.value.value, str))]
                if field == "body" and not kept:
                    kept = [ast.Pass()]
                setattr(node, field, [s for s in kept if not isinstance(s, ast.Pass)] or kept[:1])
        return node


def _normalized(src):
    return ast.dump(_StripStrings().visit(ast.parse(src)), include_attributes=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    bad = 0
    for f in a.files:
        old = subprocess.run(["git", "show", f"{a.rev}:{f}"], capture_output=True, text=True)
        if old.returncode != 0:
            print(f"SKIP {f}: not in {a.rev}")
            continue
        try:
            same = _normalized(old.stdout) == _normalized(open(f, encoding="utf-8").read())
        except SyntaxError as e:
            print(f"FAIL {f}: syntax error {e}")
            bad += 1
            continue
        print(f"{'OK  ' if same else 'FAIL'} {f}")
        bad += not same
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
