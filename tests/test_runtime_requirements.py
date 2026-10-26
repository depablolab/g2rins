# (C) 2026 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Keep cleanup helpers within the declared core runtime requirements."""

import ast
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_library_sources_parse_with_python_310_grammar():
    # This checks syntax, not full interpreter/dependency compatibility; the
    # supported-version CI matrix remains necessary for the latter.
    for path in sorted((ROOT / "src/g2rins").glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))


def test_core_import_and_sampling_do_not_require_notebook_or_test_extras():
    code = """
import importlib.abc
import sys

blocked = {'matplotlib', 'IPython', 'pydot', 'pytest', 'jupyter', 'ipykernel'}

class RejectExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ImportError('Optional dependency blocked: ' + fullname)

sys.meta_path.insert(0, RejectExtras())
import g2rins
creator = g2rins.G2rins.make('C').get_graph_creator().get_ensemble_creator()
result = creator.create_ensemble(1, output_format='smiles', ensemble_info=True, seed=7)
assert result.chains == ['C']
transformer = g2rins.get_global_transformer()
assert all(g2rins.get_global_transformer() is transformer for _ in range(5))
assert not blocked.intersection(sys.modules)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr