"""Normal import/export identity checks; no exported function is executed."""
import ast
import importlib
from pathlib import Path

import trotterlib


def original_exports():
    tree = ast.parse(Path(trotterlib.__file__).read_text())
    return [(node, alias) for node in tree.body if isinstance(node, ast.ImportFrom)
            for alias in node.names]


def test_all_413_eager_exports_match_original_module_attributes():
    exports = original_exports()
    assert len(exports) == 413
    for node, alias in exports:
        module = importlib.import_module("." * node.level + node.module, "trotterlib")
        assert getattr(trotterlib, alias.asname or alias.name) is getattr(module, alias.name)


def test_eager_star_import_and_reload_preserve_existing_exports():
    names = [alias.asname or alias.name for node, alias in original_exports()]
    before = {name: getattr(trotterlib, name) for name in names}
    namespace = {}
    exec("from trotterlib import *", namespace)
    assert all(namespace[name] is value for name, value in before.items())
    assert importlib.reload(trotterlib) is trotterlib
    assert all(getattr(trotterlib, name) is value for name, value in before.items())
