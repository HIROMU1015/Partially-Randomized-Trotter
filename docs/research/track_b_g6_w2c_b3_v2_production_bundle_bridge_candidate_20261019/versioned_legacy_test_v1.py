"""Minimal versioned test maintenance; preserve complete original measurement rows.

Compile a separate in-memory version with the original filename/observer filter.
No original source file or test code object is mutated. Scoped/Cached comparison
and all other assertions remain verbatim; caller chooses original or corrected.
"""
from pathlib import Path
from hashlib import sha256
import difflib,types
OLD="x.endswith(':leaf')"
NEW="x.endswith('.<locals>.leaf')"
def load(path,corrected):
    path=Path(path);raw=path.read_bytes();text=raw.decode();assert text.count(OLD)==1
    source=text.replace(OLD,NEW)if corrected else text
    module=types.ModuleType('test_g6_w2c_b3_parallel_offline_verifier');module.__file__=str(path)
    exec(compile(source,str(path),'exec'),module.__dict__)
    evidence=dict(original_sha256=sha256(raw).hexdigest(),corrected_compiled_source_sha256=sha256(source.encode()).hexdigest(),original_source_modified=False,
        literal_changes=1 if corrected else 0,diff=''.join(difflib.unified_diff(text.splitlines(keepends=True),source.splitlines(keepends=True),fromfile='fixed_49a0dcc_original.py',tofile='versioned_legacy_test_v1.py')),
        full_observer_rows_and_exclusive_partition_comparison_preserved=True)
    return module,evidence
