"""Identity of this new scope and every frozen certification dependency."""

from hashlib import sha256
from pathlib import Path

BASE_COMMIT = "34d9a2f1a25bd35c03566ce0635e8b66665ad996"


def source_identity():
    root = Path(__file__).parent.parent
    repository = root.parents[1]
    paths = [p for folder in ("a2_pilot", "a2_spec01_03", "a2_spec04_07")
             for p in sorted((root / folder).glob("*.py"))]
    paths.append(repository / "scripts/run_qc_a2_fixed_df_pilot.py")
    paths.append(root / "__init__.py")
    return {"version": "A2_FIXED_DF_PILOT_SOURCE_V1", "frozen_dependency_commit": BASE_COMMIT,
            "modules": {str(p.relative_to(repository)): sha256(p.read_bytes()).hexdigest() for p in paths}}
