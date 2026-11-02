"""Passive SOURCE/environment check for the partial-domain ZSXX design."""
import importlib.metadata
from pathlib import Path

from . import c1_stage1_execution as io

SCOPE='C_PR_AWARE_DF_ZSXX_CAPTURED_ENCLOSURE_READINESS_V1'
FAMILY='artifacts/research_c_pr_aware_df_zsxx/2026-10-27'


def preflight(root,path):
    from .c_pr_aware_df_zsxx_enclosure import runtime_model,PRECISION,VERSION
    from .c_pr_aware_df_c2a import P0
    root=Path(root).resolve();c=io.read(path)
    if (c['scope']!=SCOPE or c['implementation']!=VERSION or c['executable'] is not False
            or c['status']!='BLOCKED_PARTIAL_VALIDATED_DOMAINS'
            or c['maximum_logical_units'] is not None or c['precision_bits']!=PRECISION or c['compiler']!=P0
            or c['retry'] or c['resume']):raise ValueError('ZSXX_TECHNICAL_SCOPE')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for p,h in group.items():
            target=(root/p).resolve()
            if not target.is_relative_to(root) or io.sha(target)!=h:raise ValueError('SOURCE_OR_INPUT:'+p)
    for p,v in c['packages'].items():
        if importlib.metadata.version(p)!=v:raise ValueError('ENVIRONMENT:'+p)
    from importlib.util import find_spec
    qroot=Path(find_spec('qiskit').origin).parent
    for p,h in c['external_python_source_hashes'].items():
        target=(qroot/p).resolve()
        if not target.is_relative_to(qroot.resolve()) or io.sha(target)!=h:raise ValueError('COMPILER_SOURCE:'+p)
    model=runtime_model()
    return dict(status='PASSIVE_IDENTITIES_PASS_SCIENCE_BLOCKED',scientific_calls=0,
                executable=False,maximum_logical_units=None,runtime_model=model)
