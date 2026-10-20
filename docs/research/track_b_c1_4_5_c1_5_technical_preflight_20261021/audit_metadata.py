"""Bounded static byte/provenance audit; no scientific module import or array decoding."""
import hashlib
import json
from pathlib import Path
import subprocess
from support import HERE, ROOT, FIXED, BASE, source_rows, save

def gitbytes(ref, path):
    return subprocess.check_output(['git','show',ref+':'+path],cwd=ROOT)

old = 'docs/research/track_b_c1_3_r2_reference_recovery_c1_4_20261021/'
counts = {}
for name in ('source_manifest.json','input_manifest.json'):
    manifest = json.loads(gitbytes(BASE,old+name))
    for row in manifest['files']:
        raw = gitbytes(BASE,row['path'])
        assert hashlib.sha256(raw).hexdigest() == row['sha256'], row['path']
    counts[name] = dict(files=len(manifest['files']), fixed_git_blob_SHA256_equal=True,
        scope='Git blob bytes only; no arrays decoded or saved science recomputed')

private = json.loads(gitbytes(BASE,old+'preservation_baseline.json'))['private_originals']
for row in private:
    raw=Path(row['path']).read_bytes()
    assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']

required = ['src/trottertracks/algorithm_codesign/'+p for p in
    ('return_aggregation.py','g7_generator.py','g7_provider.py','g8_bounds.py','g8_pipeline.py',
     'g9_p5.py','g6_f3_generator.py','rte_development/accounting.py','rte_development/pipeline.py')]
required += ['src/trotterlib/'+p for p in
    ('df_rte_tail.py','df_rte_qiskit.py','df_partial_s2.py','rpe_hadamard_interrogation.py')]
rows=[]
for path in required:
    raw=gitbytes(FIXED,path)
    assert raw==(ROOT/path).read_bytes(),path
    rows.append(dict(path=path,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),commit=FIXED))

ns=str(HERE.relative_to(ROOT))+'/'
changed=subprocess.check_output(['git','diff',BASE,'--name-only'],cwd=ROOT,text=True).splitlines()
assert all(p.startswith(ns) for p in changed),changed
request=Path('/home/AbeHiromu/.codex/attachments/d66f6ca8-0035-408a-9aae-717b908fa789/貼り付けたテキスト.txt').read_bytes()
save('evidence/static_provenance.json',dict(base_commit=BASE,fixed_science_commit=FIXED,
    required_scientific_sources=rows,synthetic_import_source_rows=source_rows(),
    prior_manifests=counts,private_originals_checked=len(private),private_originals_unchanged=True,
    private_originals_published=False,old_tracked_changes_outside_namespace=0,
    other_worktrees='not written; no global freeze imposed',
    user_request_sha256=hashlib.sha256(request).hexdigest(),request_text_not_published=True,
    formal_common_instructions='2026-10-17; read from method-note-v1.1 fixed_reference copy; sections 8/8.4 applied',
    H4_numerical_processing=0,new_scientific_authority=0))
print(json.dumps(dict(status='PASS',required_sources=len(rows),prior_manifests=counts,
                     private_originals_unchanged=True)))
