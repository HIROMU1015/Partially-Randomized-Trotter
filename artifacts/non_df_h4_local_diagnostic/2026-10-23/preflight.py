"""Read-only byte/source/runtime/contract preflight; never evaluate H4 gates."""
from pathlib import Path
import json, hashlib, subprocess, sys, importlib.util, datetime, os, shutil
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
K=ROOT/'artifacts/non_df_h4_kl_production/2026-10-23'
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=json.loads((HERE/'github_retrieval.json').read_text())
assert raw['base_commit']['value']['structuredContent']['commit']['sha']=='80147890fa91ad8270ce7643ab671727b3db5de4'
assert raw['source_commit']['structuredContent']['commit']['sha']=='eb195bf2dedd0aed9d80ad7f84948b59ea53d854'
files=raw['files']+[dict(path='artifacts/non_df_h4_kl_production/2026-10-23/L_execution_contract_DRAFT.json',**raw['draft_contract'])]
checks=[]
for record in files:
 assert record['status']=='fulfilled' and not record['value'].get('isError'),record['path']
 d=record['value']['structuredContent'];assert d['encoding']=='utf-8'
 b=d['content'].encode();assert b==(ROOT/record['path']).read_bytes(),record['path']
 gitsha=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
 assert gitsha==d['sha'],record['path']
 checks.append(dict(path=record['path'],SHA256=sha(b),Git_blob_SHA=d['sha'],bytes=len(b),status='BYTES_MATCH'))
spec=importlib.util.spec_from_file_location('L_frozen_preflight_control',ROOT/'src/trotterlib/h4_local_control.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
seal=c.read(K/'SOURCE.json');draft=c.validate_contract(c.read(K/'L_execution_contract_DRAFT.json'))
c.verify_seal(seal);c.routing(draft,seal)
assert sha((K/'SOURCE.json').read_bytes())==draft['technical_seal_sha256']
assert all(draft[k] is False for k in ('execution_authorized','grant_valid','dispatch_enabled'))
assert not Path(draft['output_namespace']).exists()
policy=Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/research_project_instructions_unified_2026-10-23.md')
receipt=dict(path=str(policy),SHA256=sha(policy.read_bytes()),version='2026-10-23',applied_sections=['8.4','10'],reuse='Unmodified sealed L/C kernel, controller, regression tests and native witness',old_fixed_contracts_changed=False)
c.save(HERE/'instruction_receipt.json',receipt)
c.save(HERE/'github_verification.json',dict(status='PASS',base_commit='80147890fa91ad8270ce7643ab671727b3db5de4',SOURCE=seal['SOURCE'],files=checks,files_checked=len(checks)))
mem={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
resource=dict(UTC=c.utc(),cpu_affinity_count=len(os.sched_getaffinity(0)),loadavg=Path('/proc/loadavg').read_text().strip(),MemAvailable=mem['MemAvailable'],filesystem_free_bytes=shutil.disk_usage(ROOT).free,pressure={name:Path('/proc/pressure',name).read_text() for name in ('cpu','memory','io')},arbitrary_resource_cap=False,shared_environment_modified=False,execution_strategy='Reuse deduplicated small2x2/4x4 kernel and ordered scalar composition with durable first-failure reservations; no unsafe threaded Arb context; scientific implementation unchanged')
c.save(HERE/'resource_before.json',resource)
c.save(HERE/'preflight.json',dict(status='PASS',UTC=c.utc(),SOURCE=seal['SOURCE'],code_files=len(seal['code_files']),runtime_files=len(seal['runtime_files']),input_files=len(seal['input_files']),python_executable_sha256=sha(Path(sys.executable).read_bytes()),sealed_code_changed=False,fresh_namespace=True,scientific_calls=0,valid_grants_issued=0,precision_bits=draft['precision_bits'],limits=draft['limits'],forbidden_calls=draft['forbidden_calls'],remote_files_checked=len(checks)))
print('PASS: GitHub9 exact blobs; SOURCE13/runtime1050/input5; fixed128/budgets; fresh namespace; zero scientific calls/grants')
