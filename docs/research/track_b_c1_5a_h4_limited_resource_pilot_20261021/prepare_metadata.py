"""Freeze static identity and execution settings before the single scientific job."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from support import HERE,ROOT,BASE,FIXED,MODEL,sha,digest,canonical,save,verify_rows,git,resource_snapshot

old=ROOT/'docs/research/track_b_c1_3_r2_reference_recovery_c1_4_20261021'
require_text=(HERE/'technical_test_final.stderr.log').read_text()
assert 'Ran 12 tests' in require_text and require_text.rstrip().endswith('OK')
worker_count=min(max(1,len(os.sched_getaffinity(0))//3),32)
contract=dict(schema='C1_5A_FIXED_H4_EXPLORATORY_V1',base_commit=BASE,fixed_science_commit=FIXED,
    methods=['ordinary','full_return','closed_P5_full'],trials_per_method=1024,microsteps=4,
    scientific_attempts=1,retry=0,seed=20261021,bit_rule='diagnostic.Bits: C1_5A_BITS_V1 canonical key + NUL + draw uint64BE + block uint32BE; SHA256 concatenate then high width bits',
    model='linear H4 1.00 Angstrom / STO-3G / DF12 / R1 generation-prefix LD3 / 8 qubits / 324 labels / 13 bases',
    model_SHA256=MODEL,partition_hash='16b35c1dde1a6807d39c733defd87e315e4ac7317a831b240d0f7214e744a834',
    snapshot_SHA256='3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a',
    T='4/5',q_outer=1,r_RTE=4,m=5,sigma=1,tail_substep_time='1/5',outer='symmetric S2 reference schedule; deterministic halves not built',
    measured_circuit_scope='TAIL_ONLY_PLUS_SCALAR',scalar_policy='one outer P(-T*(constant+extracted_tail_identity)); deterministic halves UNKNOWN',
    precision=dict(probability_bits=2048,root_bits=4096,eta='1/1000000000000',rho='1/1000000000000',
        angle_tau='1/1000000000000',angle_root_bits=256,angle_terms=64),
    finite_target='common R1 finite P5 at each tail microstep; four ordered independent-design steps; pseudorandom IID not verified',
    canonical_certificate='UNCERTIFIED_EXPLORATORY; existing local rounding guards; no Full preflight transfer',
    full_preflight='unchanged F3 P0-P5 with supplied precision; failure STOP, no refinement',
    architecture='all four classical steps before circuit build; no early-zero shortcut; no quantum shots or fake +/-1',
    batch_trials=32,workers_per_method=worker_count,method_coordinators='three independent subprocess groups started in methods order, execute concurrently',
    worker_policy='fork read-only setup; per-worker cold angle/event caches, same policy for all; no resynthesis; unique deterministic batches',
    group_selector='unchanged existing dyadic_index for all methods; P5 linear O(L^2) group scan retained, not optimized-best runtime',
    timing='setup includes input/registry separately from sampler-normalizer; process CPU and wall stages; pool lifecycle/hierarchy CPU separately',
    root_queries='Full local packet once for raw-reduced attempts; P5 source-derived requery counters; internal exact arithmetic included in stage CPU, not exhaustively counted',
    outputs='result/<method>/{batches/*.jsonl.gz,events/*.json.gz,angles/*.json.gz,receipts,summary,setup,terminal}; raw stdout/stderr; one consumed marker',
    data_encoding='F3 exact hex integer/rational Evidence; SHA256 content addresses; deterministic gzip mtime0; floats only observational metrics',
    aggregation='inside one job: exact trace/Evidence validation; empirical binary64 fsum second moment over all 1024 incl zeros, exact observed max; no estimator mean',
    no_resource_winner=True,no_sufficient_N=True,native_acquisition=False,quantum_measurements=False,
    failure='first technical failure sets own shared abort; cancel own pending batches/process groups; preserve partial outputs; no retry/source repair',
    preservation='new namespace only; old SOURCE/results/authorizations/grant/claim/ledger/marker/STOP/private reviews unchanged; other worktrees not frozen',
    resource_policy='no artificial CPU/RAM caps; affinity-sized workers; observed free resources; shared OS/venv/cache/other jobs unchanged',
    mandatory_STOP=True)
save(HERE/'execution_contract.json',contract)
sources=json.loads((old/'source_manifest.json').read_text())['files']
extra=[str(p.relative_to(ROOT)) for p in sorted(HERE.glob('*.py'))]
extra+=['docs/research/track_b_c1_2_c1_4_integrated_adapter_validation_20261021/adapter.py',
        str((HERE/'execution_contract.json').relative_to(ROOT)),
        str((HERE/'technical_test_final.stderr.log').relative_to(ROOT)),
        str((HERE/'technical_tests.json').relative_to(ROOT))]
existing={r['path'] for r in sources}
for path in extra:
    if path not in existing:
        raw=(ROOT/path).read_bytes();sources.append(dict(path=path,bytes=len(raw),sha256=sha(raw)))
verify_rows(sources)
guard_names=('return_aggregation.py','g7_generator.py','g7_provider.py','g9_p5.py','g6_f3_generator.py',
             'g6_f3_precision.py','g6_f3_contract.py','df_rte_tail.py','df_rte_qiskit.py','rpe_hadamard_interrogation.py')
guard=[r for r in sources if r['path'].startswith(str(HERE.relative_to(ROOT))) or Path(r['path']).name in guard_names or r['path'].endswith('/adapter.py')]
save(HERE/'source_manifest.json',dict(base_commit=BASE,files=sources,online_guard_files=guard))
inputs=json.loads((old/'input_manifest.json').read_text())['files'];verify_rows(inputs)
save(HERE/'input_manifest.json',dict(model_SHA256=MODEL,files=inputs,online_guard_files=inputs,
    snapshot_use='byte hash only; no NPZ numerical arrays/state opened'))
runtime=json.loads((old/'runtime_manifest.json').read_text())['files'];verify_rows(runtime)
save(HERE/'environment_manifest.json',dict(files=runtime,python=sys.version,executable=sys.executable,
    packages={k:importlib.metadata.version(k) for k in ('numpy','scipy','qiskit','qiskit-nature','qiskit-algorithms','openfermion','pyscf','numba','h5py')},
    affinity_count=len(os.sched_getaffinity(0)),workers_per_method=worker_count,
    shared_environment_modified=False,resources=resource_snapshot()))
private=json.loads((old/'preservation_baseline.json').read_text())['private_originals']
for row in private:
    assert digest(row['path'])==row['sha256']
request=Path('/home/AbeHiromu/.codex/attachments/e3431131-cae6-4a23-beed-bfa7cba825f5/貼り付けたテキスト.txt')
save(HERE/'preflight_static.json',dict(status='TECHNICAL_SOURCE_READY_TO_SEAL',scientific_attempts=0,
    tests=12,source_files=len(sources),input_files=len(inputs),runtime_files=len(runtime),
    user_request_SHA256=digest(request),request_text_not_published=True,private_originals_SHA256_unchanged=True,
    R1_probabilities_not_instantiated_in_preparation=True,mandatory_gate_before_science='SOURCE/publication/new authorization/marker then fixed model load/preflight inside one job'))
print(json.dumps(dict(status='READY_TO_SEAL',workers_per_method=worker_count,source_files=len(sources),inputs=len(inputs))))
