"""Read existing JSON/source/config only. Never import research code or load NPZ.

Run from the dedicated checkout of 3354b8b plus this collector. Outputs must be
fresh. No preparation, numeric validation, RNG, circuit build or compilation.
"""
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
WP11 = Path('/home/AbeHiromu/projects/partially-randomized-trotter-wp11')
MAIN = Path('/home/AbeHiromu/projects/partially-randomized-trotter')
BASE = '3354b8baa70b5ed7a798069db8f2f4c4fdd4b526'
WP11_SHA = '8b574c5431acf8d7fc2059ecc2c3b56ff19c37f0'
COMMITS = ['efa90e0b14a95c296b305bfeea37867e56c83a0d',
           '26aa95c7b3f5ecd1f358280e58073ca2ee44ea5c',
           '7753eb71844e429facdb6f32c2def6e37379541f', WP11_SHA, BASE]
HASHES = ['9034397f5bde1f20cb712578d14245f4dac1236413d34642a75d83d867b52086',
          '50c77f5455cc1a3c0710ed3c7588efc957b9cead7436bd0a6d851af61738a4bc',
          'c51e35da56f3de1646896339f38da3cb40dcb8405a24d781cef134a654bab642']
OUT = ROOT/'artifacts/o2_a1_opt2_provenance/2026-11-06'
OUT.mkdir(parents=True, exist_ok=False)
EVIDENCE = {}

def digest(b): return hashlib.sha256(b).hexdigest()
def ref(p, location='base_checkout'):
    p = Path(p)
    if location == 'base_checkout': base = ROOT
    elif location == 'wp11_local': base = WP11
    else: base = MAIN
    data = p.read_bytes()
    key = location+':'+str(p.relative_to(base))
    EVIDENCE[key] = dict(location=location, path=str(p.relative_to(base)),
                        sha256=digest(data), bytes=len(data))
    return key

def read(p, location='base_checkout'):
    ref(p, location)
    return json.loads(Path(p).read_text())

def write(name, data):
    with (OUT/name).open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True)

def existing_keys(x):
    result = set()
    if isinstance(x, dict):
        for k, v in x.items(): result.add(k); result.update(existing_keys(v))
    elif isinstance(x, list):
        for v in x: result.update(existing_keys(v))
    return result

# The old hash carriers are inspected as records, never recomputed.
args = ['rg', '-l', '-F']
for h in HASHES: args += ['-e', h]
q = subprocess.run(args+['artifacts','docs/archive'], cwd=ROOT,
                   capture_output=True, text=True, check=True)
carriers = []
for rel in sorted(q.stdout.splitlines()):
    p=ROOT/rel; ref(p)
    if p.suffix != '.json': continue
    d=json.loads(p.read_text()); keys=existing_keys(d)
    carriers.append(dict(path=rel, contains=[h for h in HASHES if h in p.read_text()],
        identity_locations={k:v for k,v in d.items() if k in ('physical_instance','hamiltonian','system')},
        provenance=d.get('provenance'),
        complete_payload_keys_present=sorted(keys & {'pre_threshold_components',
             'deterministic_fragments','randomized_fragments','basis_definitions',
             'basis_change_operations'}), current_a1_record='/o2_a1_' in rel))

# Search for payload-like schema, including files without any expected hash.
key_search={}
for key in ('pre_threshold_components','deterministic_fragments',
            'coefficient_abs','basis_change_operations'):
    q=subprocess.run(['rg','-l','-F','"'+key+'"','artifacts','docs/archive'],
                     cwd=ROOT,capture_output=True,text=True)
    paths=sorted(q.stdout.splitlines())
    key_search[key]=dict(count=len(paths),by_result_directory=dict(Counter(
        '/'.join(Path(p).parts[:2]) for p in paths)),
        exact_paths=paths if key in ('pre_threshold_components','deterministic_fragments') else None)
wp_paths=subprocess.check_output(['rg','--files','--hidden','--no-ignore',
    '-g','!.venv-server-opt2/**','-g','!.git'],cwd=WP11,text=True).splitlines()
key_search['wp11_file_inventory_count']=len(wp_paths)
key_search['wp11_binary_payload_candidates']=[p for p in wp_paths
    if Path(p).suffix in ('.qpy','.pkl','.pickle','.sqlite','.db')]
write('PAYLOAD_SEARCH.json',dict(scope='base artifact/archive text plus existing WP11 files excluding venv',
    old_hash_carriers=carriers,key_search=key_search,
    original_complete_partition_recovered=False,original_complete_basis_operations_recovered=False,
    original_complete_component_catalog_recovered=False,
    absence_limit='Not a proof of nonexistence on other hosts, inaccessible stores or unexamined binary formats.'))

# Read all original checkpoint metadata, omitting price/signal result arrays.
cohorts={}; checkpoints=[]; source_sets={}
for f in sorted((WP11/'artifacts/research_direction_full_opt2').rglob('checkpoints/*.json')):
    d=read(f,'wp11_local'); result=d.get('result',{}); keys=existing_keys(d)
    metadata={k:result[k] for k in ('adapter','dependency_versions','thread_environment',
              'source_sha256','input_sha256','cuda_visible_devices') if k in result}
    cohort=digest(json.dumps(metadata,sort_keys=True).encode())
    cohorts.setdefault(cohort,metadata)
    source_sets[digest(json.dumps(metadata.get('source_sha256',{}),sort_keys=True).encode())]=metadata.get('source_sha256',{})
    checkpoints.append(dict(path=str(f.relative_to(WP11)),sha256=digest(f.read_bytes()),
        status=d.get('status'),started_at_utc=d.get('started_at_utc'),cohort=cohort,
        target_identity_or_payload_keys=sorted(keys & {'hamiltonian_hash','partition_hash',
          'preparation_hash','tail_hash','component_id','pre_threshold_components',
          'deterministic_fragments','basis_change_operations'})))
write('CHECKPOINT_METADATA.json',dict(source_root=str(WP11),wp11_head=WP11_SHA,
    classification='Structured excerpts of existing ignored checkpoint metadata; no raw cost recomputation',
    checkpoint_count=len(checkpoints),cohorts=cohorts,checkpoints=checkpoints))

# Recover recorded source bytes from known fixed commits, path history and archives.
selected = ['artifacts/research_direction_full_scope/2026-09-22/wp05a_full_controlled_interrogation_connection_v1.json',
 'artifacts/research_direction_sequence_policy/2026-09-22/wp06b_sequence_policy_proxy_bridge_v1.json',
 'artifacts/research_direction_structure_pilot/2026-09-22/wp06a_circuit_structure_pilot_v1.json']
recorded=[]; environment_records=[]
for rel in selected:
    d=read(ROOT/rel); prov=d['provenance']
    recorded.append(dict(record=rel,source_sha256=prov.get('source_sha256',{})))
    environment_records.append(dict(record=rel,command=prov.get('command'),
        git_commit=prov.get('git_commit'),generated_at_utc=prov.get('generated_at_utc'),
        versions={k:v for k,v in prov.items() if k.endswith('_version')},
        evidence_status=prov.get('evidence_status'),
        dirty_status=prov.get('git_worktree_status_before_generation')))
for key, sources in source_sets.items():recorded.append(dict(record='checkpoint_source_set:'+key,source_sha256=sources))
recovery=[]; cache={}
for item in recorded:
    rows=[]
    for path, expected in item['source_sha256'].items():
        identity=(path,expected)
        if identity not in cache:
            log=git('log','--all','--format=%H','--',path)
            commits=list(dict.fromkeys(COMMITS+log.stdout.decode().splitlines()))
            matches=[]
            for c in commits:
                p=git('show',c+':'+path)
                if p.returncode==0 and digest(p.stdout)==expected:
                    matches.append(dict(kind='git',commit=c,path=path,sha256=expected))
            for archive in (ROOT/'docs/archive').rglob(Path(path).name+'.original'):
                if archive.as_posix().endswith(path+'.original') and digest(archive.read_bytes())==expected:
                    ref(archive);matches.append(dict(kind='archive',path=str(archive.relative_to(ROOT)),sha256=expected))
            cache[identity]=matches
        rows.append(dict(path=path,recorded_sha256=expected,recovered=cache[identity]))
    recovery.append(dict(record=item['record'],files=rows,
                         all_recorded_sources_recovered=all(r['recovered'] for r in rows)))

# Static changed-function inventory. Import-time or scientific execution is absent.
changes=[]
for path in ('src/trotterlib/df_partial_s2.py','src/trotterlib/pf_delta_validation.py'):
    old=git('show',WP11_SHA+':'+path).stdout;new=(ROOT/path).read_bytes();ref(ROOT/path)
    def functions(data):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(data).body
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
    a,b=functions(old),functions(new)
    changes.append(dict(path=path,wp11_sha256=digest(old),current_sha256=digest(new),
        changed_top_level_definitions=[k for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k)]))
write('SOURCE_LINEAGE.json',dict(recorded_source_recovery=recovery,changed_definitions=changes,
    hash_input_paths=['src/trotterlib/df_partial_randomized_pf.py:df_fragment_weight/rank_df_fragments',
      'src/trotterlib/df_partial_s2.py:_partition_hash/prepare_df_partial_s2',
      'src/trotterlib/df_rte_tail.py:extract_df_diagonal_tail/canonical_basis_hash',
      'src/trotterlib/df_trotter/decompose.py:diag_hermitian',
      'src/trotterlib/df_trotter/ops.py:build_df_blocks/_import_fermionic_gaussian_decomposition'],
    limit='Recovery certifies only explicitly recorded source files, not complete transitive or binary environments.'))
(OUT/'SOURCE_DIFFERENCES.diff').write_bytes(git('diff',WP11_SHA,BASE,'--',
    'src/trotterlib/df_partial_s2.py','src/trotterlib/pf_delta_validation.py').stdout)

# Literal metadata comparison, no operation matrices or numerical equivalence test.
old=read(ROOT/selected[1])['support_basis_library']['definitions']
current_path='artifacts/o2_a1_q1_q2l/2026-11-06/diagnostic_v2/PREPARATION.json'
current=read(ROOT/current_path)
a={x['source_basis_hash'] for x in old};b={x['basis_hash'] for x in current['components']}
write('BASIS_METADATA.json',dict(old_record=selected[1],current_record=current_path,
    old_source_basis_hashes=sorted(a),current_tail_basis_hashes=sorted(b),shared=sorted(a&b),
    old_only=sorted(a-b),current_only=sorted(b-a),
    saved_support_definition_keys=sorted(old[0]),
    interpretation='Literal basis metadata fingerprints differ; physical/operator/target difference has NOT been established.',
    not_a_complete_catalog=True,operator_comparison_performed=False))

# Original server run records versus the earlier .venv311 hash-carrier records.
server_runs=[]
for f in sorted((ROOT/'artifacts/research_direction_full_opt2').rglob('run_provenance.json')):
    d=read(f)
    server_runs.append({k:d.get(k) for k in ('command','git_commit','git_branch','python_version',
        'numpy_version','qiskit_version','cpu','thread_environment','source_sha256',
        'input_sha256','generated_at_utc','evidence_status')})
venvs=[]
for base,location in ((MAIN,'main_local'),(WP11,'wp11_local')):
    for f in base.glob('.venv*/pyvenv.cfg'):
        ref(f,location);venvs.append(dict(path=str(f),contents=f.read_text()))
sp=WP11/'.venv-server-opt2/lib/python3.12/site-packages'
installed=[]
for name in ('numpy','scipy','qiskit','qiskit_nature'):
    for f in sp.glob(name+'-*.dist-info/METADATA'):
        ref(f,'wp11_local'); row=dict(path=str(f),name_and_version=[x for x in f.read_text().splitlines() if x.startswith(('Name:','Version:'))])
        w=f.parent/'WHEEL'
        if w.exists():ref(w,'wp11_local');row['wheel']=w.read_text()
        installed.append(row)
requirements=ROOT/'requirements.txt';ref(requirements)
write('ENVIRONMENT.json',dict(old_hash_carrier_records=environment_records,server_run_records=server_runs,
    current_venv_config=venvs,current_installed_metadata=installed,requirements_path='requirements.txt',
    requirements_contents=requirements.read_text(),
    old_venv311_present_at_examined_paths={str(b/'.venv311'):(b/'.venv311').exists() for b in (MAIN,WP11)},
    exact_original_wheel_blas_lapack_lock_recovered=False,
    limit='Current venv metadata is a present observation, not proof that original binary dependencies were preserved.'))
ref(Path(__file__))
write('EVIDENCE_INDEX.json',dict(base_commit=BASE,wp11_commit=WP11_SHA,
    collector_sha256=digest(Path(__file__).read_bytes()),records=list(EVIDENCE.values()),
    operations='filesystem bytes/JSON/AST and git history reads only',
    scientific_calls=dict(npz_load=0,preparation=0,cdf=0,moments=0,bias=0,rng=0,state_solve=0,full_build=0,full_compile=0)))
write('STOP.json',dict(status='MANDATORY_STOP_STATIC_PROVENANCE_ONLY',base_commit=BASE,
    original_complete_payload_recovered=False,original_binary_environment_recovered=False,
    original_target_equivalence_certified=False,scientific_target_mismatch_established=False,
    no_new_target_adopted=True,no_scientific_retry=True,measurement_authorized=False,
    next_decision='Recover new external original payload/environment evidence, or approve a new consistent finite-target contract; no automatic science.'))
print(json.dumps(dict(carrier_json_files=len(carriers),checkpoint_count=len(checkpoints),
    source_recovery=[dict(record=x['record'],total=len(x['files']),recovered=sum(bool(y['recovered']) for y in x['files'])) for x in recovery],
    basis_hashes=dict(old=len(a),current=len(b),shared=len(a&b)),scientific_calls=0),indent=2))
