"""Stdlib control plane. No scientific imports, grant issuer, or default approval."""
from dataclasses import asdict
import hashlib
import importlib.util
import importlib.machinery
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('h2_fixed_policy', ROOT/'src/trotterlib/h2_common_native_policy.py')
policy = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = policy
_spec.loader.exec_module(policy)
Stop, require = policy.Stop, policy.require
SERIES = 'H2_COMMON_PRODUCTION_V1_PROPOSED'
ACQUISITION = policy.ACQUISITION_SOURCE
PROVIDER = policy.PROVIDER_SOURCE
BINDING = dict(snapshot='62c6481bd1f8b05ec3b399b3a5293873796737866409049d808e8fe432dcacaa',
    hamiltonian='d0bc260db6fc0a485fe518050075c18ccc0dff532253dcc719a52a61f41afa80',
    state='080b7ad2a46f38ed51e2989eb399ceda4cae95fca688722959ee65463cf174bf')
KINDS = ('bootstrap', 'setup', 'prepare', 'diagnose', 'path_signal', 'reference_rebind',
         'native_validation', 'compose', 'lowering', 'wrapper', 'compile', 'transpile',
         'simulate', 'request_factory', 'rte_event', 'occupation_trajectory',
         'allowance', 'accounting', 'ranking')
BRANCHES = ('DIAGNOSTICS', 'COMPARE_REUSE_PR', 'PR_TWO_NATIVE_CONTROLS')
REUSE_FIELDS = ('hamiltonian', 'state', 'basis', 'rotation', 'preparation', 'schedule',
    'deterministic_lowering', 'event_law', 'events', 'wrapper', 'scalar_phase',
    'compiler_source', 'compiler_settings', 'runtime_closure', 'correctness', 'cohort')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def read(path):
    def bad(x): raise Stop('NONFINITE_JSON:' + x)
    return json.loads(Path(path).read_text(), parse_constant=bad)


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n'); f.flush()
        import os
        os.fsync(f.fileno())


def prior_blob(path):
    return subprocess.check_output(['git','-C',str(ROOT),'show',
        'cc7404ed189f8c768ecc50a2066485975c9c68f4:'+path])


def reference_bytes_check(path):
    original=prior_blob('artifacts/resource_applicability/track_a_pf_order_reversal_h2_common_policy_preparation/2026-10-23/saved_input_excerpts.json')
    require(Path(path).read_bytes()==original,'ORIGINAL_REFERENCE_PROJECTION_BYTES')


def saved_cost_tasks(candidate):
    original=json.loads(prior_blob('artifacts/resource_applicability/track_a_pf_order_reversal_h2_metadata_receipt/2026-10-23/received/task_acceptance_67.json'))
    rows=[]
    for row in original['tasks']:
        if row['task']['candidate']['id']!=candidate.id:continue
        result=row['result']['saved'];axes={a['axis']:a for a in result['axis_costs']}
        require(result['SOURCE']==ACQUISITION and result['native_scientific_PASS'] is True and
                result['conditional_sampling'] is False and result['cohort_pooling'] is False,'ORIGINAL_TASK_ACCEPTANCE')
        rows.append(dict(sample=row['task']['sample'],cosine=axes['cosine']['metrics']['rz_count'],
            sine=axes['sine']['metrics']['rz_count'],receipt_sha256=policy.digest(result),accepted=True))
    return rows


def validate_reuse_preflight(cfg,seen):
    for candidate,path in zip(policy.CANDIDATES[2:],cfg['reuse_witnesses']):
        witness=read(path)
        require(set(witness.get('proof',{}))=={'events','correctness','cohort'},'REUSE_PROOF_PATHS')
        require(all(Path(p) in seen for p in witness['proof'].values()),'REUSE_PROOF_NOT_SEALED')
        reuse_costs(witness,witness['proposed'],candidate)


def counts(branch):
    require(branch in BRANCHES, 'FIXED_BRANCH_ONLY')
    c = dict.fromkeys(KINDS, 0)
    c.update(bootstrap=1, setup=1, prepare=2)
    if branch != 'PR_TWO_NATIVE_CONTROLS':
        c.update(diagnose=4, path_signal=8, reference_rebind=4)
    if branch != 'DIAGNOSTICS':
        c.update(native_validation=2, compose=2, lowering=10, wrapper=4,
                 compile=4, transpile=4, simulate=12)
    if branch == 'PR_TWO_NATIVE_CONTROLS':
        c.update(request_factory=10, rte_event=192, occupation_trajectory=2)
    if branch == 'COMPARE_REUSE_PR':
        c.update(allowance=4, accounting=4, ranking=1)
    return c


class ImportSeal:
    """Check each new file-backed import before its loader executes code."""
    def __init__(self, cfg):
        self.index={r['path']:r['sha256'] for r in cfg['files']}

    def find_spec(self, fullname, path=None, target=None):
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec is not None and spec.origin not in (None,'built-in','frozen'):
            p=Path(spec.origin).resolve()
            if p.suffix=='.pyc':
                p=Path(importlib.util.source_from_cache(str(p))).resolve()
            require(str(p) in self.index and sha(p)==self.index[str(p)], 'IMPORT_NOT_SEALED:'+fullname)
        return spec

    def install(self):
        self.original_get_code=importlib.machinery.SourceFileLoader.get_code
        index=self.index
        def checked_source(loader,fullname):
            p=Path(loader.get_filename(fullname)).resolve()
            require(str(p) in index and sha(p)==index[str(p)],'DIRECT_SOURCE_IMPORT_NOT_SEALED:'+fullname)
            # Recompile sealed source bytes; never execute an unsealed cache file.
            return compile(p.read_bytes(),str(p),'exec',dont_inherit=True,optimize=sys.flags.optimize)
        importlib.machinery.SourceFileLoader.get_code=checked_source
        sys.meta_path.insert(0,self)

    def remove(self):
        sys.meta_path.remove(self)
        importlib.machinery.SourceFileLoader.get_code=self.original_get_code


def seal_check(cfg):
    require(type(cfg['files']) is list and bool(cfg['files']), 'SEALED_CLOSURE_REQUIRED')
    seen = set()
    for row in cfg['files']:
        require(set(row) == {'path', 'sha256', 'role'}, 'SEALED_FILE_SCHEMA')
        p = Path(row['path'])
        require(p.is_absolute() and str(p.resolve()) == row['path'] and p not in seen,
                'UNIQUE_CANONICAL_SEALED_PATH')
        seen.add(p); policy.sha(row['sha256'], 'sealed_sha')
        require(p.is_file() and sha(p) == row['sha256'], 'SEALED_FILE_MISMATCH:' + row['role'])
    return seen


def source_check(cfg, seen):
    roots = {'runner': ROOT, 'acquisition': Path(cfg['acquisition_root']),
             'provider': Path(cfg['provider_root'])}
    commits = {'runner': cfg['source_commit'], 'acquisition': ACQUISITION, 'provider': PROVIDER}
    for role, root in roots.items():
        entries = [r for r in cfg['source_files'] if r['root'] == role]
        require(entries, 'SOURCE_CLOSURE:' + role)
        requested=[commits[role]+':'+r['path'] for r in entries]
        require(all('\n' not in p and '\r' not in p for p in requested),'SOURCE_PATH_LINES')
        stream=subprocess.check_output(['git','-C',str(ROOT),'cat-file','--batch'],
            input=('\n'.join(requested)+'\n').encode())
        offset=0
        for r in entries:
            p = (root/r['path']).resolve()
            require(p.is_relative_to(root.resolve()) and p in seen, 'SOURCE_PATH_NOT_SEALED')
            end=stream.find(b'\n',offset);header=stream[offset:end].split()
            require(len(header)==3 and header[1]==b'blob','SOURCE_GIT_BLOB_REQUIRED')
            size=int(header[2]);offset=end+1;b=stream[offset:offset+size];offset+=size+1
            require(hashlib.sha256(b).hexdigest() == r['sha256'] == sha(p), 'FIXED_SOURCE_BYTES')
        require(offset==len(stream),'EXACT_SOURCE_OBJECT_STREAM')
    required = {'scripts/h2_common_production_contract.py', 'scripts/h2_common_production_backend.py',
                'scripts/h2_common_production_diagnostics.py', 'scripts/run_h2_common_production.py',
                'src/trotterlib/h2_common_native_policy.py'}
    require(required <= {r['path'] for r in cfg['source_files'] if r['root'] == 'runner'}, 'RUNNER_SOURCE_CLOSURE')
    for _, (commit, path, _) in policy.BINDINGS.items():
        role = 'acquisition' if commit == ACQUISITION else 'provider'
        require(any(r['root']==role and r['path']==path for r in cfg['source_files']), 'KERNEL_SOURCE_CLOSURE')
    acquisition_required={'scripts/h46_m5_contract.py','scripts/h46_new2_core.py',
        'scripts/h46_pr_pilot_repair.py','references/h46_m5_best_formula_v1.json'}
    require(acquisition_required <= {r['path'] for r in cfg['source_files'] if r['root']=='acquisition'}, 'ACQUISITION_DEPENDENCIES')
    indexed={(r['root'],r['path']) for r in cfg['source_files']}
    for p in seen:
        for role,root in roots.items():
            if p.is_relative_to(root.resolve()):
                require((role,str(p.relative_to(root.resolve()))) in indexed,'SEALED_SOURCE_MUST_BE_GIT_PINNED')


def preflight(cfg):
    required = {'schema', 'series', 'branch', 'source_commit', 'physical', 'compiler',
        'native_policy', 'allowance_policy', 'reference_policy', 'candidates', 'budget',
        'files', 'source_files', 'acquisition_root', 'provider_root', 'inputs', 'binding',
        'runtime', 'run_root', 'reuse_witnesses', 'retry', 'mandatory_STOP'}
    require(type(cfg) is dict and set(cfg)==required, 'CONTRACT_FIELDS')
    require(cfg['schema']=='h2_common_production_contract_v1' and cfg['series']==SERIES, 'CONTRACT_SERIES')
    require(policy.strict_equal(cfg['physical'], policy.PHYSICAL) and
            policy.strict_equal(cfg['compiler'], policy.COMPILER), 'FIXED_PHYSICAL_COMPILER')
    require(policy.strict_equal(cfg['candidates'], [asdict(c) for c in policy.CANDIDATES]), 'FOUR_FIXED_CANDIDATES')
    require(policy.strict_equal(cfg['budget'], counts(cfg['branch'])), 'EXACT_CALL_BUDGET')
    require(cfg['native_policy']==policy.NATIVE_POLICY and cfg['allowance_policy']==policy.ALLOWANCE_POLICY,
            'POLICY_VERSION')
    require(cfg['reference_policy']=='REUSE_OLD_FP_SCALAR_AND_COMBINED_EMPIRICAL_DIAGNOSTIC', 'REFERENCE_POLICY_BRANCH_BLOCKED')
    require(cfg['retry'] is False and cfg['mandatory_STOP'] is True, 'STOP_NO_RETRY')
    require(type(cfg['source_commit']) is str and len(cfg['source_commit'])==40 and
            all(c in '0123456789abcdef' for c in cfg['source_commit']), 'SCIENCE_SOURCE_COMMIT')
    run = Path(cfg['run_root'])
    require(run.is_absolute() and str(run.resolve())==str(run) and not run.exists(), 'UNUSED_RUN_ROOT')
    seen = seal_check(cfg)
    require(not any(p.is_relative_to(run) or run.is_relative_to(p) for p in seen), 'INPUT_OUTPUT_DISJOINT')
    source_check(cfg, seen)
    require(policy.strict_equal(cfg['binding'], BINDING), 'FIXED_SAVED_H2_BINDING')
    for key, value in cfg['binding'].items(): policy.sha(value, key)
    needed = {'snapshot', 'snapshot_receipt', 'old_reference'}
    if cfg['branch']=='PR_TWO_NATIVE_CONTROLS': needed.add('control_seeds')
    require(set(cfg['inputs'])==needed and all(Path(p) in seen for p in cfg['inputs'].values()), 'SEALED_INPUTS')
    if cfg['branch']=='PR_TWO_NATIVE_CONTROLS':
        seeds=read(cfg['inputs']['control_seeds'])
        require(set(seeds)=={c.id for c in policy.CANDIDATES[2:]},'TWO_CONTROL_SEED_GROUPS')
        for candidate in policy.CANDIDATES[2:]:
            pcs=1 if candidate.formula=='2nd' else 3
            require(set(seeds[candidate.id])=={str(o)+':'+str(p) for o in range(candidate.q) for p in range(pcs)},'EXACT_SAVED_PIECE_SEEDS')
            for value in seeds[candidate.id].values():policy.integer(value,'seed')
    require(cfg['runtime']['python_sha256']==sha(Path(sys.executable).resolve()), 'PYTHON_RUNTIME_SEAL')
    require(cfg['runtime']['qiskit']=='1.3.0' and cfg['runtime']['numpy']=='1.26.4', 'RUNTIME_VERSIONS')
    if cfg['runtime'].get('closure_mode')=='STATIC_CLOSED_IMPORT_SET_V1':
        from h2_diagnostics_binding import verify_runtime_binding
        verify_runtime_binding(cfg)
    else:
        require(cfg['runtime'].get('loaded_closure_complete') is True, 'LOADED_CLOSURE_NOT_ATTESTED')
    reference = read(cfg['inputs']['old_reference'])['old_reference']
    reference_bytes_check(cfg['inputs']['old_reference'])
    require(all(reference['binding'][k]==v for k,v in cfg['binding'].items()) and reference['T']==.8,
            'REFERENCE_INPUT_BINDING')
    require(set(reference['reference'])=={'real','imag'}, 'REFERENCE_COMPLEX_SCHEMA')
    for v in reference['reference'].values(): policy.number(v, 'reference_scalar', nonnegative=False)
    policy.number(reference['error'], 'reference_uncertainty')
    require(cfg['branch']!='COMPARE_REUSE_PR' or len(cfg['reuse_witnesses'])==2, 'PR_C_WITNESSES_REQUIRED')
    for path in cfg['reuse_witnesses']: require(Path(path) in seen, 'REUSE_WITNESS_SEAL')
    if cfg['branch']=='COMPARE_REUSE_PR':validate_reuse_preflight(cfg,seen)
    return dict(status='TECHNICAL_PREFLIGHT_PASS', science_GO=False, contract_sha256=policy.digest(cfg))


def authorize(cfg, approval, grant):
    preflight(cfg); d = policy.digest(cfg)
    require(type(approval) is dict and approval.get('issuer')=='USER_EXPLICIT' and
            approval.get('execution_authorized') is True and approval.get('contract_sha256')==d,
            'EXPLICIT_USER_SCIENCE_APPROVAL_REQUIRED')
    require(approval.get('policies_adopted')==[cfg[k] for k in ('native_policy','allowance_policy','reference_policy')],
            'EXPLICIT_POLICY_ADOPTION_REQUIRED')
    require(type(grant) is dict and set(grant)=={'id','contract_sha256','approval_sha256',
        'run_root','budget','state','retry'}, 'GRANT_FIELDS')
    require(type(grant['id']) is str and bool(grant['id']) and grant['state']=='UNUSED' and grant['retry'] is False,
            'UNUSED_ONE_SHOT_GRANT_REQUIRED')
    require(grant['contract_sha256']==d and grant['approval_sha256']==policy.digest(approval) and
            grant['run_root']==cfg['run_root'] and policy.strict_equal(grant['budget'],cfg['budget']), 'GRANT_SCOPE')


class Session:
    """One process, exclusive output root, nonrefundable per-kind reservations."""
    def __init__(self, cfg, approval, grant):
        authorize(cfg, approval, grant)
        self.cfg, self.used, self.active = cfg, dict.fromkeys(KINDS,0), None
        self.root = Path(cfg['run_root']); self.root.mkdir(exist_ok=False)
        save(self.root/'CONSUMED.json', dict(grant=grant, approval_sha256=policy.digest(approval),
              state='CONSUMED_NO_RETRY_NO_REFUND'))
        self.marker_sha=sha(self.root/'CONSUMED.json'); self.sequence=0
        (self.root/'ledger').mkdir(); (self.root/'results').mkdir()

    def verify(self):
        require(sha(self.root/'CONSUMED.json')==self.marker_sha, 'CONSUMED_MARKER_CHANGED')
        seal_check(self.cfg)

    def take(self, kind, amount=1):
        require(kind in KINDS and type(amount) is int and amount>0, 'CALL_KIND')
        require(self.used[kind]+amount<=self.cfg['budget'][kind], 'CALL_BUDGET_EXCEEDED:'+kind)
        self.sequence+=1
        save(self.root/'ledger'/('%06d.json'%self.sequence), dict(kind=kind,amount=amount,
             nonrefundable=True,contract_sha256=policy.digest(self.cfg)))
        self.used[kind]+=amount

    def call(self, kind, fn, *args, **kwargs):
        self.take(kind)
        return fn(*args, **kwargs)

    def loaded_closure(self):
        sealed={r['path'] for r in self.cfg['files']}
        for name, module in tuple(sys.modules.items()):
            f=getattr(module,'__file__',None)
            if not f or str(f).startswith('<'): continue
            p=Path(f).resolve()
            if p.suffix=='.pyc':
                try:p=Path(importlib.util.source_from_cache(str(p))).resolve()
                except ValueError:pass
            require(str(p) in sealed, 'UNSEALED_LOADED_MODULE:'+name)
        if self.cfg['runtime'].get('closure_mode')=='STATIC_CLOSED_IMPORT_SET_V1':
            from h2_diagnostics_binding import verify_mapped_runtime
            verify_mapped_runtime(self.cfg)


def reuse_costs(witness, current, candidate):
    require(set(witness)=={'candidate','saved','proposed','all_identities_equal','tasks','correctness_scope','proof'}, 'REUSE_SCHEMA')
    require(witness['candidate']==candidate.id and witness['all_identities_equal'] is True, 'REUSE_CANDIDATE')
    require(set(witness['saved'])==set(REUSE_FIELDS)==set(witness['proposed']), 'REUSE_IDENTITY_FIELDS')
    for key in REUSE_FIELDS:
        policy.sha(witness['saved'][key],key); policy.sha(witness['proposed'][key],key)
        require(witness['saved'][key]==witness['proposed'][key]==current[key], 'REUSE_IDENTITY:'+key)
    require(witness['correctness_scope']=='SAVED_COHORT_ACCEPTANCE_WITH_DECLARED_VECTOR_GATE_SCOPE', 'REUSE_GATE_SCOPE')
    require(policy.strict_equal(witness['tasks'],saved_cost_tasks(candidate)), 'ORIGINAL_PAIRED_COST_PROJECTION')
    n=33 if candidate.q==4 else 34
    require(len(witness['tasks'])==n and {t['sample'] for t in witness['tasks']}==set(range(n)), 'REUSE_COHORT')
    for t in witness['tasks']:
        require(set(t)=={'sample','cosine','sine','receipt_sha256','accepted'} and t['accepted'] is True, 'REUSE_TASK_ACCEPTANCE')
        policy.sha(t['receipt_sha256'],'task_receipt')
        for axis in ('cosine','sine'):policy.integer(t[axis],axis,minimum=1)
    return witness['tasks']


def execute(cfg, approval, grant, *, backend_factory=None):
    s=Session(cfg, approval, grant)  # every authorization check precedes science import
    guard=None
    try:
        s.take('bootstrap')
        if backend_factory is None:
            guard=ImportSeal(cfg);guard.install()
            from h2_common_production_backend import Backend
            backend_factory=Backend
        backend=backend_factory(s)
        s.call('setup',backend.setup)
        for prefix in (3,1):s.call('prepare',backend.prepare,prefix)
        diagnostics={};costs={}
        if cfg['branch']=='COMPARE_REUSE_PR':
            for candidate,path in zip(policy.CANDIDATES[2:],cfg['reuse_witnesses']):
                costs[candidate.id]=backend.reuse(candidate,read(path))
        if cfg['branch']!='PR_TWO_NATIVE_CONTROLS':
            for c in policy.CANDIDATES:
                s.verify();diagnostics[c.id]=s.call('diagnose',backend.diagnose,c)
        if cfg['branch']!='DIAGNOSTICS':
            candidates=policy.CANDIDATES[2:] if cfg['branch']=='PR_TWO_NATIVE_CONTROLS' else policy.CANDIDATES[:2]
            for c in candidates:
                s.verify();costs[c.id]=s.call('native_validation',backend.native,c,diagnostics.get(c.id))
        if cfg['branch']=='COMPARE_REUSE_PR':
            accounting=[]
            for c in policy.CANDIDATES:
                allowance=s.call('allowance',backend.allowance,c,diagnostics[c.id])
                accounting.append(s.call('accounting',backend.account,c,diagnostics[c.id],allowance,costs[c.id]))
            result=s.call('ranking',backend.rank,accounting)
        else:result=dict(status='DIAGNOSTICS_ONLY' if cfg['branch']=='DIAGNOSTICS' else 'TWO_PR_CONTROLS_ONLY')
        s.verify();s.loaded_closure()
        require(s.used==cfg['budget'], 'INCOMPLETE_CALL_LEDGER')
        result_files=[dict(path=str(p.relative_to(s.root)),sha256=sha(p))
                      for p in sorted((s.root/'results').iterdir()) if p.is_file()]
        save(s.root/'completion.json',dict(status='CONTRACT_COMPLETE_NOT_SCIENTIFIC_GO',series=SERIES,
             contract_sha256=policy.digest(cfg),used=s.used,result=result,result_files=result_files,
             source_commit=cfg['source_commit'],binding=cfg['binding'],runtime=cfg['runtime'],
             physical=cfg['physical'],native_policy=cfg['native_policy'],
             allowance_policy=cfg['allowance_policy'],reference_policy=cfg['reference_policy'],
             marker_sha256=s.marker_sha,mandatory_STOP=True))
        return result
    except BaseException as exc:
        save(s.root/'STOP.json',dict(status='STOP_PARTIAL_NOT_PASS',reason=str(exc),used=s.used,
              retry=False,mandatory_STOP=True))
        raise
    finally:
        if guard is not None:guard.remove()
