"""Submit existing preparation leaves to the optimizer's shared executor.

The controller retains candidate selection, original frozen inputs, publication
and result adoption. Worker-owned checkouts and immutable artifact/Git bundles
make the next leaf independent of the previous execution host.
"""
import base64
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import tempfile
import threading
import runpy

from .rpe_preparation_artifacts import artifact_path, df_calibration_receipt


_DF_SIGNAL_BOOTSTRAP = """
import json
from pathlib import Path
import runpy
import sys

sys.argv = sys.argv[1:]
runner = Path(sys.argv[0]).resolve()
sys.path.insert(0, str(runner.parents[1]/'src'))
import trottertracks
from trottertracks.rpe_preparation_paths import resolve
payload = Path(sys.argv[sys.argv.index('--input')+1])
view = Path(resolve(json.loads(payload.read_bytes())['runtime']['signal_view']))
# Legacy signal leaves get their fixed provider from the path resolver.
trottertracks.__path__.insert(0, str(view/'src'/'trottertracks'))
runpy.run_path(str(runner), run_name='__main__')
"""

_PF_STEP_INPUTS = """
import json
from pathlib import Path
import sys

root, payload = map(Path, sys.argv[1:])
sys.path.insert(0, str(root/'src'))
from trottertracks import pf_evidence_preparation as E

prepared = json.loads(payload.read_bytes())['value']['prepared']
prefix = prepared['configuration']['spec']['L_D']
inputs = [E.scalar_source(prefix)] if prefix < 10 else []
if prefix in (7, 8, 9):
    inputs.append(E.ref(E.PS.T2_SOURCE, E.PS.T2_DIR+'CELLS.csv'))
elif prefix == 10:
    inputs.extend([E.ref(E.F.FULL, 'docs/research/h4_full_pf_heff_scan_preparation_20261114/PLAN.json'),
                   E.ref(E.F.FULL, E.F.FULLDIR+'C_HANDOFF.json')])
for descriptor in E.saved_step_descriptors(root, prepared):
    inputs.extend(descriptor[name] for name in ('operator_source', 'scope_source', 'summary_source'))
print(json.dumps(inputs))
"""

_PF_NATIVE_GUARD_INPUTS = tuple(
    dict(commit='c31ef424175c02379eafebe8fc6c7548c1d9aabe', path=path) for path in (
        'src/trotterlib/df_trotter/ops.py',
        'src/trottertracks/resource_applicability/ax2b_native_df_v5.py',
        'src/trottertracks/resource_applicability/h6_hash_reuse_v1.py',
        'src/trotterlib/rte_compiled_cost.py'))

# Modules loaded with git show by the pinned native PF compilation provider.
_PF_COMPILE_ADAPTER_INPUTS = tuple(
    dict(commit='748be6875b5007707d864e16aa7a4fc4250f3714', path='src/trottertracks/'+name+'.py')
    for name in ('efficient_controlled_zz', 'p1_efficient_zz', 'compiled_price_fast', 'candidate_price_adapter'))

_PF_PRICE_GUARD_INPUTS = tuple(
    dict(commit=commit, path='src/trottertracks/'+name+'.py') for name, commit in (
        ('h4_df10_rpe_connection', '437ba4acdb82708b6fc98e3cd119a2a734f7cf64'),
        ('pf_pr_rpe_pipeline', '437ba4acdb82708b6fc98e3cd119a2a734f7cf64'),
        ('pr_rpe_rmse_controller', '437ba4acdb82708b6fc98e3cd119a2a734f7cf64'),
        ('full_baseline_v2_requests', 'e7dca5f7301c8633b17a59b6a7161331303aef2d')))

_DF_SPLIT_PRICE_IMPORTS = tuple(
    dict(commit='e827367629ca6c2920d9d908fc1631314399db86',
         path='src/trottertracks/'+name+'.py')
    for name in ('compiled_price_fast', 'candidate_price_adapter', 'ld12_price_requests'))

_DF_SNAPSHOT_INDEX = dict(
    commit='7a8bc7f132e166390c943e6b32c92cc6abe1baf4',
    path='artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json',
)
_DF_SNAPSHOT_LOADERS = (
    dict(commit='ac873e7fcc42e937084fa153476cbd87565702b4',
         path='src/trottertracks/h4_df10_rpe_batch_provider.py'),
    dict(commit='155529e20a8fac39e00786c40b57830b71203c4f',
         path='scripts/h46_m5_tol_saved_input.py'),
)
_PF_SIGNAL_KERNEL_INPUTS = tuple(dict(commit=commit, path=path) for commit, path in (
    ('e417ffc9f52b13c18f5493193ce510d858664259', 'scripts/h46_new2_core.py'),
    ('dd24dd329949c255d5e37b90ff8993fa8df7159b', 'src/trottertracks/pf_power_stability.py'),
    ('256caad3414a80de02fb68eca89c13c957ed6354', 'src/trottertracks/h6_df19_rpe_d2.py'),
    ('71ba49bb28b295aa28e51101bc3b1bf07d500767', 'src/trotterlib/rte.py'),
))
_DF_PRICE_BACKEND = tuple(
    dict(commit='1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7',
         path='src/trottertracks/'+name+'.py')
    for name in ('r_price_phase1', 'r_price_phase1_runner', 'r_price_phase1_science')
) + tuple(
    dict(commit=commit, path=path) for commit, paths in (
        ('ea9e66092465d39dfcd4aeb6fea8e9f09808cabb', ('scripts/run_pr_df_ld_split.py',)),
        ('155529e20a8fac39e00786c40b57830b71203c4f', (
            'scripts/h46_long_time_native.py', 'scripts/h46_m5_tol_saved_input.py',
            'scripts/h46_new2_core.py', 'scripts/h46_new2_native.py',
            'scripts/h46_pr_pilot_repair.py', 'src/trotterlib/h46_m5_plan.py')),
        ('326feadd093920bbea29b498108aa4eae6f67a75', (
            'src/trottertracks/efficient_controlled_zz.py', 'src/trottertracks/p1_efficient_zz.py',
            'scripts/h46_b3_fast_fingerprint.py')),
        ('c31ef424175c02379eafebe8fc6c7548c1d9aabe', (
            'src/trotterlib/df_partial_randomized_pf.py', 'src/trotterlib/df_partial_s2.py',
            'src/trotterlib/df_rte_circuit.py', 'src/trotterlib/df_rte_qiskit.py',
            'src/trotterlib/df_trotter/ops.py', 'src/trotterlib/pr2_matched_accuracy_m1_execution.py',
            'src/trotterlib/rte.py', 'src/trotterlib/rte_compiled_cost.py',
            'src/trottertracks/resource_applicability/ax2b_native_df_v5.py')),
        ('d83737b6d5241962ce39c81d39c56af3506c0cdd', ('src/trottertracks/price_model_validation.py',)),
    ) for path in paths
)

STAGES = {
    'PAULI_SIGNALS': 'signal', 'DF_SIGNALS': 'signal',
    'DF_DESCRIPTOR': 'signal', 'A_BASE_P0': 'signal', 'PF_SIGNALS': 'signal',
    'PAULI_COMPILE': 'compile', 'DF_CALIBRATION': 'compile',
    'DF_HOLDOUT': 'compile', 'PF_COMPILE': 'compile',
    'PF_HOLDOUT_COMPILE': 'compile',
    'PRICE_MODEL_VALIDATION': 'compile',
    'PAULI_AUDIT': 'audit', 'DF_MODEL_FREEZE': 'audit',
    'DF_HOLDOUT_COMPARE': 'audit', 'PF_AUDIT': 'audit',
    'PF_MODEL_FREEZE': 'audit', 'PF_FINISH': 'audit',
    'PF_HOLDOUT_AUDIT': 'audit',
}
DEPENDENCIES = {
    'PAULI_AUDIT': ('PAULI_COMPILE',), 'DF_MODEL_FREEZE': ('DF_CALIBRATION',),
    'DF_HOLDOUT': ('DF_MODEL_FREEZE',), 'DF_HOLDOUT_COMPARE': ('DF_HOLDOUT','DF_MODEL_FREEZE'),
    'PF_AUDIT': ('PF_COMPILE',), 'PF_MODEL_FREEZE': ('PF_AUDIT',),
    'PF_HOLDOUT_COMPILE': ('PF_MODEL_FREEZE',), 'PF_HOLDOUT_AUDIT': ('PF_HOLDOUT_COMPILE',),
    'PF_FINISH': ('PF_SIGNALS','PF_AUDIT','PF_HOLDOUT_AUDIT'),
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def df_freeze_output_problem(request, result):
    """A clean provider exit must return the model and prices for this leaf."""
    prep = request['preparation']
    if prep.get('provider') != 'DF_MODEL_FREEZE' or prep.get('artificial'):
        return None
    try:
        command = prep['command']
        payload = Path(command[command.index('--input') + 1])
        leaf = artifact_path(payload.relative_to(prep['controller_root']).parent.as_posix())
    except (KeyError, IndexError, TypeError, ValueError):
        return 'DF_FREEZE_INPUT_LOCATION_UNAVAILABLE'
    artifacts = result.get('artifacts') or result.get('artifact_manifest') or {}
    inputs = prep['inputs']
    missing = []
    for filename in ('FROZEN_MODELS.json', 'AXIS_PRICES.json'):
        name = (Path('workspace') / leaf / filename).as_posix()
        if name not in artifacts or artifacts[name] == inputs.get(name):
            missing.append(filename)
    if missing:
        return 'DF_FREEZE_REQUIRED_OUTPUT_NOT_RETURNED:' + ','.join(missing)


def contract_runtime(values, script, configured, provider=None):
    """Extract declared package/compiler identities without importing a provider."""
    result=deepcopy(configured);packages=result.setdefault('packages',{});compilers=set()
    if provider in ('DF_DESCRIPTOR', 'DF_SIGNALS'):
        # The fixed saved-context loader imports trotterlib's public package,
        # including plotting and chemistry imports. Probe these before a leaf
        # is assigned; their availability was absent from the numeric pins.
        result['imports'] = sorted(set(result.get('imports', ())) |
                                   {'matplotlib.pyplot', 'openfermionpyscf'})
    def require(name,value):
        if name in packages and packages[name]not in(None,value):raise ValueError('Preparation runtime version conflicts with original input: '+name)
        packages[name]=value
    def visit(value):
        if isinstance(value,dict):
            for name,item in value.items():
                if name in('qiskit_version','numpy_version','scipy_version')and isinstance(item,str):require(name[:-8],item)
                if name=='compiler'and isinstance(item,str)and len(item)==64:compilers.add(item)
                visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
    for value in values:visit(value)
    # DF's existing numerical worker spells its fixed NumPy/SciPy guard as
    # a tuple comparison. Read that fixed syntax instead of importing NumPy.
    tree=ast.parse(script.read_text())
    for node in ast.walk(tree):
        if not isinstance(node,ast.Compare)or len(node.comparators)!=1:continue
        left,right=node.left,node.comparators[0]
        if not isinstance(left,ast.Tuple)or not isinstance(right,ast.Tuple)or len(left.elts)!=len(right.elts):continue
        for name,version in zip(left.elts,right.elts):
            if(isinstance(name,ast.Attribute)and name.attr=='__version__'and isinstance(name.value,ast.Name)
               and name.value.id in('numpy','scipy','qiskit')and isinstance(version,ast.Constant)and isinstance(version.value,str)):
                require(name.value.id,version.value)
    return result,sorted(compilers)


def fixed_input_hashes(root, documents, artifacts, *, df_raw_commit=None):
    """Hash every fixed Git locator in original JSON, including nested ones."""
    from .rpe_q3_coordinator import input_hashes
    working=[]
    git_blobs = {}

    def df_receipt(path):
        # A copied PATH_INDEX can still name the original owner's artifact.
        # The fixed DF audit resolves WORKING at --raw-commit. Ship that exact
        # original blob only when the collected copy matches it byte for byte.
        if not df_raw_commit or not path.startswith('search-evidence/'):
            return None
        commit, relocated = df_raw_commit
        parts = path.split('/')
        if (len(parts) < 5 or parts[2] != 'completion' or
                parts[-1] not in ('cosine.json', 'sine.json', 'PATH_RECEIPT.json')):
            return None
        suffix = '/'.join(parts[2:])
        matches = [sha for name, sha in artifacts.items()
                   if name.startswith(relocated) and name.endswith('/' + suffix)]
        if len(matches) != 1:
            return None
        if path not in git_blobs:
            blob = subprocess.run(['git', 'show', commit + ':' + path],
                                  cwd=root, capture_output=True)
            git_blobs[path] = hashlib.sha256(blob.stdout).hexdigest() if blob.returncode == 0 else None
        return commit if git_blobs[path] == matches[0] else None
    # PF's original audit binds these provisional sidecars to its collected
    # raw publication. Hash the same evidence without editing the sidecars.
    compiled={}
    for value in documents:
        if (isinstance(value,dict) and value.get('status')=='COMPILE_RECEIPTS_READY' and
                isinstance(value.get('relative'),str) and isinstance(value.get('raw_commit'),str)):
            path=value['relative']+'/IMPLEMENTATION_EVIDENCE.json'
            previous=compiled.setdefault(path,value['raw_commit'])
            if previous!=value['raw_commit']:raise ValueError('Conflicting PF raw publications: '+path)
    def fixed(value):
        if isinstance(value,list):return[fixed(item)for item in value]
        if not isinstance(value,dict):return value
        result={name:fixed(item)for name,item in value.items()}
        if value.get('commit')=='WORKING'and isinstance(value.get('path'),str):
            # Existing providers use WORKING for a raw artifact already in the
            # owned output tree. Bind that artifact, never pretend it is Git.
            path = value['path']
            matches=[(name,sha)for name,sha in artifacts.items()if name.endswith('/'+path)]
            if not matches and df_raw_commit:
                commit = df_receipt(path)
                if commit:
                    result['commit'] = commit
                    return result
            if len(matches)!=1:raise ValueError('Unfrozen or ambiguous WORKING input: '+value['path'])
            working.append(dict(path=matches[0][0],sha256=matches[0][1]));result.pop('commit')
        if value.get('commit')=='FILL_WITH_PUBLISHED_COMMIT' and value.get('path') in compiled:
            result['commit']=compiled[value['path']]
        return result
    payload={'route':{'kind':'preparation_input','documents':fixed(documents)}}
    return input_hashes(root,payload),list({row['path']:row for row in working}.values())


def operational_source_requirements(documents):
    """Declare original external runtime views for profile-based host resolution."""
    found={}
    paths={
        'signal_view':('src/trottertracks/pr_rpe_signal_provider.py','src/trottertracks/pr_rpe_signal_plan.py','src/trottertracks/pr_rpe_rmse_s0_v2.py'),
        'd2_view':('src/trottertracks/pr_price_rules.py',),
        'df_price_view':('scripts/run_pr_df_ld7_v0.py','src/trottertracks/pr_df_signal_entry.py'),
    }
    def visit(value):
        if isinstance(value,list):
            for item in value:visit(item)
        elif isinstance(value,dict):
            for name,item in value.items():
                if name in paths and isinstance(item,str):
                    root=Path(item).resolve()
                    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
                    files={}
                    for relative in paths[name]:
                        raw=subprocess.check_output(['git','show',commit+':'+relative],cwd=root)
                        if(root/relative).read_bytes()!=raw:raise ValueError('Original operational SOURCE view changed: '+str(root/relative))
                        files[relative]=hashlib.sha256(raw).hexdigest()
                    found[str(root)]=dict(path=str(root),field=name,source=dict(commit=commit,files=files))
                visit(item)
    visit(documents)
    return list(found.values())


def collect_fixed_inputs(root, documents, artifacts=None):
    return fixed_input_hashes(root,documents,artifacts or{})


def used_path_aliases(aliases, bindings, documents, command):
    """Carry aliases referenced by this leaf, not every past worker checkout."""
    text = json.dumps([*documents, command], ensure_ascii=False)
    return {old: bindings[original] for old, original in aliases.items()
            if original in bindings and json.dumps(old, ensure_ascii=False)[:-1] in text}


def collect_required_views(documents):
    rows=operational_source_requirements(documents)
    return {'runtime_'+row['field']+'_'+row['source']['commit'][:10]:
            dict(source=row['source'],controller_path=row['path'])for row in rows}


def pauli_calibration_view(repository):
    from .rpe_preparation_paths import (
        PAULI_CALIBRATION_FILES, PAULI_CALIBRATION_SOURCE, PAULI_CALIBRATION_VIEW,
    )
    files = {}
    for relative in PAULI_CALIBRATION_FILES:
        raw = subprocess.check_output(
            ['git', 'show', PAULI_CALIBRATION_SOURCE + ':' + relative], cwd=repository)
        files[relative] = hashlib.sha256(raw).hexdigest()
    return {PAULI_CALIBRATION_VIEW: dict(
        source=dict(commit=PAULI_CALIBRATION_SOURCE, files=files))}


def pauli_saved_kernel_inputs(root, packets):
    """Bind the Git blobs read by the fixed saved-signal fingerprint check."""
    from .pauli_signal_entry import KERNEL_FILES, LOCAL, S3, read_ref

    commits = {S3.P1_KERNEL}
    for ref in packets:
        for packet in read_ref(root, ref)['packets']:
            for round_value in packet['rounds']:
                record = round_value.get('mean_record') or {}
                if (round_value.get('mean_source') and record.get('status') == LOCAL
                        and record.get('numerical', {}).get('gate_pass') is True):
                    commits.add(record['source']['commit'])
    return [dict(commit=commit, path=path) for commit in sorted(commits) for path in KERNEL_FILES]


def pauli_price_guard_inputs(request):
    # The fixed adapter checks these Git files in its owned workspace before
    # any compile/audit body. Include them even when absent from the JSON.
    groups = {
        '02e7cc6a127954fa7c446e7383d037f0617d50d3': (
            'src/trottertracks/pauli_calibration_request.py',
            'scripts/prepare_pauli_calibration.py', 'scripts/run_pauli_split_cal.py'),
        '54b4353b9826a00b1e6226290bf92db2b15ef9f6': (
            'scripts/pauli_ld2_step_audit.py', 'scripts/pauli_ld3_cal_audit.py'),
    }
    paths = [dict(commit=commit, path=path) for commit, names in groups.items() for path in names]
    commit = request.get('execution_source') if request.get('scoped_registration') else next(reversed(groups))
    if commit:
        paths.append(dict(commit=commit, path='scripts/pauli_split_cal_audit_request.py'))
    return paths


def pauli_audit_raw_inputs(name, state):
    """Declare sibling-branch artifacts consumed through the audit's Git reads."""
    prefix = Path('workspaces/provider')
    path = Path(artifact_path(name))
    if (not path.is_relative_to(prefix) or path.name != 'PRICE_STATE.json'
            or state.get('status') != 'COMPILE_READY_FOR_AUDIT'):
        return []
    directory = path.parent.relative_to(prefix)
    base = Path(artifact_path(state['driver_spec']['base']))
    if base != directory/'calibration':
        raise ValueError('Pauli audit raw source is outside its prepared group')
    return [dict(commit=state['raw_commit'], path=(base/relative).as_posix())
            for relative in ('BINDING.json', 'run01/COMPILE_RECEIPTS.json', 'run01/RESULT.json')
            ] + [dict(commit=state['inputs_commit'], path=(directory/'CONFIGURATION.json').as_posix())]


def collect_required_paths(documents, controller_paths=None):
    """Reuse the existing native runtime/config contract, without its approval flow.

    Only runtime.provider_root and runtime.input_config are interpreted. The
    provider manifest comes from that original config's provider/runtime seals;
    optional caches and historical permission/ledger fields are not imported.
    """
    found={}
    controller_paths=controller_paths or {}
    def visit(value):
        if isinstance(value,list):
            for item in value:visit(item)
        elif isinstance(value,dict):
            runtime=value.get('runtime')
            if isinstance(runtime,dict)and(runtime.get('provider_root')or runtime.get('input_config')):
                config_path=runtime.get('input_config')
                if not isinstance(config_path,str):raise ValueError('Native provider runtime needs its original input_config')
                original_config=Path(config_path)
                configured=controller_paths.get(str(original_config),{})
                config_path=Path(configured.get('path',original_config)).resolve()
                if not config_path.is_file():raise ValueError('Native runtime input_config missing: '+str(config_path))
                raw=config_path.read_bytes();sha=hashlib.sha256(raw).hexdigest();config=json.loads(raw)
                if configured and configured.get('sha256')!=sha:
                    raise ValueError('Native runtime input_config differs from its fixed host mapping')
                found['runtime_input_config_'+sha[:12]]=dict(controller_path=str(original_config),kind='file',sha256=sha)
                provider=runtime.get('provider_root')
                if provider:
                    original_provider=Path(provider)
                    mapped=controller_paths.get(str(original_provider),{})
                    provider=Path(mapped.get('path',original_provider)).resolve()
                    if not provider.is_dir():raise ValueError('Native provider_root missing: '+str(provider))
                    files={}
                    for row in config.get('sealed_files',[]):
                        if row.get('role')!='provider/runtime':continue
                        original=Path(row['path'])
                        if not original.is_relative_to(original_provider):continue
                        relative=original.relative_to(original_provider).as_posix();artifact_path(relative)
                        path=provider/relative
                        expected=row.get('sha256')
                        if not isinstance(expected,str)or len(expected)!=64:raise ValueError('Original provider seal lacks hash: '+str(path))
                        if not path.is_file()or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                            raise ValueError('Original sealed provider/runtime file missing or changed: '+str(path))
                        files[relative]=expected
                    if not files:raise ValueError('Native provider_root has no original provider/runtime seals: '+str(provider))
                    found['runtime_provider_root_'+digest(files)[:12]]=dict(controller_path=str(original_provider),kind='directory',files=files)
            for item in value.values():visit(item)
    visit(documents)
    return found


def build_preparation_git_bundle(root, ref, prerequisites, locators, path, workspace_objects=()):
    """Write the same Git bundle format used by the preparation transport."""
    revisions = [ref, *('^' + commit for commit in prerequisites)]
    # Git's bundle command emits thin packs. Keep the increment, but
    # require a pack whose deltas never depend on omitted Git objects.
    subprocess.run(['git', '-c', 'pack.window=0', '-c', 'pack.reuseDeltas=false',
                    '-c', 'pack.reuseObjects=false', 'bundle', 'create', str(path), *revisions], cwd=root,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    # Give pack-objects path names so it can compress similar objects
    # within this pack. Use an explicit object set and no external deltas.
    changed = subprocess.check_output(['git', 'rev-list', '--objects', *revisions], cwd=root)
    entries = dict(line.split(b' ', 1) if b' ' in line else (line, b'')
                   for line in changed.split(b'\n') if line)
    for oid in workspace_objects:
        entries.setdefault(oid, b'')
    # Sparse worker views can lack blobs and trees from earlier commits
    # even when Git already has their commit objects.
    expressions = {}
    for commit, path_name in locators:
        parts = artifact_path(path_name).parts
        expressions.update({commit: '', commit + '^{tree}': ''})
        for depth in range(1, len(parts) + 1):
            name = '/'.join(parts[:depth])
            expressions[commit + ':' + name] = name
    if expressions:
        objects = subprocess.check_output(['git', 'rev-parse', *expressions], cwd=root).splitlines()
        for oid, name in zip(objects, expressions.values(), strict=True):
            entries.setdefault(oid, name.encode())
    with path.open('rb') as original:
        header = bytearray()
        for line in original:
            header.extend(line)
            if line == b'\n':
                break
    data = b''.join(oid + (b' ' + name if name else b'') + b'\n' for oid, name in sorted(entries.items()))
    with path.open('wb') as packed:
        packed.write(header)
        packed.flush()
        # Threaded delta search can produce different pack bytes from
        # the same object list and invalidate worker cache reuse.
        subprocess.run(
            ['git', 'pack-objects', '--stdout', '--threads=1', '--no-reuse-delta', '--no-reuse-object'],
            cwd=root, input=data, stdout=packed, check=True, stderr=subprocess.PIPE,
        )


class DistributedPreparationBridge:
    def __init__(self, executor, root, settings=None, *, approved=False):
        self.executor = executor
        self.root = Path(root).resolve()
        self.settings = deepcopy(settings or {})
        self.approved = approved is True
        self.alias_path=Path(executor.run_dir)/'preparation'/'PATH_ALIASES.json'
        self.aliases=read(self.alias_path)if self.alias_path.exists()else{}
        self.lock=threading.RLock()
        self.bundle_hashes = {}
        self.saved_bundle_hashes = None
        self.bundle_locks = {}
        self.shared_bundle_heads = {}
        self.compact_bindings = {}
        self.stage_return_locks={}
        self.accepted_revision=0

    def recover_accepted_stages(self, completion, *, stop_requested=lambda:False, failures=None):
        """Import exact accepted owners without waiting for candidate traversal.

        Only a stage marker in this run can authorize importing its original
        successful receipt. No request is submitted here.
        """
        import time
        failures = {} if failures is None else failures
        from .pr_candidate_completion import Held, PreparationWait
        imported=[]
        root=Path(self.executor.run_dir).resolve()
        for job in self.executor.store.unadopted_preparation_successes():
            if stop_requested():break
            if time.monotonic()<failures.get(job,0):continue
            row=self.executor.store.row(job);prep=(row or {}).get('request',{}).get('preparation',{})
            stage=prep.get('provider');base=Path(prep.get('controller_root',self.root)).resolve()
            if stage not in STAGES:continue
            markers=[]
            for output in prep.get('outputs',()):
                if not output.startswith('workspace/'):continue
                directory=(base/artifact_path(output.removeprefix('workspace/'))).resolve()
                if not directory.is_relative_to(root):continue
                marker=directory/(stage+'.DISTRIBUTED.json')
                try:
                    if marker.is_file() and read(marker).get('job_id')==job:markers.append(marker)
                except (OSError,ValueError):continue
            if len(set(markers))!=1:
                failures[job]=time.monotonic()+60;continue
            try:
                if self.import_received_stage(completion,markers[0],stage):
                    imported.append(job);failures.pop(job,None)
                    self.accepted_revision+=1
                    self.executor.log(event='ACCEPTED_PREPARATION_IMPORTED',job=job,provider=stage)
                    with self.executor.state.cv:self.executor.state.cv.notify_all()
                else:failures[job]=time.monotonic()+30
            except (Held,PreparationWait,ValueError,KeyError,OSError,subprocess.CalledProcessError) as error:
                failures[job]=time.monotonic()+60
                self.executor.log(event='ACCEPTED_PREPARATION_IMPORT_PENDING',job=job,provider=stage,reason=str(error)[:1000])
        return imported

    def start_accepted_recovery(self,completion):
        """One independent receiver, retained only for the active planning cycle."""
        stop=threading.Event();failures={}
        def run():
            while not stop.is_set() and not self.executor.state.closed:
                try:self.recover_accepted_stages(completion,stop_requested=stop.is_set,failures=failures)
                except (sqlite3.Error,OSError,ValueError) as error:
                    self.executor.log(event='ACCEPTED_PREPARATION_RECOVERY_PENDING',reason=str(error))
                with self.executor.state.cv:
                    if not stop.is_set():self.executor.state.cv.wait(2)
        thread=threading.Thread(target=run,name='rpe-accepted-preparation',daemon=True);thread.start()
        def close():
            stop.set()
            with self.executor.state.cv:self.executor.state.cv.notify_all()
            thread.join()
        return close

    @staticmethod
    def handles(stage):
        return stage in STAGES

    def follow_preparations(self, items, advance, stop_requested):
        """Resume a candidate on this producer thread after its job changes."""
        remaining = iter(items)
        waiting = {}
        exhausted = False
        while not stop_requested():
            ready = self.executor.ready_preparations(waiting, wait=exhausted, stop_requested=stop_requested)
            if stop_requested() or self.executor.state.failure is not None:
                break
            if ready:
                job_id = ready[0]
                candidates = waiting[job_id]
                item = candidates.pop(0)
                if not candidates:
                    del waiting[job_id]
            elif not exhausted:
                try:
                    item = next(remaining)
                except StopIteration:
                    exhausted = True
                    continue
            else:
                break
            if job_id := advance(item):
                waiting.setdefault(job_id, []).append(item)

    @staticmethod
    def _dependencies(directory, stage):
        jobs = []
        for name in DEPENDENCIES.get(stage, ()):
            marker = directory/(name+'.DISTRIBUTED.json')
            recovered = directory/(name+'.RECOVERED_AUDIT.json')
            if recovered.is_file():
                jobs.append(read(recovered)['compile_job'])
            elif marker.is_file():
                jobs.append(read(marker)['job_id'])
        return list(dict.fromkeys(jobs))

    def recovered_audit(self, directory, stage, stage_input, compiled):
        """Consume C-checked raw compiler output while preserving the failed audit."""
        from .rpe_q3_coordinator import digest
        from .rpe_release_registry import write
        from .pr_candidate_completion import Held

        marker = directory/(stage+'.DISTRIBUTED.json')
        if stage not in ('PF_AUDIT', 'PF_HOLDOUT_AUDIT') or not marker.is_file():
            return None
        job_id = read(marker)['job_id']
        recovery = self.settings.get('pf_stage_audit_recoveries', {}).get(job_id)
        if not recovery:
            return None
        original = self.recovery_evidence(directory, stage, job_id)
        owner = self.executor.store.row(job_id)
        compile_job = recovery['compile_job']
        compile_stage = 'PF_COMPILE' if stage == 'PF_AUDIT' else 'PF_HOLDOUT_COMPILE'
        producer = self.recovery_evidence(directory, compile_stage, compile_job)
        if (original.get('attempt_id') != recovery['audit_attempt']
                or original['action'] not in ('PRESERVE_FAILED_ATTEMPT', 'RECOVER_ORIGINAL_ATTEMPT')
                or (original['state'] == 'UNKNOWN' and
                    (owner['started_epoch'] is not None or owner['detail'].get('resource_released') is not True))
                or owner['request']['preparation']['dependencies'] != [compile_job]
                or digest(stage_input) != recovery['input_sha256']
                or compiled != recovery['compiled']
                or producer['action'] != 'IMPORT_SUCCESSFUL_STAGE'
                or producer['attempt_id'] != recovery['compile_attempt']):
            raise Held('PREPARATION_RESULT_MISMATCH:'+stage+':'+job_id+':AUDIT_RECOVERY')
        result = recovery['result']
        # C verified all returned raw axes; retain the original raw path
        # and receipt lineage when the successor bundles these prices.
        for price in result['prices']:
            loc = price['price_evidence']['source']
            if loc['path'] != compiled['relative']+'/PRICES.json':
                raise Held('PREPARATION_RESULT_MISMATCH:'+stage+':'+job_id+':RAW_PRICE_LOCATOR')
        write(directory/(stage+'.RECOVERED_AUDIT.json'), recovery)
        return result

    def repair_compile_context(self, directory, stage, submitted):
        """Use the selected transport revision after a proved pre-wrapper failure."""
        from .pr_candidate_completion import Held
        from .rpe_release_registry import write

        job_id = submitted.get('job_id')
        repair = self.settings.get('pf_stage_audit_recoveries', {}).get(job_id)
        if not repair or repair.get('kind') != 'PF_KERNEL_CONTEXT_REPAIR':
            return submitted
        original = self.recovery_evidence(directory, stage, job_id)
        owner = self.executor.store.row(job_id)
        path = 'private/pf-transport/scripts/run_pf_preparation_stage.py'
        if (stage not in ('PF_COMPILE', 'PF_HOLDOUT_COMPILE')
                or original['action'] != 'PRESERVE_FAILED_ATTEMPT'
                or original.get('attempt_id') != repair['audit_attempt']
                or owner['request_sha256'] != repair['request_sha256']
                or submitted['request']['preparation']['inputs'].get(path) != repair['original_adapter_sha256']):
            raise Held('PREPARATION_RESULT_MISMATCH:'+stage+':'+job_id+':CONTEXT_REPAIR')
        body = (Path(__file__).resolve().parents[2]/'scripts/run_pf_preparation_stage.py').read_bytes()
        if hashlib.sha256(body).hexdigest() != repair['replacement_adapter_sha256']:
            raise Held('PREPARATION_RESULT_MISMATCH:'+stage+':'+job_id+':REPAIR_ADAPTER')
        request = deepcopy(submitted['request'])
        request.pop('request_id', None)
        preparation = request['preparation']
        preparation['inputs'][path] = self._blob(body)
        pins = [record for record in preparation['working_input_hashes'] if record['path'] == path]
        if len(pins) != 1 or pins[0]['sha256'] != repair['original_adapter_sha256']:
            raise Held('PREPARATION_RESULT_MISMATCH:'+stage+':'+job_id+':ORIGINAL_ADAPTER')
        pins[0]['sha256'] = repair['replacement_adapter_sha256']
        request['preparation']['supersedes_preparation'] = dict(
            job_id=job_id, attempt_id=repair['audit_attempt'], reason='FIXED_NUMERICAL_KERNEL_BEFORE_WRAPPER')
        write(directory/(stage+'.ORIGINAL_CONTEXT_ATTEMPT.json'), submitted)
        return dict(request=request, source=submitted['source'], automatic_retry=False)

    def redeliver_pf_compile(self, completion, command, directory, stage, output, submitted):
        """Rebuild transport for a selected failed PF job, keeping native inputs."""
        from .pr_candidate_completion import Held

        job = submitted.get('job_id')
        recovery = self.settings.get('pf_stage_audit_recoveries', {}).get(job, {})
        if recovery.get('kind') != 'PF_PORTABLE_INPUT_DELIVERY':
            recovery = self.pf_delivery_failure(directory, stage, job)
            if recovery is None:
                return None
        proof = self.recovery_evidence(directory, stage, job)
        owner = self.executor.store.row(job)
        stage_input = directory/(stage+'.INPUT.json')
        raw = stage_input.read_bytes()
        if (stage not in ('PF_COMPILE', 'PF_HOLDOUT_COMPILE')
                or proof['action'] != 'PRESERVE_FAILED_ATTEMPT'
                or proof['attempt_id'] != recovery['audit_attempt']
                or owner['request_sha256'] != recovery['request_sha256']
                or hashlib.sha256(raw).hexdigest() != recovery['input_sha256']
                or json.loads(raw)['value']['configuration']['id'] != recovery['configuration_id']
                or (directory/'STOP.json').exists()):
            raise Held('PREPARATION_RESULT_MISMATCH:'+stage+':'+job+':PF_INPUT_REDELIVERY')
        original = owner['request']['preparation']
        request = self.build_request(completion, command, directory, stage, output)
        replacement = request['preparation']
        if (original['source']['commit'] != replacement['source']['commit']
                or any(replacement['source']['files'].get(name) != sha
                       for name, sha in original['source']['files'].items())
                or owner['request']['original_contract'] != request['original_contract']
                or 'private/pf-transport/scripts/run_pf_preparation_stage.py' not in replacement['inputs']):
            raise Held('PREPARATION_RESULT_MISMATCH:'+stage+':'+job+':PF_SCIENTIFIC_SOURCE')
        if recovery.get('automatic') and all(
                original.get(field) == replacement.get(field)
                for field in ('inputs', 'input_git_bundles', 'path_bindings', 'required_paths', 'workspaces')):
            return None  # No workspace repair was found.
        replacement['supersedes_preparation'] = dict(job_id=job, attempt_id=proof['attempt_id'],
            reason=recovery.get('reason', 'PORTABLE_FIXED_INPUT_DELIVERY'))
        return dict(request=request, source=submitted['source'], automatic_retry=False)

    def pf_delivery_failure(self, directory, stage, job):
        """Recognize a proved pre-provider Git failure from an older workspace."""
        if (stage not in ('PF_COMPILE', 'PF_HOLDOUT_COMPILE') or not self.approved
                or (directory/'STOP.json').exists() or self.executor._user_stop_requested()
                or job in self.settings.get('failed_preparation_recalculations', {})):
            return None
        proof = self.recovery_evidence(directory, stage, job)
        if proof['action'] != 'PRESERVE_FAILED_ATTEMPT':
            return None
        owner = self.executor.store.row(job)
        preparation = owner['request']['preparation']
        if ('private/pf-transport/scripts/run_pf_preparation_stage.py' in preparation['inputs']
                and any(bundle.get('workspace_objects') for bundle in preparation.get('input_git_bundles', ()))):
            return None  # Already using the complete portable delivery path.
        job_dir = self.executor.run_dir/'jobs'/owner['method']/job
        try:
            launch = read(job_dir/'LAUNCH.json')
            failure = read(job_dir/'OUTCOME.json').get('workspace_error') or {}
            stage_input = directory/(stage+'.INPUT.json')
            raw = stage_input.read_bytes()
            configuration_id = json.loads(raw)['value']['configuration']['id']
        except (OSError, ValueError, TypeError, KeyError):
            return None
        if (preparation.get('supersedes_preparation') or preparation.get('artificial')
                or (launch['host_id'], launch['attempt_id']) != (owner['host_id'], proof['attempt_id'])
                or failure.get('step') not in ('git sparse-checkout add', 'git owned workspace')
                or failure.get('scientific_body_started') is not False
                or type(failure.get('returncode')) is not int or failure['returncode'] == 0):
            return None
        sha = hashlib.sha256(raw).hexdigest()
        if {digest for name, digest in preparation['inputs'].items()
                if name.endswith('/'+stage_input.name)} != {sha}:
            return None
        return dict(audit_attempt=proof['attempt_id'], request_sha256=owner['request_sha256'],
                    input_sha256=sha, configuration_id=configuration_id,
                    reason='FIXED_GIT_WORKSPACE_PRELAUNCH_FAILURE', automatic=True)

    def recovered_holdout_prices(self, directory, stage_input):
        """Rebind C-verified wrapper results produced before a packaging failure."""
        from .pr_candidate_completion import Held
        from .rpe_q3_coordinator import digest
        from .rpe_release_registry import write

        stage = 'PF_HOLDOUT_COMPILE'
        marker = directory/(stage+'.DISTRIBUTED.json')
        if not marker.is_file():
            return None
        job_id = read(marker)['job_id']
        recovery = self.settings.get('pf_stage_audit_recoveries', {}).get(job_id)
        if not recovery or recovery.get('kind') != 'VERIFIED_ORIGINAL_HOLDOUT_PRICES':
            return None
        original = self.recovery_evidence(directory, stage, job_id)
        owner = self.executor.store.row(job_id)
        if (original.get('attempt_id') != recovery['audit_attempt']
                or original['action'] != 'PRESERVE_FAILED_ATTEMPT'
                or owner['request_sha256'] != recovery['request_sha256']
                or digest(stage_input) != recovery['input_sha256']
                or (directory/'PF_HOLDOUT_AUDIT.DISTRIBUTED.json').exists()):
            raise Held('PREPARATION_RESULT_MISMATCH:'+stage+':'+job_id+':HOLDOUT_RECOVERY')
        value = stage_input['value']
        wanted = {(row['m'], axis, row['physical_candidate']['id'])
                  for row in value['prepared']['pending']['requests'] if row['m'] in stage_input['rounds']
                  for axis in stage_input.get('axes_by_round', {}).get(str(row['m']), ('X', 'Y'))}
        prices = recovery['result']['prices']
        if (value['configuration']['id'] != recovery['configuration_id']
                or len(prices) != len(wanted)
                or {(row['m'], row['axis'], row['physical_id']) for row in prices} != wanted
                or any(row['gate'] != 'C_IMPLEMENTATION_GUARD_PASS' for row in prices)):
            raise Held('PREPARATION_RESULT_MISMATCH:'+stage+':'+job_id+':RAW_PRICE_AXES')
        write(directory/(stage+'.RECOVERED_PRICES.json'), recovery)
        return prices

    def recovery_evidence(self, directory, stage, job_id):
        """Classify an exact owned attempt without starting or mutating it."""
        from .rpe_preparation_queue import normalize_preparation_request
        from .rpe_q3_coordinator import digest
        unmatched = dict(action='CHECK_ORIGINAL_REQUEST', job_id=job_id)
        if stage not in STAGES or not isinstance(job_id, str) or re.fullmatch(r'prep-[0-9a-f]{24}', job_id) is None:
            return unmatched
        try:
            row = self.executor.store.row(job_id)
            if not row:
                return unmatched
            matching = []
            for path in directory.rglob(stage+'.DISTRIBUTED.json'):
                marker = read(path)
                if marker.get('job_id') == job_id:
                    matching.append((path, marker))
            if len(matching) != 1:
                return unmatched
            path, marker = matching[0]
            request = normalize_preparation_request(marker['request'], dependencies=self._dependencies(path.parent, stage))
            if (request['request_id'] != job_id or request != row['request']
                    or digest(request) != row['request_sha256']
                    or request['preparation']['provider'] != stage
                    or request['preparation']['source']['commit'] != marker['source']):
                return unmatched
            state = row['state']
            status = row['detail'].get('outcome_status')
            result = dict(job_id=job_id, attempt_id=row['attempt_id'], state=state, outcome_status=status)
            if state == 'ACCEPTED' and row['attempt_id'] and row['completion_sha256']:
                # Older queue versions may omit the outcome status in detail.
                # The fixed accepted outcome and process return still identify
                # a completed stage. Do not turn that receipt into a retry.
                if status is None:
                    job = self.executor.run_dir/'jobs'/row['method']/job_id
                    outcome = read(job/'OUTCOME.json')
                    receipt = read(job/'PROCESS_RETURN.json')
                    if (outcome.get('request_sha256') != row['request_sha256']
                            or outcome.get('request_id') != job_id
                            or (outcome.get('status') == 'PREPARATION_COMPLETE'
                                and (receipt.get('returncode') != 0 or receipt.get('signal') is not None))):
                        return unmatched
                    status = outcome.get('status')
                    result['outcome_status'] = status
                action = 'CHECK_ORIGINAL_RECEIPT'
                if status == 'PREPARATION_COMPLETE':
                    if stage == 'DF_MODEL_FREEZE' and not request['preparation'].get('artificial'):
                        job = self.executor.run_dir/'jobs'/row['method']/job_id
                        outcome = read(job/'OUTCOME.json')
                        if (outcome.get('request_sha256') != row['request_sha256']
                                or outcome.get('status') != status):
                            return unmatched
                        if problem := df_freeze_output_problem(request, outcome):
                            return dict(result, action='CHECK_ORIGINAL_OUTPUT', reason=problem)
                    action = 'IMPORT_SUCCESSFUL_STAGE'
                elif status == 'PREPARATION_FAILED':
                    job = self.executor.run_dir/'jobs'/row['method']/job_id
                    outcome = read(job/'OUTCOME.json')
                    receipt = read(job/'PROCESS_RETURN.json')
                    artifacts = outcome.get('artifacts') or {}
                    if (outcome.get('request_sha256') != row['request_sha256']
                            or outcome.get('status') != status):
                        return unmatched
                    result.update(returncode=receipt.get('returncode'), signal=receipt.get('signal'),
                                  partial_artifacts=len(artifacts))
                    if (workspace_error := outcome.get('workspace_error')) and workspace_error.get('scientific_body_started') is False:
                        result['pre_science_failure'] = workspace_error
                    if isinstance(outcome.get('failure_summary'), str):
                        result['failure_summary'] = outcome['failure_summary']
                    if isinstance(outcome.get('failure_diagnostic'), dict):
                        result['failure_diagnostic'] = outcome['failure_diagnostic']
                    if stage == 'DF_SIGNALS' and artifacts:
                        numerical = self.df_signal_failure_rounds(request['preparation']['outputs'], artifacts)
                        if numerical:
                            result['failed_signal_rounds'] = numerical
                    action = ('CHECK_FAILED_ATTEMPT_ARTIFACTS' if artifacts else 'PRESERVE_FAILED_ATTEMPT')
            elif row['attempt_id']:
                action = 'RECOVER_ORIGINAL_ATTEMPT'
            else:
                action = 'RECHECK_RESERVATION'
            return dict(result, action=action)
        except (OSError, KeyError, TypeError, ValueError, sqlite3.Error):
            return unmatched

    def df_signal_failure_rounds(self, outputs, artifacts):
        failures = []
        cache = self.executor.artifact_cache
        for output in outputs:
            raw = output+'/df_signals/raw/'
            for name, sha in sorted(artifacts.items()):
                if not name.startswith(raw) or not re.fullmatch(r'm\d+\.json', name[len(raw):]):
                    continue
                if cache.missing([sha]):
                    continue
                record = read(cache.path(sha))
                if record.get('status') == 'NUMERICAL_FAILURE':
                    failures.append(dict(m=record['m'], artifact=name, sha256=sha,
                                         error=record.get('error'), reasons=record.get('numerical_failure_reasons', [])))
        return sorted(failures, key=lambda item: item['m'])

    def _exec_rejected(self, row):
        """Require positive evidence that the kernel never started the provider."""
        try:
            job = self.executor.run_dir/'jobs'/row['method']/row['job_id']
            if row['state'] == 'ACCEPTED':
                outcome = read(job/'OUTCOME.json')
                return (outcome['status'] == 'PREPARATION_FAILED' and not outcome['artifacts']
                        and outcome['request_sha256'] == row['request_sha256']
                        and outcome['workspace_error'] == dict(step='exec', error='E2BIG',
                                                                 scientific_body_started=False))
            # Older local workers could exit before returning a structured
            # outcome. Accept only the original subprocess exec rejection.
            local = self.executor.distributed.get('local_worker') or {}
            if (row['state'] != 'UNKNOWN' or row['detail'].get('resource_released') is not True
                    or row['host_id'] != local.get('host_id')):
                return False
            spool = Path(local['work_dir'])/row['job_id']/row['attempt_id']
            launch = read(job/'LAUNCH.json')
            receipt = read(spool/'PROCESS_RETURN.json')
            if ((launch['host_id'], launch['attempt_id']) != (row['host_id'], row['attempt_id'])
                    or read(spool/'ORIGINAL_REQUEST.json') != row['request']
                    or read(spool/'PREPARATION_STARTED.json')['request_sha256'] != row['request_sha256']
                    or receipt['returncode'] != 1 or receipt['signal'] is not None
                    or receipt['automatic_retry'] is not False
                    or any((spool/name).exists() for name in ('OUTCOME.json', 'PREPARATION_RECEIPT.json'))
                    or (spool/'PREPARATION_BODY.log').stat().st_size != 0):
                return False
            with (spool/'WORKER_EXECUTE.log').open('rb') as log:
                log.seek(-min(4096, os.fstat(log.fileno()).st_size), os.SEEK_END)
                tail = log.read().decode('utf-8', errors='replace').rstrip()
            return ('in execute\n    process=subprocess.run(command,cwd=source,env=env,' in tail
                    and 'raise child_exception_type(errno_num, err_msg, err_filename)' in tail
                    and tail.split('\n')[-1].startswith('OSError: [Errno 7] Argument list too long:'))
        except (OSError, KeyError, TypeError, ValueError):
            return False

    def _compact_frozen_df_bindings(self, preparation):
        """Keep only historical aliases referenced by exact frozen input bytes."""
        bindings = preparation['path_bindings']
        aliases = {alias for alias, original in self.aliases.items()
                   if bindings.get(alias) is not None and bindings[alias] == bindings.get(original)}
        if not aliases:
            return bindings
        documents = [json.dumps(preparation['command'], ensure_ascii=False).encode()]
        for name, sha in preparation['inputs'].items():
            if name.endswith('.json'):
                body = self.executor.artifact_cache.path(sha).read_bytes()
                if hashlib.sha256(body).hexdigest() != sha:
                    raise ValueError('Frozen preparation artifact differs: '+name)
                documents.append(body)
        for entry in preparation['input_hashes']:
            if entry['path'].endswith('.json'):
                body = subprocess.check_output(['git', 'show', entry['commit']+':'+entry['path']],
                                               cwd=preparation['controller_root'], stderr=subprocess.PIPE)
                if hashlib.sha256(body).hexdigest() != entry['sha256']:
                    raise ValueError('Frozen preparation Git input differs: '+entry['path'])
                documents.append(body)
        text = b'\n'.join(documents).decode('utf-8')
        return {path: binding for path, binding in bindings.items()
                if path not in aliases or path in text
                or json.dumps(path, ensure_ascii=True)[1:-1] in text}

    @staticmethod
    def max_width_signal_locator(preparation):
        command=preparation.get('command',[])
        if (preparation.get('provider')!='DF_SIGNALS' or '--transport-source' not in command
                or 'private/df-terminal-transport/src/trottertracks/pr_df_max_width.py' not in preparation.get('inputs',{})):
            return None
        root=Path(preparation['controller_root']).resolve()
        path=Path(command[command.index('--input')+1]).resolve().relative_to(root).as_posix()
        commit=command[command.index('--inputs-commit')+1]
        return dict(commit=commit,path=path)

    def max_width_failed_input_owner(self, directory, job, row, locator):
        plan=self.settings.get('failed_preparation_recalculations',{}).get(job,{})
        if (not plan.get('df_signal_input_recovery') or not self.approved or self.executor._user_stop_requested()
                or not row or row['state']!='ACCEPTED' or not row.get('completion_sha256')
                or (row['attempt_id'],row['request_sha256'])!=(plan.get('attempt_id'),plan.get('request_sha256'))
                or row['request']['preparation'].get('supersedes_preparation')):
            return None
        proof=self.recovery_evidence(directory,'DF_SIGNALS',job)
        if proof.get('action') not in ('CHECK_FAILED_ATTEMPT_ARTIFACTS','PRESERVE_FAILED_ATTEMPT') or proof.get('failed_signal_rounds'):
            return None
        owner=self.executor.run_dir/'jobs'/row['method']/job
        outcome=read(owner/'OUTCOME.json');receipt=read(owner/'PROCESS_RETURN.json')
        expected=repr(['git','show',locator['commit']+':'+locator['path']])
        failure=outcome.get('failure_summary','')
        if not failure and (body:=plan.get('df_signal_body_log')):
            # Older PBS transports return the receipt but omit the body summary.
            # Accept only the captured log of this selected, already-ended attempt.
            if body.get('attempt_id')!=row['attempt_id']:return None
            raw=Path(body['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=body['sha256'] or b'raw = read_git(a.inputs_commit, relative)' not in raw:return None
            failure=raw.decode(errors='replace')
        if (outcome.get('status')!='PREPARATION_FAILED' or outcome.get('request_sha256')!=row['request_sha256']
                or expected not in failure or 'exit status 128' not in failure
                or receipt.get('returncode') in (None,0) or receipt.get('signal') is not None
                or receipt.get('automatic_retry') is not False
                or any('/df_signals/' in path for path in outcome.get('artifacts',{}))
                or any((directory/'df_signals').glob('started/*.json'))):
            return None
        return plan

    def repair_max_width_signal_input(self, directory, stage, submitted):
        """Repair an omitted frozen locator, never an executed numerical round."""
        if stage!='DF_SIGNALS' or not submitted.get('job_id'):return None
        job=submitted['job_id'];original=submitted['request'];prep=original['preparation']
        locator=self.max_width_signal_locator(prep)
        if locator is None or any((x['commit'],x['path'])==(locator['commit'],locator['path']) for x in prep['input_hashes']):return None
        row=self.executor.store.row(job);plan=None
        if row and row['state']=='ACCEPTED':
            plan=self.max_width_failed_input_owner(directory,job,row,locator)
            if plan is None:return None
            with self.executor.store._connection() as db:
                successors=db.execute("SELECT job_id FROM jobs WHERE json_extract(request,'$.preparation.supersedes_preparation.job_id')=?",(job,)).fetchall()
            if len(successors)>1:raise ValueError('Multiple successors of a fixed failed signal request')
            if successors:
                request=deepcopy(self.executor.store.row(successors[0][0])['request']);request.pop('request_id',None)
                return dict(request=request,source=submitted['source'],automatic_retry=False)
        elif row and (row['state']!='CANCELLED' or row.get('attempt_id') is not None):
            return None  # An assigned or running request remains owned by its original worker.
        request=deepcopy(row['request'] if row else original);request.pop('request_id',None);prep=request['preparation']
        root=Path(prep['controller_root']).resolve()
        raw=subprocess.check_output(['git','show',locator['commit']+':'+locator['path']],cwd=root)
        sha=hashlib.sha256(raw).hexdigest()
        if prep['inputs'].get('workspace/'+locator['path'])!=sha:
            raise ValueError('Original signal request bytes differ from the fixed Git input')
        prep['input_hashes'].append(dict(locator,sha256=sha))
        prep['input_git_bundles']=[dict(x,checkout_input_head=False) for x in self._bundle(root,prep['source']['commit'],prep['input_hashes'],shared_base=True)]
        self._validate_bundle_inputs(prep)
        if plan:
            prep['supersedes_preparation']=dict(job_id=job,attempt_id=row['attempt_id'],reason=plan['reason'])
        return dict(request=request,source=submitted['source'],automatic_retry=False)

    def _transport_repair(self, directory, stage, submitted):
        """Build a successor of a proved transport-only failure from its own request."""
        from .pr_candidate_completion import _DF_PRICE_BOOTSTRAP

        if stage not in STAGES or (directory/'STOP.json').exists() or getattr(self.executor, '_user_stop_requested', lambda: False)():
            return None
        job = submitted['job_id']
        proof = self.recovery_evidence(directory, stage, job)
        if not proof.get('attempt_id'):
            return None
        row = self.executor.store.row(job)
        request = deepcopy(row['request'])
        request.pop('request_id')
        preparation = request['preparation']
        if preparation.get('supersedes_preparation') or preparation.get('artificial'):
            return None
        if stage == 'DF_MODEL_FREEZE' and proof.get('action') == 'CHECK_ORIGINAL_OUTPUT':
            command = preparation['command']
            legacy = _DF_PRICE_BOOTSTRAP.split("if '--input' in sys.argv and ('freeze'", 1)[0]
            legacy = legacy.replace("    fixed_inputs = json.loads(document.read_bytes())\n"
                                    "    provider = resolve(fixed_inputs['runtime']['provider_root'])",
                                    "    provider = resolve(json.loads(document.read_bytes())['runtime']['provider_root'])")
            legacy += 'raise SystemExit(module.main())\n'
            if (proof['action'] != 'CHECK_ORIGINAL_OUTPUT'
                    or not proof['reason'].startswith('DF_FREEZE_REQUIRED_OUTPUT_NOT_RETURNED:')
                    or command.count(legacy) != 1 or 'freeze' not in command
                    or '--raw-commit' not in command or not preparation['dependencies']):
                return None
            try:
                receipt = read(self.executor.run_dir/'jobs'/row['method']/job/'PROCESS_RETURN.json')
            except (OSError, ValueError):
                return None
            if receipt['returncode'] != 0 or receipt['signal'] is not None:
                return None
            command[command.index(legacy)] = _DF_PRICE_BOOTSTRAP
            reason = 'AUDIT_OUTPUT_REDIRECTION'
        elif stage in ('DF_CALIBRATION','DF_HOLDOUT') and self._exec_rejected(row):
            compact = self.compact_bindings.get(job)
            if compact is None:
                try:
                    compact = self._compact_frozen_df_bindings(preparation)
                except (OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError):
                    return None
                self.compact_bindings[job] = compact
            if compact == preparation['path_bindings']:
                return None
            preparation['path_bindings'] = compact
            reason = 'EXEC_REJECTED_UNUSED_PATH_ALIASES'
        elif (stage == 'DF_CALIBRATION' and self.approved and not self.executor._user_stop_requested()
                and proof['action'] == 'CHECK_FAILED_ATTEMPT_ARTIFACTS' and proof.get('signal') == 15
                and job not in self.settings.get('failed_preparation_recalculations', {})):
            plan = self.settings.get('interrupted_calibrations', {}).get(job)
            if plan is None:
                plan = self.interrupted_calibration_artifacts(row)
            if plan is None:
                return None
            request = self.interrupted_calibration_request(job, plan)
            return dict(request=request, source=submitted['source'], automatic_retry=False)
        elif (proof.get('pre_science_failure') and proof.get('action') == 'PRESERVE_FAILED_ATTEMPT'
                and proof.get('state') == 'ACCEPTED' and self.approved):
            reason = 'VERIFIED_PRE_SCIENCE_WORKSPACE_FAILURE'
        else:
            return None
        preparation['supersedes_preparation'] = dict(job_id=job, attempt_id=proof['attempt_id'], reason=reason)
        return dict(request=request, source=submitted['source'], automatic_retry=False)

    def interrupted_calibration_artifacts(self, row):
        """Check returned axes against the frozen acquisition input using its normal receipt gate."""
        job = row['job_id']
        root = self.executor.run_dir/'jobs'/row['method']/job
        launch = read(root/'LAUNCH.json')
        if (launch['host_id'], launch['attempt_id']) != (row['host_id'], row['attempt_id']):
            raise ValueError('Interrupted calibration launch owner differs: '+job)
        if read(root/'PREPARATION_RECEIPT.json').get('provider_session_drained') is not True:
            return None
        preparation = row['request']['preparation']
        [output] = preparation['outputs']
        manifest = read(root/'OUTCOME.json')['artifacts']
        candidates = [name for name in manifest if df_calibration_receipt(name, output)]
        if not candidates:
            return None
        cache = self.executor.artifact_cache
        input_name = output+'/df_prices/INPUTS.json'
        input_sha = preparation['inputs'][input_name]
        if cache.missing([input_sha, *(manifest[name] for name in candidates)]):
            return None
        raw = cache.path(input_sha).read_bytes()
        command = preparation['command']
        fixed_commit = command[command.index('--inputs-commit')+1]

        def git_bytes(commit, path):
            return subprocess.check_output(['git', 'show', commit+':'+path],
                                           cwd=preparation['controller_root'], stderr=subprocess.PIPE)

        inputs = json.loads(raw)
        if (input_name != 'workspace/'+inputs['output_directory']+'/INPUTS.json'
                or raw != git_bytes(fixed_commit, inputs['output_directory']+'/INPUTS.json')):
            raise ValueError('Interrupted calibration input differs from its fixed request')
        gate = runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/run_pr_df_candidate_prices.py'))
        cells = {cell['m']: cell for cell in inputs['cells'] if cell['role'] == 'CALIBRATION'}
        witnesses, reuse = {}, {}

        def read_axis(locator):
            return read(cache.path(manifest[locator['path']]))

        for name in sorted(candidates):
            m, index, kind = df_calibration_receipt(name, output).groups()
            if kind != 'PATH_RECEIPT':
                continue
            cell = cells.get(int(m))
            index = int(index)
            if cell is None or index not in cell['indices']:
                raise ValueError('Interrupted calibration returned receipt outside its fixed cells')
            folder = artifact_path(name).parent
            for axis, filename in (('X', 'cosine.json'), ('Y', 'sine.json')):
                axis_path = str(folder/filename)
                if axis_path not in manifest:
                    continue
                axis_key = int(m), index, axis
                witness = dict(path_receipt=dict(commit='WORKING', path=name),
                               receipt=dict(commit='WORKING', path=axis_path))
                validated = gate['validate_axis'](cell, index, axis, witness,
                                                  reader=read_axis, source_reader=git_bytes)
                if axis_key in witnesses:
                    raise ValueError('Interrupted calibration returned more than one receipt for an axis')
                witnesses[axis_key] = validated
                reuse.update({name: manifest[name], axis_path: manifest[axis_path]})
        if not witnesses:
            return None
        for m, index, axis in witnesses:
            other = 'Y' if axis == 'X' else 'X'
            companion = witnesses.get((m, index, other))
            if companion is not None and companion['event_digest'] != witnesses[m, index, axis]['event_digest']:
                raise ValueError('Interrupted calibration X/Y event digest differs')
        return dict(attempt_id=row['attempt_id'], request_sha256=row['request_sha256'], reused_artifacts=reuse)

    def interrupted_calibration_request(self, original_job, recovery):
        """Continue a selected, ended calibration with its verified path receipts."""
        from .rpe_q3_coordinator import digest as request_digest

        row = self.executor.store.row(original_job)
        if (not row or row['state'] != 'ACCEPTED' or not row['completion_sha256']
                or row['attempt_id'] != recovery['attempt_id']
                or row['request_sha256'] != recovery['request_sha256']
                or request_digest(row['request']) != row['request_sha256']
                or row['detail'].get('outcome_status') != 'PREPARATION_FAILED'):
            raise ValueError('Calibration does not have its original terminal ownership: '+original_job)
        root = self.executor.run_dir/'jobs'/row['method']/original_job
        outcome = read(root/'OUTCOME.json')
        receipt = read(root/'PREPARATION_RECEIPT.json')
        if (outcome.get('request_sha256') != row['request_sha256'] or outcome['status'] != 'PREPARATION_FAILED'
                or receipt.get('signal') != 15 or receipt.get('returncode') is not None
                or receipt.get('automatic_retry') is not False):
            raise ValueError('Calibration predecessor is not a recorded interruption: '+original_job)
        request = deepcopy(row['request'])
        request.pop('request_id')
        preparation = request['preparation']
        if preparation['provider'] != 'DF_CALIBRATION' or preparation.get('supersedes_preparation'):
            raise ValueError('Selected predecessor is not an original calibration: '+original_job)
        paths = recovery['reused_artifacts']
        if any(not df_calibration_receipt(name, preparation['outputs'][0]) for name in paths):
            raise ValueError('Interrupted calibration receipt is outside its original result scope')
        if self.executor.artifact_cache.missing(paths.values()):
            raise ValueError('Interrupted calibration receipt is missing from the fixed input cache')
        for name, sha in paths.items():
            if preparation['inputs'].setdefault(name, sha) != sha:
                raise ValueError('Interrupted calibration receipt would change a frozen input: '+name)
            binding = dict(path=name, sha256=sha)
            if binding not in preparation['working_input_hashes']:
                preparation['working_input_hashes'].append(binding)
        preparation['supersedes_preparation'] = dict(job_id=original_job, attempt_id=row['attempt_id'],
                                                      reason='INTERRUPTED_CALIBRATION_MISSING_WRAPPERS')
        return request

    def reconcile_interrupted_calibrations(self):
        """Send selected continuations through the normal preparation scheduler."""
        from .rpe_preparation_queue import preparation_job_id
        from .rpe_release_registry import write

        plans = self.settings.get('interrupted_calibrations', {})
        if not plans or not self.approved or self.executor._user_stop_requested():
            return
        path = self.executor.run_dir/'preparation'/'INTERRUPTED_CALIBRATIONS.json'
        states = read(path) if path.is_file() else {}
        for original, plan in sorted(plans.items()):
            try:
                request = self.interrupted_calibration_request(original, plan)
                job = preparation_job_id(request)
                states[original] = dict(job_id=job, expected_new_wrappers=plan['missing_wrappers'])
                write(path, states)
                self.executor.submit_preparation(request, budget=self.settings['budgets']['compile'],
                    approved=self.approved, contract=request['original_contract'],
                    dependencies=request['preparation']['dependencies'], expected_job_id=job)
                result = self.executor.preparation_result(job, adopt=False, materialize=False)
                states[original]['status'] = result['status'] if result else 'PENDING'
            except (OSError, KeyError, TypeError, ValueError, sqlite3.Error) as error:
                states[original] = dict(status='RECOVERY_INPUT_HOLD', reason=str(error))
        write(path, states)

    def failed_preparation_request(self, job, plan):
        """Derive one successor from a failed, fully returned preparation."""
        from .pr_candidate_completion import (_DF_PRICE_BOOTSTRAP, _DF_PRICE_STARTUP_RECOVERY_BOOTSTRAP,
                                              _LEGACY_DF_PRICE_BOOTSTRAP)
        from .rpe_q3_coordinator import digest as request_digest

        owner = self.executor.run_dir
        if plan.get('prior_run'):
            from contextlib import closing
            from .rpe_q3_store import JobStore

            owner = Path(plan['prior_run']).resolve()
            if owner not in {path.resolve() for path in self.executor.prior_run_dirs}:
                raise ValueError('Preparation predecessor is outside the fixed prior runs: '+job)
            database = owner/'distributed'/'QUEUE.sqlite3'
            with closing(sqlite3.connect(database.as_uri()+'?mode=ro', uri=True)) as db:
                db.row_factory = sqlite3.Row
                row = JobStore._decoded(db.execute('SELECT * FROM jobs WHERE job_id=?', (job,)).fetchone())
        else:
            row = self.executor.store.row(job)
        if (not row or row['state'] != 'ACCEPTED' or not row['completion_sha256']
                or not row['attempt_id'] or row['attempt_id'] != plan['attempt_id']
                or row['request_sha256'] != plan['request_sha256']
                or request_digest(row['request']) != row['request_sha256']):
            raise ValueError('Original preparation has no matching terminal attempt: '+job)
        request = deepcopy(row['request'])
        preparation = request['preparation']
        root = owner/'jobs'/row['method']/job
        launch = read(root/'LAUNCH.json')
        receipt = read(root/'PROCESS_RETURN.json')
        outcome = read(root/'OUTCOME.json')
        command = preparation['command']
        repair_loader = (plan.get('repair_legacy_price_loader') is True
                         and preparation['provider'] in ('DF_CALIBRATION', 'DF_HOLDOUT')
                         and command[2:4] == ['-c', _LEGACY_DF_PRICE_BOOTSTRAP])
        startup = plan.get('df_startup_recovery')
        if ((preparation.get('supersedes_preparation') and not (repair_loader or startup))
                or preparation.get('optional_price_validation')
                or read(root/'REQUEST.json') != request
                or (launch['host_id'], launch['attempt_id']) != (row['host_id'], row['attempt_id'])
                or outcome.get('request_sha256') != row['request_sha256']
                or outcome.get('status') != 'PREPARATION_FAILED' or outcome.get('artifacts')
                or receipt.get('automatic_retry') is not False
                or not (receipt.get('returncode') or receipt.get('signal'))
                or (plan.get('prior_run') and preparation['dependencies'])):
            raise ValueError('Original preparation needs ownership or partial-result reconciliation: '+job)
        reason = plan['reason']
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('Missing reason for preparation recalculation: '+job)
        if plan.get('repair_legacy_price_loader') and not repair_loader:
            raise ValueError('Original preparation does not use the legacy DF price loader: '+job)
        if repair_loader:
            command[3] = _DF_PRICE_BOOTSTRAP
        if startup:
            if (plan.get('prior_run') or not preparation.get('supersedes_preparation')
                    or preparation['provider'] != 'DF_CALIBRATION' or preparation['stage'] != 'compile'
                    or command[2:4] != ['-c', _DF_PRICE_BOOTSTRAP]):
                raise ValueError('DF startup recovery requires a failed fixed calibration successor')
            self.bind_failed_df_startup_inputs(preparation, row, startup)
            command[3] = _DF_PRICE_STARTUP_RECOVERY_BOOTSTRAP
        if (repair_loader or startup) and (core := self.settings.get('df_price_openblas_coretype')):
            # Old provider CLIs predate --openblas-coretype. Fix the kernel in
            # this child before its imports, using the run's existing policy.
            if core not in ('HASWELL', 'NEHALEM'):
                raise ValueError('Unsupported fixed DF price OpenBLAS core')
            command[3] = (
                "import os\n"
                f"os.environ.update(OPENBLAS_CORETYPE={core!r}, OPENBLAS_NUM_THREADS='1', "
                "OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')\n" + command[3]
            )
        request.pop('request_id')
        preparation['supersedes_preparation'] = dict(job_id=job, attempt_id=row['attempt_id'], reason=reason)
        return request

    def bind_failed_df_startup_inputs(self, preparation, row, startup):
        """Keep the provider's startup reads in the same immutable input set."""
        from .rpe_preparation_artifacts import artifact_path

        log = Path(startup['body_log']).read_bytes()
        evidence = read(Path(startup['evidence']))
        if (hashlib.sha256(log).hexdigest() != startup['body_log_sha256']
                or (evidence['job_id'], evidence['attempt_id']) != (row['job_id'], row['attempt_id'])
                or evidence['paths'] != ['DRIVER.lock']
                or b'R=runtime();base=inputs' not in log
                or b'subprocess.CalledProcessError: Command ' not in log or b"'git', 'show'" not in log):
            raise ValueError('DF startup recovery requires an exited Git input failure before any path started')
        cache = self.executor.artifact_cache
        input_name = next((name for name in preparation['inputs'] if name.endswith('/df_prices/INPUTS.json')), None)
        if input_name is None:
            raise ValueError('DF startup recovery has no fixed price inputs')
        input_sha = preparation['inputs'][input_name]
        if cache.missing([input_sha]):
            raise ValueError('DF startup recovery is missing the saved price inputs')
        for name, saved in read(cache.path(input_sha))['saved_input_files'].items():
            relative = artifact_path('private/df-physical-inputs/'+name).as_posix()
            sha = saved['sha256']
            if cache.missing([sha]) or cache.path(sha).stat().st_size != saved['bytes']:
                raise ValueError('DF startup recovery is missing the saved physical input '+name)
            preparation['inputs'][relative] = sha
            preparation['path_bindings'][saved['path']] = '@WORK@/'+relative
        known = {(row['commit'], row['path']): row['sha256'] for row in preparation['input_hashes']}
        for locator in startup['git_inputs']:
            key = (locator['commit'], locator['path'])
            if key not in known:
                preparation['input_hashes'].append(locator)
                known[key] = locator['sha256']
            elif known[key] != locator['sha256']:
                raise ValueError('DF startup recovery changes a fixed Git input')

    def restore_recalculation_inputs(self, request, prior_run):
        """Restore the fixed input objects from the checked predecessor's run."""
        from .rpe_preparation_artifacts import ArtifactCache
        from .rpe_preparation_worker import metadata_bundle_hashes

        cache = self.executor.artifact_cache
        hashes = [*request['preparation']['inputs'].values(), *metadata_bundle_hashes(request)]
        missing = cache.missing(hashes)
        if missing:
            prior_cache = ArtifactCache(Path(prior_run)/'distributed'/'artifacts')
            unavailable=prior_cache.missing(missing)
            if unavailable:
                raise ValueError('Missing cached artifacts: '+','.join(unavailable))
            for sha in missing:
                cache.put_file(prior_cache.path(sha), sha)

    def reconcile_failed_preparations(self):
        """Reserve exact retries through the existing stage budgets and persist their outcome."""
        from .rpe_preparation_queue import preparation_job_id
        from .rpe_release_registry import write

        plans = self.settings.get('failed_preparation_recalculations', {})
        if not plans or not self.approved or self.executor._user_stop_requested():
            return
        path = self.executor.run_dir/'preparation'/'FAILED_PREPARATION_RECALCULATIONS.json'
        states = read(path) if path.is_file() else {}
        for original, plan in sorted(plans.items()):
            if self.executor._user_stop_requested():
                break
            if plan.get('df_signal_input_recovery'):continue  # the owning candidate submits and adopts this successor
            reason = plan['reason']
            try:
                request = self.failed_preparation_request(original, plan)
                preparation = request['preparation']
                job = preparation_job_id(request)
                with self.executor.store._connection() as db:
                    other = db.execute("""SELECT job_id FROM jobs WHERE job_id != ?
                        AND json_extract(request, '$.preparation.supersedes_preparation.job_id') = ?
                        AND state != 'CANCELLED' LIMIT 1""", (job, original)).fetchone()
                if other:
                    states[original] = dict(job_id=other[0], reason=reason, status='EXISTING_SUCCESSOR')
                    continue
                row = self.executor.store.row(job)
                if plan.get('prior_run') and (row is None or row['state'] in ('QUEUED', 'CANCELLED')):
                    self.restore_recalculation_inputs(request, plan['prior_run'])
                self.executor.submit_preparation(request, budget=self.settings['budgets'][preparation['stage']],
                    approved=self.approved, contract=request['original_contract'],
                    dependencies=preparation['dependencies'], expected_job_id=job)
                result = self.executor.preparation_result(job, adopt=False, materialize=False)
                if (plan.get('df_startup_recovery') and result and result['status'] == 'PREPARATION_COMPLETE'
                        and (row is None or row['adopted_epoch'] is None)):
                    result = self.executor.preparation_result(job, adopt=False)
                    [output] = preparation['outputs']
                    summary = read(Path(result['artifact_directory']) / artifact_path(output)
                                   / 'df_prices/calibration/acquisition/RESULT.json')
                    if (summary['status'] != 'ACQUIRED' or summary['held_axes'] != 0
                            or summary['source'] != preparation['source']['commit']):
                        raise ValueError('Returned calibration does not cover all fixed axes')
                    self.executor.adopt_preparation(job)
                states[original] = dict(job_id=job, reason=reason, status=result['status'] if result else 'PENDING')
            except (OSError, KeyError, TypeError, ValueError, sqlite3.Error) as error:
                states[original] = dict(status='RECOVERY_INPUT_HOLD', reason=reason, detail=str(error))
        write(path, states)

    def reconcile_pf_input_redeliveries(self, settings, interpreter):
        """Resume a proved delivery failure at its original PF stage."""
        from types import SimpleNamespace

        from .pr_candidate_completion import Held, PreparationWait
        from .rpe_preparation_queue import normalize_preparation_request
        from .rpe_release_registry import write

        plans = {job: proof for job, proof in self.settings.get('pf_stage_audit_recoveries', {}).items()
                 if proof.get('kind') == 'PF_PORTABLE_INPUT_DELIVERY'}
        if not plans or not self.approved or self.executor._user_stop_requested():
            return
        run = self.executor.run_dir
        index = run/'pf_auto/WAVES.json'
        if not index.is_file():
            return
        candidates = read(index)
        states = {}
        for job, proof in plans.items():
            if self.executor._user_stop_requested():
                break
            try:
                candidate = proof['configuration_id']
                row = candidates[candidate]
                if row.get('configurations') or row['status'] == 'READY_RETURNED_TO_SAME_OPTIMIZER':
                    states[job] = dict(status='CANDIDATE_ALREADY_READY')
                    continue
                base = run/row.get('price_recheck_directory', (
                    'pf_auto/price-recovery' if row.get('price_receipt_recheck') else 'pf_auto'))
                workspace = (base/'candidates'/candidate/'pf_auto/pf_completion/provider_workspace').resolve()
                if not workspace.is_relative_to(run.resolve()):
                    raise ValueError('PF delivery workspace is outside the fixed run')
                directory = workspace/'docs/research/distributed_selected_pf'/candidate
                owner = self.executor.store.row(job)
                prep = owner['request']['preparation']
                stage = prep['provider']
                if stage not in ('PF_COMPILE', 'PF_HOLDOUT_COMPILE'):
                    raise ValueError('PF input redelivery only supports compile stages')
                marker = read(directory/(stage+'.DISTRIBUTED.json'))
                original = marker['request']['preparation'].get('supersedes_preparation', {})
                if marker['job_id'] == job:
                    requested = normalize_preparation_request(
                        marker['request'], dependencies=self._dependencies(directory, stage))
                    if requested != owner['request']:
                        raise ValueError('Original PF stage request differs from the queue')
                elif (original.get('job_id'), original.get('attempt_id')) != (job, proof['audit_attempt']):
                    raise ValueError('A different request owns the original PF stage')
                provider = read(workspace.parent/'PROVIDER_SOURCE.json')
                if provider['source'] != prep['source']['commit']:
                    raise ValueError('Original PF provider SOURCE differs')
                stage_input = directory/(stage+'.INPUT.json')
                if hashlib.sha256(stage_input.read_bytes()).hexdigest() != proof['input_sha256']:
                    raise ValueError('Original PF stage input differs')
                completion = SimpleNamespace(directory=directory, settings=settings,
                    source=prep['source']['commit'], preparation_root=workspace,
                    preparation_source_manifest=prep['source'], log=self.executor.log)
                output = directory/(stage+'.RESULT.json')
                cmd = [interpreter, '-B', str(self.root/'scripts/run_pf_preparation_stage.py'),
                       '--provider-root', str(workspace), '--input', str(stage_input), '--output', str(output)]
                self.call(completion, cmd, directory, stage, output, wait=False)
                states[job] = dict(status='STAGE_COMPLETE')
            except PreparationWait as error:
                states[job] = dict(status='WAITING_FOR_STAGE', reason=str(error)[:500])
            except (Held, OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError) as error:
                states[job] = dict(status='RECOVERY_INPUT_HOLD', reason=str(error)[:500])
        write(run/'preparation/PF_INPUT_REDELIVERIES.json', states)

    def recoverable_transport(self, directory, reason):
        """Use the original marker to identify the failed stage cheaply."""
        if not reason.startswith(('DISTRIBUTED_PREPARATION_OPERATIONAL_HOLD:', 'DISTRIBUTED_PREPARATION_IN_PROGRESS:',
                                  'STAGE_FAILED:', 'PREPARATION_RESULT_MISMATCH:')):
            return False
        if (Path(directory)/'STOP.json').exists() or getattr(self.executor, '_user_stop_requested', lambda: False)():
            return False
        if reason.startswith('DISTRIBUTED_PREPARATION_IN_PROGRESS:'):
            _, _, payload = reason.partition(':')
            stage, _, job_id = payload.partition(':')
            # Revisit the exact received receipt, including a failed result so
            # its normal failure classification is recorded. Never retry it.
            return self.recovery_evidence(directory, stage, job_id).get('state') == 'ACCEPTED'
        # A later success receipt takes precedence over an old operational
        # hold for this exact request, irrespective of provider/stage.
        if self.received_after_reservation_stop(directory, reason):
            return True
        job = re.search(r'\bprep-[0-9a-f]{24}\b', reason)
        if not job:
            return False
        for stage in STAGES:
            if stage not in reason and not reason.startswith('PREPARATION_RESULT_MISMATCH:ORIGINAL_AUDIT_OUTPUT_MISSING:'):
                continue
            for path in directory.rglob(stage+'.DISTRIBUTED.json'):
                try:
                    submitted = read(path)
                    superseded = submitted['request']['preparation'].get('supersedes_preparation', {})
                    if superseded.get('job_id') == job[0]:
                        return True  # Resume the already-recorded successor.
                    if submitted.get('job_id') == job[0]:
                        if stage=='DF_SIGNALS':
                            locator=self.max_width_signal_locator(submitted['request']['preparation'])
                            row=self.executor.store.row(job[0])
                            if locator and (row is None or (row['state']=='CANCELLED' and row.get('attempt_id') is None)):
                                return True
                            if locator and self.max_width_failed_input_owner(path.parent,job[0],row,locator):return True
                        if self.pf_delivery_failure(path.parent, stage, job[0]):
                            return True
                        if self._transport_repair(path.parent, stage, submitted):
                            return True
                except (OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError):
                    continue
        return False

    def hold_evidence(self, directory, reason):
        prefix, _, payload = reason.partition(':')
        stage, _, detail = payload.partition(':')
        if prefix == 'DISTRIBUTED_PREPARATION_IN_PROGRESS' and stage in STAGES:
            return self.recovery_evidence(directory, stage, detail)
        if prefix == 'PREPARATION_RESULT_ARTIFACTS_MISSING':
            return self.recovery_evidence(directory, stage, detail.split(':', 1)[0])
        if prefix == 'STAGE_FAILED':
            _, _, detail = detail.partition(':NO_AUTOMATIC_RETRY:')
        elif prefix != 'DISTRIBUTED_PREPARATION_OPERATIONAL_HOLD':
            return None
        try:
            result = ast.literal_eval(detail)
        except (SyntaxError, ValueError):
            return None
        if not isinstance(result, dict):
            return None
        job_id = result.get('job_id') or result.get('request_id')
        return self.recovery_evidence(directory, stage, job_id) if job_id else None

    def received_stage(self, directory, stage, job_id):
        return self.recovery_evidence(directory, stage, job_id)['action'] == 'IMPORT_SUCCESSFUL_STAGE'

    def needs_original_compile_audit(self, directory, stage, job_id, audit):
        if not self.received_stage(directory, stage, job_id):
            return False
        try:
            return not self.executor.store.has_preparation_successor(job_id, audit)
        except sqlite3.Error:
            return False

    def resume_unreserved_stage(self, directory, reason):
        """Recover the same unclaimed request after a stop or a released budget hold."""
        prefix, _, payload = reason.partition(':')
        stage, _, detail = payload.partition(':')
        if prefix != 'DISTRIBUTED_PREPARATION_OPERATIONAL_HOLD' or stage not in STAGES:
            return False
        try:
            result = ast.literal_eval(detail)
            if not isinstance(result, dict):
                return False
            stopped = result.get('status') == 'PREPARATION_STOPPED_BEFORE_RESERVATION'
            stopped |= (result.get('status') == 'PREPARATION_HOLD'
                        and result.get('reason') == 'USER_REQUESTED_STOP')
            budget_hold = result.get('status') == 'PREPARATION_BUDGET_OR_PERMISSION_HOLD'
            if not (stopped or budget_hold):
                return False
            job = result['job_id']
            if self.executor._user_stop_requested() or self.executor.state.failure is not None:
                return False
            row = self.executor.store.row(job)
            if row is not None and (row['state'] not in ('QUEUED', 'CANCELLED') or row['attempt_id'] is not None):
                return False
            if budget_hold and (row is None or row['state'] == 'CANCELLED'):
                budget = self.settings.get('budgets', {}).get(STAGES[stage], 0)
                used = self.executor.store.preparation_reservations().get(STAGES[stage], 0)
                if not self.approved or not (budget == 'UNLIMITED' or type(budget) is int and used < budget):
                    return False
            from .rpe_preparation_queue import normalize_preparation_request
            markers = [path for path in Path(directory).rglob(stage+'.DISTRIBUTED.json')
                       if read(path).get('job_id') == job]
            if len(markers) != 1:
                return False
            marker = read(markers[0])
            request = normalize_preparation_request(marker['request'],
                dependencies=self._dependencies(markers[0].parent, stage))
            return (request['request_id'] == job and request['preparation']['provider'] == stage
                    and request['preparation']['source']['commit'] == marker['source']
                    and (row is None or request == row['request']))
        except (OSError, SyntaxError, KeyError, TypeError, ValueError, sqlite3.Error):
            return False

    def received_after_reservation_stop(self, directory, reason):
        prefix, _, payload = reason.partition(':')
        stage, _, detail = payload.partition(':')
        if prefix == 'PREPARATION_RESULT_ARTIFACTS_MISSING':
            job_id = detail.split(':', 1)[0]
            if not self.received_stage(directory, stage, job_id):
                return False
            wait = directory/(stage+'.IMPORT_WAIT.json')
            if wait.is_file():
                evidence = read(wait)
                if (evidence.get('job_id') != job_id or not evidence.get('missing_path')
                        or not Path(evidence['missing_path']).exists()):
                    return False
            # The missing artifact is recoverable only when the original
            # accepted result and its declared blobs are available again.
            result = self.executor.preparation_result(job_id, adopt=False)
            return bool(result and result.get('status') == 'PREPARATION_COMPLETE')
        if prefix != 'DISTRIBUTED_PREPARATION_OPERATIONAL_HOLD' or stage not in STAGES:
            return False
        if not detail.startswith("{'status':"):
            return False
        try:
            result = ast.literal_eval(detail)
        except (SyntaxError, ValueError):
            return False
        if not isinstance(result, dict) or not self.received_stage(directory, stage, result.get('job_id') or result.get('request_id')):
            return False
        if result.get('status') == 'PREPARATION_RESULT_ARTIFACTS_MISSING':
            recovered = self.executor.preparation_result(result['job_id'], adopt=False)
            return bool(recovered and recovered.get('status') == 'PREPARATION_COMPLETE')
        return True  # exact accepted success, regardless of the former operational wait

    def _blob(self, raw):
        sha = hashlib.sha256(raw).hexdigest()
        self.executor.artifact_cache.put({sha: base64.b64encode(raw).decode('ascii')})
        return sha

    def _saved_h4_snapshot(self, root, snapshot, inputs, bindings, working_hashes):
        """Bind saved H4 file bytes under their original names for any PF/DF leaf."""
        index = json.loads(subprocess.check_output(
            ['git', 'show', _DF_SNAPSHOT_INDEX['commit']+':'+_DF_SNAPSHOT_INDEX['path']], cwd=root))
        for item in (index['T20_reference_files']['H4'], *index['saved_inputs']['H4']['inputs'].values()):
            original = Path(item['path'])
            content = (Path(snapshot)/original.name).read_bytes()
            sha = self._blob(content)
            if sha != item['sha256']:
                raise ValueError('Fixed H4 snapshot digest differs: '+original.name)
            name = 'private/df-snapshot/'+original.name
            inputs[name] = sha
            bindings[str(original)] = '@WORK@/'+name
            working_hashes.append(dict(path=name, sha256=sha))

    def _pf_native_runtime(self, script, inputs, bindings, working_hashes):
        # The transport installed on each host can still be the prior worker.
        # Send the selected compiler wrapper and its path resolver as verified
        # request inputs, together with the exact native configuration.
        helper = Path(__file__).resolve().with_name('rpe_preparation_paths.py')
        adapter = helper.parents[2] / 'scripts/run_pf_preparation_stage.py'
        accelerated = adapter.with_name('rpe_pf_shared_fingerprints.py')
        constants = runpy.run_path(str(helper))
        origin = constants['PF_NATIVE_ORIGIN_CONFIG']
        for file, relative in ((adapter, 'scripts/run_pf_preparation_stage.py'),
                               (accelerated, 'scripts/rpe_pf_shared_fingerprints.py'),
                               (helper, 'src/trottertracks/rpe_preparation_paths.py')):
            destination = 'private/pf-transport/'+relative
            sha = self._blob(file.read_bytes())
            inputs[destination] = sha
            working_hashes.append(dict(path=destination, sha256=sha))
        bindings[str(script)] = '@WORK@/private/pf-transport/scripts/rpe_pf_shared_fingerprints.py'

        controller_paths = self.settings.get('controller_paths') or {}
        configured = controller_paths.get(origin, {})
        config_path = Path(configured.get('path', origin))
        raw = config_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != constants['PF_NATIVE_ORIGIN_SHA256']:
            raise ValueError('PF native origin configuration differs from its fixed SHA-256')
        config = json.loads(raw)
        for name in constants['PF_NATIVE_H4_INPUTS']:
            item = config['systems']['H4']['inputs'][name]
            original = item['path']
            default_path = Path(original)
            snapshot = self.settings.get('df_snapshot_directory')
            if snapshot:
                default_path = Path(snapshot)/default_path.name
            path = Path(controller_paths.get(original, {}).get('path', default_path))
            data = path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            if sha != item['sha256']:
                raise ValueError('PF native H4 input differs from its fixed SHA-256: '+name)
            destination = 'private/pf-native/'+name
            inputs[destination] = self._blob(data)
            bindings[original] = '@WORK@/'+destination
            working_hashes.append(dict(path=destination, sha256=sha))
        return dict(runtime=dict(input_config=origin,
                                 provider_root=config['provider_root']))

    @staticmethod
    def _input_tip(root, head, resolved):
        parents=[head,*sorted(set(resolved)-{head})]
        tree,date=subprocess.check_output(['git','show','-s','--format=%T %aI',head],cwd=root,text=True).split()
        args=['git','-c','user.name=HIROMU1015','-c','user.email=u411612g@ecs.osaka-u.ac.jp','commit-tree',tree]
        for parent in parents:args.extend(['-p',parent])
        return subprocess.check_output(args+['-m','Make declared preparation input commits reachable'],cwd=root,text=True,
            env=dict(os.environ,GIT_AUTHOR_DATE=date,GIT_COMMITTER_DATE=date)).strip()

    def _shared_local_git_inputs(self, source, input_hashes, *, provider=None):
        """A node-local worker inherits the same immutable object stores.

        Confirm the exact locators from an otherwise empty object database;
        the normal worker still verifies each content hash and fixed SOURCE.
        Do not replace a requested workspace checkout with locator-only data.
        """
        providers=getattr(getattr(self,'executor',None),'local_shared_preparation_providers',set())
        if (not self.settings.get('local_shared_git_inputs') and provider not in providers) or not input_hashes:
            return False
        if not os.environ.get('GIT_ALTERNATE_OBJECT_DIRECTORIES'):
            return False
        expressions = [source+'^{commit}', *sorted({row['commit']+':'+row['path']
                                                   for row in input_hashes})]
        with tempfile.TemporaryDirectory(prefix='rpe-shared-git-check-') as temporary:
            subprocess.run(['git','init','--bare','--quiet',temporary],check=True,
                           capture_output=True)
            result = subprocess.run(['git','--git-dir='+temporary,'cat-file','--batch-check'],
                input='\n'.join(expressions)+'\n',text=True,capture_output=True,check=True)
            rows = result.stdout.splitlines()
            return (len(rows)==len(expressions) and
                    all(len(row.split())==3 and row.split()[1] in ('commit','blob','tree')
                        for row in rows))

    def _bundle(self, root, source, input_hashes=(), *, tracked_workspace=False, shared_base=False, provider=None):
        if not tracked_workspace and self._shared_local_git_inputs(source, input_hashes, provider=provider):
            return []
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        resolved=[]
        if input_hashes:
            commits=sorted({row['commit']+'^{commit}' for row in input_hashes})
            resolved=subprocess.check_output(['git','rev-parse',*commits],cwd=root,text=True).split()
        # The fixed-provider checkout does not use the synthetic input tip's
        # tree. For newly prepared runs, anchor it to the fixed SOURCE, not
        # the continually advancing controller publication HEAD.
        stable = (self.settings.get('stable_input_bundle') is True and not tracked_workspace
                  and bool(input_hashes) and (shared_base or self.settings.get('bundle_fixed_source_only') is True))
        if stable:
            head = source
            # One common first input base per fixed run/source, including
            # ordinary fixed-source preparations, not just max-width callers.
            shared_base = True
        shared_locators=()
        if shared_base:
            # Private callers use the provider's fixed checkout. Carry the
            # first request's fixed locator history in one common base bundle
            # so later independent requests need only their own new objects.
            if tracked_workspace or not input_hashes:raise ValueError('Shared root bundle requires only declared inputs')
            identity=(str(root),source)
            with self.lock:
                if identity not in self.shared_bundle_heads:
                    saved_path=Path(self.executor.run_dir)/'preparation/SHARED_INPUT_BASES.json'
                    key=digest(identity)
                    saved=read(saved_path) if stable and saved_path.is_file() else {}
                    base=saved.get(key)
                    if base is None:
                        tip=self._input_tip(root,head,resolved)
                        locators=deepcopy(input_hashes)
                        if stable:
                            ref='refs/rpe-preparation/input-bundles/base-'+tip
                            subprocess.run(['git','update-ref',ref,tip],cwd=root,check=True)
                            from .rpe_q3_coordinator import atomic_json
                            saved[key]=dict(tip=tip,locators=locators)
                            saved_path.parent.mkdir(parents=True,exist_ok=True)
                            atomic_json(saved_path,saved)
                    else:
                        tip,locators=base['tip'],base['locators']
                        subprocess.run(['git','cat-file','-e',tip+'^{commit}'],cwd=root,check=True,capture_output=True)
                    self.shared_bundle_heads[identity]=(tip,locators)
                head,shared_locators=self.shared_bundle_heads[identity]
        bundles = []
        namespace = 'refs/rpe-preparation/input-bundles/'
        # With an explicit common bundle, the worker receives this prerequisite
        # from the same request, and does not rely on a moving controller HEAD.
        fixed_source_only = not shared_base and bool(input_hashes) and self.settings.get('bundle_fixed_source_only') is True
        if not fixed_source_only and head != source and subprocess.check_output(
            ['git', 'rev-list', '-1', head, '^' + source], cwd=root, text=True,
        ).strip():
            ref = namespace + 'base-' + head
            with self.lock:
                subprocess.run(['git', 'update-ref', ref, head], cwd=root, check=True)
            bundles.append(self._pack_bundle(root, ref, [source], shared_locators, not input_hashes))
        if not input_hashes:
            return bundles
        # The per-request bundle reaches every declared producer commit even
        # when they belong to different branches. The common base is imported
        # first without replacing the scientific provider checkout.
        tip=self._input_tip(root,head,resolved)
        ref = namespace + 'inputs-' + tip
        with self.lock:
            subprocess.run(['git','update-ref',ref,tip],cwd=root,check=True)
        prerequisites = [source] if fixed_source_only else list(dict.fromkeys([source, head]))
        bundles.append(self._pack_bundle(root,ref,prerequisites,input_hashes,tracked_workspace))
        return bundles

    def _saved_bundle(self, key):
        if self.settings.get('stable_input_bundle') is not True:
            return None
        with self.lock:
            if self.saved_bundle_hashes is None:
                self.saved_bundle_hashes={}
                path=Path(self.executor.run_dir)/'preparation/BUNDLE_CACHE.jsonl'
                if path.is_file():
                    try:
                        with path.open('r',errors='replace') as stream:
                            for line in stream:
                                try:
                                    row=json.loads(line)
                                    if re.fullmatch('[0-9a-f]{64}',row['key']) and re.fullmatch('[0-9a-f]{64}',row['sha256']):
                                        self.saved_bundle_hashes[row['key']]=row['sha256']
                                except (ValueError,KeyError,TypeError):
                                    continue  # a truncated disposable cache record
                    except OSError:
                        pass  # exact pack generation remains available
            return self.saved_bundle_hashes.get(key)

    def _remember_bundle(self, key, sha):
        if self.settings.get('stable_input_bundle') is not True:
            return
        with self.lock:
            if self.saved_bundle_hashes.get(key)==sha:
                return
            path=Path(self.executor.run_dir)/'preparation/BUNDLE_CACHE.jsonl'
            try:
                path.parent.mkdir(parents=True,exist_ok=True)
                # Do not rewrite the accumulated index. A leading newline
                # isolates any last incomplete record after a controller crash.
                with path.open('a') as stream:
                    stream.write('\n'+json.dumps(dict(key=key,sha256=sha),separators=(',',':'))+'\n')
                    stream.flush();os.fsync(stream.fileno())
                self.saved_bundle_hashes[key]=sha
            except OSError:
                pass  # the verified CAS object is already durable

    def _pack_bundle(self, root, ref, prerequisites, input_hashes, checkout, workspace_objects=()):
        locators = sorted({(row['commit'], row['path']) for row in input_hashes})
        key = (str(Path(root).resolve()),ref, tuple(prerequisites), tuple(locators), tuple(workspace_objects))
        with self.lock:
            lock=self.bundle_locks.setdefault(key,threading.Lock())
        with lock:
            with self.lock:
                sha = self.bundle_hashes.get(key)
            cache=self.executor.artifact_cache
            if not sha or not cache.path(sha).is_file():
                # The ref is an immutable synthetic tip, but bind its actual
                # Git identity too. Never reuse a packing recipe for new input.
                saved_key=None
                if self.settings.get('stable_input_bundle') is True:
                    tip=subprocess.check_output(['git','rev-parse',ref],cwd=root,text=True).strip()
                    saved_key=digest(('RPE_PREPARATION_BUNDLE_V1',key,tip))
                sha=self._saved_bundle(saved_key) if saved_key else None
                if sha and not cache.missing([sha]):
                    try:
                        heads=subprocess.check_output(['git','bundle','list-heads',str(cache.path(sha)),ref],
                                                      cwd=root,text=True).splitlines()
                        if heads != [tip+' '+ref]:sha=None
                    except (OSError,subprocess.SubprocessError):
                        sha=None
                else:
                    sha=None
                if sha is None:
                    with tempfile.TemporaryDirectory(prefix='rpe-preparation-git-') as temporary:
                        path = Path(temporary) / 'input.bundle'
                        build_preparation_git_bundle(root, ref, prerequisites, locators, path, workspace_objects)
                        from .rpe_preparation_artifacts import file_digest
                        sha = file_digest(path)
                        cache.put_file(path,sha)
                    if saved_key:self._remember_bundle(saved_key,sha)
                with self.lock:
                    self.bundle_hashes[key] = sha
        return dict(sha256=sha, ref=ref, checkout_input_head=checkout)

    def _workspace_bundle(self, root, view, prefix, outputs, *, extra_directories=()):
        """Carry the Git objects an owned sparse workspace needs to materialize."""
        from .rpe_preparation_worker import checkout_directories, output_directories

        manifest = view['source']
        base = manifest['commit']
        cache = self.executor.artifact_cache
        checkout = base
        if view.get('checkout_input_head', True):
            for item in view['input_git_bundles']:
                if item.get('checkout_input_head', True):
                    checkout = subprocess.check_output(
                        ['git','bundle','list-heads',str(cache.path(item['sha256'])),item['ref']],
                        text=True).split()[0]
        directories = checkout_directories(manifest['files']) | output_directories(outputs, prefix, root)
        ancestors = {'.'} | {str(p) for d in directories for p in Path(d).parents}
        objects = set()

        def add_listing(commit, flags, paths):
            listing = subprocess.check_output(['git','ls-tree','-z',*flags,commit,'--',*paths],cwd=root)
            for entry in listing.split(b'\0'):
                if entry:
                    _, kind, oid = entry.split(b'\t',1)[0].split()
                    if kind in (b'blob', b'tree'):
                        objects.add(oid)

        for commit in {base, checkout}:
            # Cone checkout includes the files directly inside each ancestor.
            for name in sorted(ancestors):
                add_listing(commit, (), [] if name == '.' else [name+'/'])
            # Fixed Git workflows and licenses can be read when a scoped
            # commit expands a sparse index.
            add_listing(commit, ('-r','-t'),
                        sorted(directories | {'.github','LICENSES',*extra_directories}))
        key = hashlib.sha256(b'\0'.join(sorted(objects))).hexdigest()
        ref = 'refs/rpe-preparation/input-bundles/workspace-' + base + '-' + key
        with self.lock:
            if subprocess.run(['git','show-ref','--verify','--quiet',ref],cwd=root).returncode != 0:
                date = subprocess.check_output(['git','show','-s','--format=%aI',base],cwd=root,text=True).strip()
                tip = subprocess.check_output(
                    ['git','-c','user.name=PREPARATION','-c','user.email=preparation@localhost',
                     'commit-tree',base+'^{tree}','-p',base,'-m','Carry sparse checkout objects'],
                    cwd=root,text=True,env=dict(os.environ,GIT_AUTHOR_DATE=date,GIT_COMMITTER_DATE=date)).strip()
                subprocess.run(['git','update-ref',ref,tip],cwd=root,check=True)
        return dict(self._pack_bundle(root,ref,[base],(),False,sorted(objects)), workspace_objects=True)

    def _validate_bundle_inputs(self, spec):
        sources = self.settings.get('bundle_validation_sources')
        if not sources or not (spec['input_git_bundles'] or spec['workspaces']):
            return
        from .rpe_preparation_worker import _clone, _guard_owned_source, _include_declared_outputs

        cache = self.executor.artifact_cache
        with tempfile.TemporaryDirectory(prefix='rpe-owned-preparation-check-') as temporary:
            directory = Path(temporary)
            owned = {}
            views = dict(source=spec, **spec['workspaces'])
            for alias, view in views.items():
                manifest = view['source']
                baseline = sources.get(manifest['commit'])
                if not baseline and alias == 'source':
                    from .rpe_preparation_worker import select_preparation_profile
                    configured = getattr(self.executor, 'distributed', {}).get('preparation_profiles', {}).get('local')
                    if configured:
                        local = configured if isinstance(configured, dict) else read(Path(configured))
                        profile = select_preparation_profile(local, dict(preparation=spec))
                        baseline = profile.get('preparation_sources', {}).get(manifest['commit'])
                        if isinstance(baseline, dict):
                            baseline = baseline.get('root')
                if not baseline:
                    raise ValueError('Fixed bundle validation view missing: '+manifest['commit'])
                baseline = Path(baseline)
                if subprocess.check_output(['git','rev-parse','HEAD'],cwd=baseline,text=True).strip() != manifest['commit']:
                    raise ValueError('Fixed bundle validation view SOURCE differs: '+manifest['commit'])
                target = directory/('workspace' if alias == 'source' else 'workspaces/'+alias)
                owned[alias] = (_clone(baseline,target,view.get('input_git_bundles',[]),cache,
                    view.get('checkout_input_head',True),manifest['files']),manifest)
            _include_declared_outputs(owned,directory,spec['outputs']+spec.get('optional_outputs',[]))
            _include_declared_outputs(owned,directory,spec['inputs'])
            cache.materialize(spec['inputs'],directory)
            for workspace, manifest in owned.values():
                _guard_owned_source(workspace,manifest)
            for locator in spec['input_hashes']:
                raw = subprocess.check_output(['git','show',locator['commit']+':'+locator['path']],cwd=owned['source'][0])
                if hashlib.sha256(raw).hexdigest() != locator['sha256']:
                    raise ValueError('Fresh preparation workspace fixed input differs: '+locator['path'])

    def build_request(self, completion, command, directory, stage, output=None):
        """Freeze a portable leaf request; never invoke its scientific body."""
        if stage not in STAGES:
            raise ValueError('No distributed preparation stage mapping: ' + stage)
        root = Path(getattr(completion, 'preparation_root', self.root)).resolve()
        directory = Path(directory).resolve()
        relative = directory.relative_to(root).as_posix()
        source = completion.stage_source(stage) if hasattr(completion, 'stage_source') else completion.source
        command = list(command)
        script = next((Path(arg) for arg in command[1:]
                       if isinstance(arg, str) and arg.endswith('.py') and Path(arg).is_file()), None)
        if script is None:
            raise ValueError('Preparation leaf requires an explicit fixed script')
        script = script.resolve()
        df_terminal = None
        df_normalizer = None
        if stage == 'DF_SIGNALS' and script.name == 'run_pr_df_candidate_signal_jobs.py' and '--input' in command:
            df_input = read(Path(command[command.index('--input')+1]))
            df_plan = df_input.get('plan', {})
            df_normalizer = df_input.get('runtime', {}).get('normalizer_revision')
            if df_plan.get('terminal_reference') is not None:
                requested = [row for row in df_plan['rounds'] if row['decision'] == 'REQUEST']
                if (len(requested) != 1 or requested[0]['m'] != 13
                        or [row['m'] for row in df_plan['rounds'] if row['decision'] == 'REUSE']
                        != list(range(13))):
                    raise ValueError('DF terminal preparation must request only the new last round')
                df_terminal = requested[0]['job']['physical_context']['physical_candidate']['id']
        if script.name in ('run_pr_pauli_candidate_signals.py','run_pr_pauli_candidate_prices.py')and'--no-push'not in command:
            command.append('--no-push')
        maximum = getattr(completion,'settings',{}).get('df_max_width_source')
        df_max_width = bool(stage=='DF_SIGNALS' and maximum and '--transport-source' in command
                            and command[command.index('--transport-source')+1]==maximum)
        if df_max_width and completion.max_width_source(df_input['configuration'])!=maximum:
            raise ValueError('Max-width transport configuration SOURCE differs')
        df_transport = df_terminal or df_normalizer or df_max_width
        if script.is_relative_to(root) or df_transport:
            script_relative = ('scripts/run_pr_df_candidate_signal_jobs.py' if df_transport
                               else script.relative_to(root).as_posix())
            expected = subprocess.check_output(['git', 'show', source + ':' + script_relative], cwd=root)
            source_files = {script_relative: hashlib.sha256(expected).hexdigest()}
            if not df_transport and expected != script.read_bytes():
                raise ValueError('Preparation leaf script differs from its declared SOURCE')
            for name in ('src/trottertracks/rpe_preparation_paths.py',
                         'scripts/run_pr_pauli_candidate_signals.py',
                         *(['src/trottertracks/pr_df_descriptor_extension.py']
                           if script.name=='run_pr_df_candidate_signal_jobs.py' else []),
                         *(['src/trottertracks/rpe_saved_signal_index.py']
                           if script.name=='run_pr_pauli_candidate_signals.py' else [])):
                item=root/name
                if item.is_file():
                    raw=subprocess.check_output(['git','show',source+':'+name],cwd=root)
                    # Dependencies such as the Pauli price helper are read
                    # from the selected leaf SOURCE in the worker checkout.
                    if raw!=item.read_bytes() and name==script_relative:
                        raise ValueError('Preparation helper differs from fixed SOURCE: '+name)
                    source_files[name]=hashlib.sha256(raw).hexdigest()
        else:
            # PF's source-frozen provider is invoked by the new transport
            # wrapper. The provider manifest is explicit and separately fixed.
            source_files = deepcopy(getattr(completion, 'preparation_source_manifest', {}).get('files', {}))
            terminal_wrapper = Path(__file__).resolve().parents[2]/'scripts'/'run_rpe_pf_horizon_signal.py'
            if not source_files or not (script.is_relative_to(self.root) or
                    (stage == 'PF_SIGNALS' and script == terminal_wrapper)):
                raise ValueError('External preparation wrapper requires a fixed provider manifest')
        inputs, outputs, optional_outputs = {}, ['workspace/' + relative], []
        bindings = {str(root): '@SOURCE@', command[0]: '@PYTHON@'}
        if root != self.root:
            bindings[str(self.root)] = '@TRANSPORT@'
        workspaces = {};workspace_roots = {};runtime_inputs=[]
        pauli_inputs = []
        pauli_locators = []
        df_inputs = []
        df_locators = []
        if script.name == 'run_pr_df_candidate_prices.py' and stage in ('DF_CALIBRATION', 'DF_HOLDOUT'):
            from .pr_candidate_completion import DF_PRICE_FLANKS, accelerate_df_price_command
            command = accelerate_df_price_command(command, stage, DF_PRICE_FLANKS)
            if command[-1] == str(DF_PRICE_FLANKS):
                destination = 'private/df-price-transport/' + DF_PRICE_FLANKS.name
                inputs[destination] = sha = self._blob(DF_PRICE_FLANKS.read_bytes())
                df_inputs.append(dict(path=destination, sha256=sha))
                bindings[str(DF_PRICE_FLANKS)] = '@WORK@/' + destination
        snapshot = self.settings.get('df_snapshot_directory')
        pf_terminal_signal = stage == 'PF_SIGNALS' and script.name == 'run_rpe_pf_horizon_signal.py'
        pf_signal = stage == 'PF_SIGNALS' and (script.name == 'run_pf_preparation_stage.py' or pf_terminal_signal)
        if df_transport:
            if not snapshot or (df_terminal and not re.fullmatch(r'[0-9a-f]{20}', df_terminal)):
                raise ValueError('DF terminal signal requires a fixed physical round and snapshot')
            caller = (maximum if df_max_width else completion.settings['df_signal_caller_source'] if df_normalizer else
                      completion.settings.get('df_terminal_transport_source') or self.settings.get('horizon_transport_source'))
            if caller is None:
                caller = subprocess.check_output(['git','rev-parse','HEAD'],cwd=script.parents[1],text=True).strip()
            if not isinstance(caller, str) or not re.fullmatch(r'[0-9a-f]{40}', caller):
                raise ValueError('DF terminal transport SOURCE must be pinned')
            for name in (script_relative, 'src/trottertracks/pr_df_descriptor_extension.py'):
                raw = subprocess.check_output(['git', 'show', caller + ':' + name], cwd=root)
                if raw != (script.parents[1] / name).read_bytes():
                    raise ValueError('DF terminal transport differs from fixed SOURCE: ' + name)
                destination = 'private/df-terminal-transport/' + name
                inputs[destination] = sha = self._blob(raw)
                df_inputs.append(dict(path=destination, sha256=sha))
                df_locators.append(dict(commit=caller, path=name))
            if df_max_width:
                helpers=('pr_df_max_width.py','pr_rpe_rmse.py','rpe_scope_map.py','pr_rpe_signal_plan.py',
                         'pf_pr_rpe_pipeline.py','h4_df10_rpe_connection.py','c1_stage3_accuracy.py')
                # The passive scheduling code extracts two fixed AST functions
                # and imports the allocation helper relative to its own file.
                for path in (*(f'src/trottertracks/{name}' for name in helpers),
                             'src/trotterlib/h46_m5_plan.py','scripts/h46_new2_core.py','scripts/h46_long_time_signal.py'):
                    raw=subprocess.check_output(['git','show',caller+':'+path],cwd=root)
                    if raw!=(script.parents[1]/path).read_bytes():raise ValueError('Fixed max-width transport differs: '+path)
                    destination='private/df-terminal-transport/'+path
                    inputs[destination]=sha=self._blob(raw)
                    df_inputs.append(dict(path=destination,sha256=sha))
                    df_locators.append(dict(commit=caller,path=path))
            bindings[str(script)] = '@WORK@/private/df-terminal-transport/' + script_relative
            if '--transport-source' not in command:
                command.extend(('--transport-source', caller))
            elif command[command.index('--transport-source') + 1] != caller:
                raise ValueError('DF signal command differs from the fixed caller SOURCE')
            if df_normalizer:
                df_locators.append(dict(commit=df_normalizer, path='src/trottertracks/pr_rpe_signal_provider.py'))
        if pf_terminal_signal:
            if not snapshot:
                raise ValueError('Terminal PF signal requires the fixed H4 snapshot')
            caller_root = script.parents[1]
            caller = self.settings.get('horizon_transport_source')
            if caller is None:
                caller = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=caller_root, text=True).strip()
            if not re.fullmatch(r'[0-9a-f]{40}', caller):
                raise ValueError('PF terminal transport SOURCE must be pinned')
            for name in ('scripts/run_rpe_pf_horizon_signal.py', 'scripts/run_pf_preparation_stage.py',
                         'src/trottertracks/rpe_preparation_paths.py'):
                raw = subprocess.check_output(['git', 'show', caller + ':' + name], cwd=root)
                if raw != (caller_root / name).read_bytes():
                    raise ValueError('Terminal PF transport differs from its fixed SOURCE: ' + name)
                destination = 'private/pf-terminal-transport/' + name
                inputs[destination] = sha = self._blob(raw)
                df_inputs.append(dict(path=destination, sha256=sha))
                df_locators.append(dict(commit=caller, path=name))
            bindings[str(script)] = '@WORK@/private/pf-terminal-transport/scripts/run_rpe_pf_horizon_signal.py'
            command.extend(('--caller-source', caller))
        pf_compile = stage in ('PF_COMPILE', 'PF_HOLDOUT_COMPILE')
        if pf_compile and script.name == 'run_pf_preparation_stage.py':
            runtime_inputs.append(self._pf_native_runtime(script, inputs, bindings, df_inputs))
        if (stage in ('PF_AUDIT', 'PF_HOLDOUT_AUDIT') and script.name == 'run_pf_preparation_stage.py'
                and getattr(completion, 'settings', {}).get('selected_axis_compile')):
            # Use the same selected-axis audit as the controller. Older
            # host transports predate the per-axis raw-receipt check.
            adapter = Path(__file__).resolve().parents[2] / 'scripts/run_pf_preparation_stage.py'
            destination = 'private/pf-transport/scripts/run_pf_preparation_stage.py'
            sha = self._blob(adapter.read_bytes())
            inputs[destination] = sha
            df_inputs.append(dict(path=destination, sha256=sha))
            bindings[str(script)] = '@WORK@/' + destination
        if snapshot and (pf_signal or script.name in (
                'run_pr_df_candidate_prices.py', 'run_pr_df_candidate_signal_jobs.py',
                'derive_pr_df_candidate_descriptor.py')):
            if not pf_signal:
                extension = 'src/trottertracks/pr_df_descriptor_extension.py'
                raw = subprocess.check_output(['git', 'show', source + ':' + extension], cwd=root)
                source_files[extension] = hashlib.sha256(raw).hexdigest()
            df_locators.extend((_DF_SNAPSHOT_INDEX, *_DF_SNAPSHOT_LOADERS))
            self._saved_h4_snapshot(root, snapshot, inputs, bindings, df_inputs)
        if script.name=='run_pr_df_candidate_prices.py' and '-c' in command:
            runtime_inputs.append(dict(df_price_view=command[command.index(str(script))+1]))
        if script.name=='derive_pr_df_candidate_descriptor.py' and '--signal-view' in command:
            runtime_inputs.append(dict(signal_view=command[command.index('--signal-view')+1]))
        # Original JSON remains byte-for-byte unchanged. Only runtime path
        # access is resolved by rpe_preparation_paths in the existing provider.
        for index,arg in enumerate(command):
            if index and command[index-1]in('--output','--output-directory','--run-dir'):continue
            if not isinstance(arg, str) or not arg.endswith('.json'):
                continue
            path = Path(arg)
            if not path.is_file():
                continue
            if path.resolve().is_relative_to(root):
                inputs['workspace/' + path.resolve().relative_to(root).as_posix()] = self._blob(path.read_bytes())
            value = read(path)
            runtime_inputs.append(value)
            if not isinstance(value, dict):
                continue
            workspace = value.get('workspace')
            if stage == 'PAULI_SIGNALS' and value.get('pauli_saved_inputs'):
                from .rpe_pauli_saved_inputs import INDEX_PATH, INDEX_SOURCE, PROVIDER_SOURCE, provider_files, saved_inputs
                paths = value['pauli_saved_inputs']
                helper = 'src/trottertracks/rpe_pauli_saved_inputs.py'
                raw = subprocess.check_output(['git', 'show', source + ':' + helper], cwd=root)
                if raw != (root/helper).read_bytes():
                    raise ValueError('Pauli saved-input helper differs from fixed SOURCE')
                source_files[helper] = hashlib.sha256(raw).hexdigest()
                pauli_locators.extend(dict(commit=PROVIDER_SOURCE, path=name)
                                      for name in provider_files('run_pauli_signal_entry.py'))
                pauli_locators.append(dict(commit=INDEX_SOURCE, path=INDEX_PATH))
                for original, sha in saved_inputs(Path(workspace), paths).items():
                    raw = Path(paths[original]).read_bytes()
                    if self._blob(raw) != sha:
                        raise ValueError('Pauli saved input changed during capture: ' + original)
                    name = 'private/pauli-saved-inputs/' + Path(original).name
                    inputs[name] = sha
                    bindings[original] = '@WORK@/' + name
                    pauli_inputs.append(dict(path=name, sha256=sha))
            if script.name == 'run_pr_pauli_candidate_signals.py' and value.get('workspace') and 'saved_packets' in value:
                pauli_locators.extend(pauli_saved_kernel_inputs(root, value['saved_packets']))
            if script.name == 'run_pr_pauli_candidate_prices.py' and value.get('workspace'):
                pauli_locators.extend(pauli_price_guard_inputs(value))
            if workspace:
                workspace = Path(workspace).resolve()
                if workspace != root:
                    alias = 'provider'
                    commit = source
                    files={}
                    for name in ('scripts/run_pauli_signal_entry.py','scripts/prepare_pauli_calibration.py',
                                 'scripts/run_pauli_split_cal.py','scripts/pauli_split_cal_audit_request.py',
                                 'scripts/pauli_ld2_step_audit.py','scripts/pauli_ld3_cal_audit.py',
                                 'src/trottertracks/rpe_preparation_paths.py',
                                 'src/trottertracks/pauli_signal_entry.py','src/trottertracks/pauli_calibration_request.py'):
                        item=self.root/name
                        if item.is_file():
                            raw=subprocess.check_output(['git','show',commit+':'+name],cwd=self.root)
                            # This checkout is the controller. The provider
                            # workspace is cloned from the fixed worker view,
                            # which checks the pinned helper hash below.
                            if name!='src/trottertracks/rpe_preparation_paths.py' and raw!=item.read_bytes():
                                raise ValueError('Provider workspace SOURCE differs: '+name)
                            files[name]=hashlib.sha256(raw).hexdigest()
                    if not files:raise ValueError('Preparation workspace has no fixed provider source')
                    if pauli_inputs:
                        for name in provider_files('run_pauli_signal_entry.py'):
                            raw = subprocess.check_output(['git','show',commit+':'+name],cwd=root)
                            files[name] = hashlib.sha256(raw).hexdigest()
                    workspaces[alias] = dict(source=dict(commit=commit, files=files), input_git_bundles=self._bundle(workspace,commit),checkout_input_head=False)
                    workspace_roots[alias] = workspace
                    bindings[str(workspace)] = '@WORKSPACE:' + alias + '@'
                    leaf_directory = value.get('directory')
                    if leaf_directory:
                        target = workspace / artifact_path(leaf_directory)
                        remote = 'workspaces/' + alias + '/' + leaf_directory
                        outputs.append(remote)
                        if target.exists():
                            inputs.update(self._capture(target, remote))
            private = value.get('private_root')
            if private:
                bindings[str(Path(private).resolve())] = '@WORK@/private'
                optional_outputs.append('private')
                if Path(private).exists():
                    inputs.update(self._capture(Path(private), 'private'))
        inputs.update(self._capture(directory, 'workspace/' + relative))
        original_documents=[*pauli_locators, *df_locators]
        if df_max_width:
            # The command checks its own frozen request via git show. This
            # locator must be explicit when a shared bundle predates the
            # candidate, not merely present in a mutable controller HEAD.
            request_file=Path(command[command.index('--input')+1]).resolve()
            inputs_commit=command[command.index('--inputs-commit')+1]
            request_locator=dict(commit=inputs_commit,path=request_file.relative_to(root).as_posix())
            raw=subprocess.check_output(['git','show',inputs_commit+':'+request_locator['path']],cwd=root)
            if raw!=request_file.read_bytes():raise ValueError('Max-width frozen signal request bytes differ')
            original_documents.append(request_locator)
        if script.name == 'derive_pr_df_candidate_descriptor.py':
            original_documents.append(dict(commit='7f4c9a171a1d4caeaeaee6c443217edec761f4c9',
                                           path='scripts/h46_new2_core.py'))
        if script.name == 'run_pr_df_candidate_prices.py':
            # The fixed split compiler loads these with git show from inside
            # the sparse attempt checkout, even for audit-only commands.
            original_documents.extend(_DF_SPLIT_PRICE_IMPORTS)
            original_documents.extend(_DF_PRICE_BACKEND)
        if pf_terminal_signal:
            original_documents.extend(_PF_SIGNAL_KERNEL_INPUTS)
            frozen = read(Path(command[command.index('--request')+1]))
            locator = frozen['configuration']
            config = json.loads(subprocess.check_output(
                ['git', 'show', locator['commit'] + ':' + locator['path']], cwd=root))
            for key in locator.get('selector', []):
                config = config[key]
            prefix = config['spec']['L_D']
            if prefix < 10:
                original_documents.append(dict(commit='5e6f9fd214631f7ce92fad0e9c0d273dfb479533',
                    path=f'docs/research/truncated_pf_t1_results_20261114/run01/prefix_{prefix}.json'))
        if script.name == 'run_pf_preparation_stage.py':
            if stage == 'PF_SIGNALS':
                original_documents.extend(_PF_SIGNAL_KERNEL_INPUTS)
                payload = command[command.index('--input')+1]
                original_documents.extend(json.loads(subprocess.check_output(
                    [command[0], '-B', '-c', _PF_STEP_INPUTS, str(root), payload], cwd=root)))
            elif stage in ('PF_COMPILE', 'PF_HOLDOUT_COMPILE', 'PF_AUDIT', 'PF_HOLDOUT_AUDIT'):
                original_documents.extend(_PF_NATIVE_GUARD_INPUTS)
                if pf_compile:
                    original_documents.extend(_PF_COMPILE_ADAPTER_INPUTS)
            if stage in ('PF_AUDIT', 'PF_HOLDOUT_AUDIT', 'PF_MODEL_FREEZE', 'PF_FINISH'):
                original_documents.extend(_PF_PRICE_GUARD_INPUTS)
        for name,sha in inputs.items():
            if name.endswith('.json'):
                try:document=json.loads(self.executor.artifact_cache.path(sha).read_bytes())
                except (ValueError,UnicodeError):raise ValueError('Original preparation JSON is incomplete or invalid: '+name)
                original_documents.append(document)
                if stage == 'PAULI_AUDIT' and isinstance(document, dict):
                    original_documents.extend(pauli_audit_raw_inputs(name, document))
        df_raw_commit = None
        if (stage in ('DF_MODEL_FREEZE', 'DF_HOLDOUT_COMPARE')
                and script.name == 'run_pr_df_candidate_prices.py' and '--raw-commit' in command):
            raw = command[command.index('--raw-commit') + 1]
            if not re.fullmatch(r'[0-9a-f]{40}', raw):
                raise ValueError('DF audit requires an immutable raw commit')
            df_raw_commit = (raw, 'workspace/' + relative + '/')
        input_hashes,working_hashes=fixed_input_hashes(root,original_documents,inputs,df_raw_commit=df_raw_commit)
        working_hashes.extend(pauli_inputs)
        working_hashes.extend(df_inputs)
        required_views=collect_required_views(runtime_inputs)
        if stage in ('PAULI_COMPILE', 'PAULI_AUDIT') and script.name=='run_pr_pauli_candidate_prices.py':
            required_views.update(pauli_calibration_view(root))
        required_paths=collect_required_paths(runtime_inputs,self.settings.get('controller_paths'))
        with self.lock:
            aliases = dict(self.aliases)
        bindings.update(used_path_aliases(aliases, bindings, original_documents, command))
        checkout_input_head=not script.is_relative_to(root) and not df_max_width
        if self.settings.get('stable_input_bundle') is True and (pf_terminal_signal or df_transport):
            # These paths already force checkout_input_head=False below.
            checkout_input_head=False
        root_bundles=self._bundle(root,source,input_hashes,
                                 tracked_workspace=checkout_input_head,
                                 shared_base=df_max_width,provider=stage)
        if script.is_relative_to(root) or pf_terminal_signal or df_transport:
            # This bundle supplies locators for frozen inputs. Checking out
            # its HEAD would substitute controller code for the provider.
            root_bundles=[dict(bundle,checkout_input_head=False)for bundle in root_bundles]
        # Pauli's publication workspace has a separate object database. It
        # reads the controller's newly committed configuration too; importing
        # only the prior provider branch would leave that locator unavailable.
        for workspace in workspaces.values():
            combined=root_bundles+workspace['input_git_bundles']
            workspace['input_git_bundles']=list({item['sha256']:item for item in combined}.values())
        requirements,compiler_bindings=contract_runtime(runtime_inputs,script,self.settings.get('runtime_requirements', {}).get(STAGES[stage], {}),provider=stage)
        pauli_signals = stage == 'PAULI_SIGNALS' and script.name == 'run_pr_pauli_candidate_signals.py'
        if pauli_signals or stage in ('PF_SIGNALS', 'PF_COMPILE', 'PF_HOLDOUT_COMPILE'):
            pins = subprocess.check_output(['git', 'show', source+':requirements.txt'], cwd=root, text=True)
            packages = requirements['packages']
            for line in pins.splitlines():
                name, separator, version = line.partition('==')
                if separator and name in ('openfermionpyscf', 'openfermion', 'pyscf'):
                    if packages.get(name) not in (None, version):
                        raise ValueError('Preparation runtime version conflicts with fixed SOURCE: '+name)
                    packages[name] = version
            if pauli_signals:
                requirements['imports'] = sorted(set(requirements.get('imports', ())) | {'openfermionpyscf'})
        if (stage == 'DF_SIGNALS' and not df_transport and script.name == 'run_pr_df_candidate_signal_jobs.py'
                and not getattr(completion, 'settings', {}).get('df_signal_source')):
            position = command.index(str(script))
            command[position:position] = ['-c', _DF_SIGNAL_BOOTSTRAP]
        prep = dict(stage=STAGES[stage], provider=stage, command=list(command),
                    source=dict(commit=source, files=source_files), source_workspace=True,
                    controller_root=str(root),
                    input_git_bundles=root_bundles, workspaces=workspaces,
                    path_bindings=bindings, inputs=inputs, outputs=list(dict.fromkeys(outputs)),
                    input_hashes=input_hashes,working_input_hashes=working_hashes,
                    required_views=required_views,
                    required_paths=required_paths,
                    optional_outputs=optional_outputs, return_git_delta=True, stage_inputs=False,
                    runtime_requirements=requirements,compiler_bindings=compiler_bindings)
        binding_publication = (stage == 'PAULI_COMPILE' and
                               (getattr(completion, 'settings', {}) or {}).get('pauli_price_wrapper_source') == source)
        if binding_publication:
            prep['publish_pauli_binding'] = True
        if stage == 'PAULI_SIGNALS' or binding_publication or pf_compile:
            declared = [*prep['outputs'], *prep.get('optional_outputs', [])]
            # The PF workspace also expands the index containing fixed
            # references when adding the provider's declared output paths.
            extra_directories = ('references',) if pf_compile else ()
            prep['input_git_bundles'].append(self._workspace_bundle(
                root, prep, Path('workspace'), declared, extra_directories=extra_directories))
            for alias, view in workspaces.items():
                view['input_git_bundles'].append(self._workspace_bundle(
                    workspace_roots[alias], view, Path('workspaces')/alias, declared,
                    extra_directories=extra_directories))
        if stage in ('DF_CALIBRATION', 'DF_HOLDOUT') and '--input' in command:
            if any(name.startswith('private/df-price-transport/') for name in inputs):
                # A new request pins the portable codec as an ordinary input.
                # Existing submitted markers retain their original request.
                prep['df_event_layout'] = 'DICTIONARY_PAIRED_V1'
            frozen = read(Path(command[command.index('--input')+1]))
            if frozen.get('parallel_axes') is True:
                if frozen['source'] != source:
                    raise ValueError('Parallel DF price input SOURCE differs from its assigned runner')
                prep['parallel_df_price_axes'] = True
        contract = dict(source=source, stage=stage, automatic_retry=False)
        if pf_terminal_signal:
            frozen = read(Path(command[command.index('--request')+1]))
            if not re.fullmatch(r'[0-9a-f]{20}', frozen.get('physical_id', '')):
                raise ValueError('Terminal signal needs a fixed PF physical round')
            prep['horizon_round'] = dict(m=13, physical_id=frozen['physical_id'])
            contract.update(rounds=[13], reference_state_actions=2, signal_calls=1)
        elif df_terminal:
            prep['horizon_round'] = dict(m=13, physical_id=df_terminal)
            contract.update(rounds=[13], reference_state_actions=0, signal_calls=1)
        self._validate_bundle_inputs(prep)
        return dict(schema='RPE_PREPARATION_REQUEST_V1', route=dict(kind='preparation'), preparation=prep,
                    original_contract=contract)

    def _capture(self, path, prefix):
        path = Path(path).resolve()
        files = sorted(path.rglob('*')) if path.is_dir() else [path]
        result = {}
        for item in files:
            if item.is_symlink():
                raise ValueError('Preparation payload must not contain symlinks')
            if not item.is_file() or '.git' in item.relative_to(path).parts:
                continue
            name = prefix + ('/' + item.relative_to(path).as_posix() if path.is_dir() else '')
            artifact_path(name)
            result[name] = self._blob(item.read_bytes())
        return result

    def inspect_prior_owner(self, command, directory, stage):
        """Read selected producer output markers before cloning can hide them.

        This is limited to the exact original request/output roots, not a
        global archive/process search. Existing distributed attempts are
        reconciled by JobStore instead of this pre-submission observation.
        """
        directory=Path(directory).resolve();roots=[directory];observations=[]
        def state(path):
            try:return read(path)
            except (OSError,ValueError):return {}  # An owner may still be publishing this terminal.
        for argument in command:
            if not isinstance(argument,str)or not argument.endswith('.json'):continue
            path=Path(argument)
            if not path.is_file():continue
            value=read(path)
            if isinstance(value,dict)and value.get('workspace')and value.get('directory'):
                roots.append(Path(value['workspace']).resolve()/artifact_path(value['directory']))
        for root in dict.fromkeys(roots):
            if not root.exists():continue
            # A protected STOP belongs to this exact selected producer output.
            for path in root.rglob('STOP*.json'):
                if not path.is_file():continue
                observations.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),state='STOP'))
            legacy=root/(stage+'.STARTED.json')
            markers=list(root.rglob('*_STARTED.json'))
            if legacy.is_file():markers.append(legacy)
            started=root/'started'
            if started.is_dir():markers.extend(started.glob('*.json'))
            for path in dict.fromkeys(markers):
                companion=path.with_name(path.stem+'_RETURN.json')
                terminals=[]
                if path.name=='SIGNALS_STARTED.json':
                    companion=path.with_name('SIGNALS_RETURN.json');terminals=[path.parent/'run01/result.json']
                elif path.name=='CALIBRATION_STARTED.json':
                    terminals=list(path.parent.glob('**/run01/RESULT.json'))
                elif path.name=='AUDIT_STARTED.json':terminals=list(path.parent.glob('**/PRICE_26AXIS.json'))
                elif path==legacy:companion=root/(stage+'.RETURN.json')
                elif path.parent.name=='started':terminals=[path.parent.parent/'terminal'/path.name]
                completed=False
                if companion.is_file():
                    result=state(companion)
                    if result.get('returncode')==0 and result.get('signal')is None:completed=True
                    elif result.get('returncode')is not None or result.get('signal')is not None:
                        observations.append(dict(path=str(companion),sha256=hashlib.sha256(companion.read_bytes()).hexdigest(),state='FAILED'))
                for terminal in terminals:
                    if not terminal.is_file():continue
                    status=state(terminal).get('status','')
                    completed=completed or status in('VALID','ENTRY_CARRIERS_COMPLETE_NOT_RPE_QUALIFIED','PRICE_RULE_SUPPORTED_MODEL_REFERENCE')or(status.startswith('FINITE_LD')and status.endswith('_CAL_PATHS_COMPLETE'))
                observations.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                         state='COMPLETED_ARTIFACTS_AVAILABLE'if completed else'RUNNING_OR_UNKNOWN_ORIGINAL_OWNER'))
        return dict(scope='EXACT_ORIGINAL_SELECTED_PRODUCER_OUTPUTS_ONLY',observations=observations,
            failed=any(row['state']in('STOP','FAILED')for row in observations),
            pending=any(row['state']=='RUNNING_OR_UNKNOWN_ORIGINAL_OWNER'for row in observations))

    def _result_bundle_inputs(self, view, result):
        if not view.get('checkout_input_head', True):
            raise ValueError('Result claims a Git input HEAD for a fixed checkout')
        cache = self.executor.artifact_cache
        inputs = [row for row in view.get('input_git_bundles', []) if not row.get('workspace_objects')]
        chosen = next((row for row in reversed(inputs) if row.get('checkout_input_head', True)), None)
        if not chosen or cache.missing(row['sha256'] for row in inputs):
            raise ValueError('Missing original Git inputs of returned preparation delta')
        heads = subprocess.check_output(
            ['git','bundle','list-heads',str(cache.path(chosen['sha256'])),chosen.get('ref','HEAD')],
            cwd=self.root,text=True,
        ).splitlines()
        if len(heads)!=1 or heads[0].split()[0]!=result['input_head']:
            raise ValueError('Returned Git delta does not bind the assigned input HEAD')
        return [(cache.path(row['sha256']),row.get('ref','HEAD')) for row in inputs]

    def _import_result(self, result, request, *, inherited_workspace=None):
        artifacts = result.get('artifacts') or result.get('artifact_manifest') or {}
        directory = Path(result['artifact_directory'])
        from .pr_candidate_completion import put
        with self.lock:
            for original,actual in result.get('resolved_path_bindings',{}).items():
                if original in request['preparation'].get('path_bindings',{}):
                    canonical=self.aliases.get(original,original)
                    self.aliases[actual]=canonical
            put(self.alias_path,self.aliases)
        # Import scientific Git objects first, preserving original source-bound
        # locators from the remote provider without checkout/HEAD replacement.
        workspaces = {'workspace': Path(request['preparation'].get('controller_root', self.root))}
        if inherited_workspace is not None:
            original, destination = inherited_workspace
            if workspaces['workspace'].resolve() != original.resolve() or request['preparation'].get('workspaces'):
                raise ValueError('Inherited PF result names another producer workspace')
            workspaces['workspace'] = destination
        for alias, descriptor in request['preparation'].get('workspaces', {}).items():
            old = next(key for key, value in request['preparation']['path_bindings'].items()
                       if value == '@WORKSPACE:' + alias + '@')
            workspaces['workspaces/' + alias] = Path(old)
        for bundle in result.get('git_bundles', []):
            alias = bundle.get('workspace', 'workspace')
            destination = workspaces[alias]
            view = request['preparation'] if alias=='workspace' else request['preparation']['workspaces'][alias.split('/',1)[1]]
            if bundle['base']!=view['source']['commit']:
                raise ValueError('Returned Git bundle SOURCE differs')
            inputs = self._result_bundle_inputs(view,bundle) if 'input_head' in bundle else []
            path = None
            if 'path' in bundle:
                path = directory / artifact_path(bundle['path'])
                key = artifacts[bundle['path']]
                if hashlib.sha256(path.read_bytes()).hexdigest()!=key:
                    raise ValueError('Returned scientific Git bundle changed')
            elif not inputs or bundle['head']!=bundle['input_head']:
                raise ValueError('Returned Git result is missing its bundle')
            else:
                key = digest(bundle)
            # The central optimizer's GitReader can be a different repository
            # from an owned provider clone. Both must resolve the original
            # returned commit; neither working HEAD is changed.
            namespace = 'refs/rpe-preparation/' + digest(str(self.executor.run_dir))[:16]
            result_ref = namespace + '/' + key
            source_ref = bundle.get('ref', 'HEAD')
            if path is not None:
                heads = subprocess.check_output(['git','bundle','list-heads',str(path),source_ref],text=True).splitlines()
                if len(heads) != 1 or heads[0].split()[0] != bundle['head']:
                    raise ValueError('Returned Git bundle HEAD differs')

            def import_tip(repository):
                command = (['rev-parse','--verify',bundle['input_head']+'^{commit}'] if path is None
                           else ['fetch','--no-tags','--no-write-fetch-head',str(path),source_ref+':'+result_ref])
                return subprocess.run(['git',*command],cwd=repository,capture_output=True,text=True)

            with self.lock:
                for repository in dict.fromkeys([destination,self.root]):
                    fetched = import_tip(repository)
                    if fetched.returncode and inputs:
                        for file, ref in inputs:
                            subprocess.run(['git','fetch','--no-tags','--no-write-fetch-head',str(file),
                                            ref+':'+namespace+'/input/'+file.name],
                                           cwd=repository,check=True,capture_output=True)
                        fetched = import_tip(repository)
                    fetched.check_returncode()
                    if path is None:
                        tip = fetched.stdout.strip()
                        if tip != bundle['head']:
                            raise ValueError('Returned Git bundle HEAD differs')
                        subprocess.run(['git','update-ref',result_ref,tip],cwd=repository,check=True)
        private = next((old for old,value in request['preparation'].get('path_bindings',{}).items()if value=='@WORK@/private'),None)
        if private:workspaces['private']=Path(private)
        for name, sha in artifacts.items():
            if request['preparation'].get('inputs',{}).get(name)==sha:
                continue  # Another independent branch may now have newer output.
            relative = artifact_path(name)
            owner = next((prefix for prefix in sorted(workspaces, key=len, reverse=True)
                          if name.startswith(prefix + '/')), None)
            if owner is None or name.endswith('.bundle'):
                continue
            destination = workspaces[owner] / artifact_path(name[len(owner) + 1:])
            raw = (directory / relative).read_bytes()
            if hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError('Returned preparation artifact hash differs')
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Only this run's declared provider output roots are copied back.
            # Prior immutable results remain protected by the provider's gates.
            if not destination.exists() or destination.read_bytes() != raw:
                destination.write_bytes(raw)

    def call(self, completion, command, directory, stage, output=None, science=True, *, wait=True):
        from .pr_candidate_completion import Held, PreparationWait, put
        directory = Path(directory)
        marker = directory / (stage + '.DISTRIBUTED.json')
        inherited_workspace = None
        if marker.exists():
            submitted = self.repair_compile_context(directory, stage, read(marker))
            repair = self.repair_max_width_signal_input(directory,stage,submitted)
            if repair is None:
                repair = self.redeliver_pf_compile(completion, command, directory, stage, output, submitted)
            if repair is None and submitted.get('job_id'):
                repair = self._transport_repair(directory, stage, submitted)
            if repair is not None:
                from .rpe_preparation_queue import preparation_job_id

                archive = marker.with_name(stage+'.ORIGINAL_TRANSPORT_ATTEMPT.json')
                if not archive.exists():
                    put(archive, submitted)
                # Publish the successor identity before submitting it. A lost
                # response or controller interruption reuses this exact request.
                submitted = dict(repair, job_id=preparation_job_id(
                    repair['request'], dependencies=self._dependencies(directory, stage)))
                put(marker, submitted)
            request = submitted['request']
            original = getattr(completion, 'inherited_preparation_root', None)
            if original is not None and Path(request['preparation']['controller_root']).resolve() == original.resolve():
                inherited_workspace = original, Path(completion.preparation_root)
            portable = 'private/pf-transport/scripts/run_pf_preparation_stage.py'
            if (stage in ('PF_COMPILE', 'PF_HOLDOUT_COMPILE')
                    and self.executor.distributed.get('pf_legacy_compile_hosts')
                    and portable not in request['preparation'].get('inputs', {})):
                job = submitted['job_id']
                with self.executor.state.cv:
                    row = self.executor.store.row(job)
                    if row is None or (row['state'] == 'CANCELLED' and row['attempt_id'] is None):
                        try:
                            replacement = self.build_request(completion, command, directory, stage, output)
                        except (OSError, ValueError, subprocess.CalledProcessError) as error:
                            raise PreparationWait('DISTRIBUTED_PREPARATION_OPERATIONAL_HOLD:'+stage+
                                       ':INPUT_OR_SOURCE_METADATA_UNAVAILABLE:'+str(error)) from error
                        if portable not in replacement['preparation']['inputs']:
                            raise ValueError('Replacement PF compile has no fixed portable adapter')
                        archive = marker.with_name(stage+'.LEGACY_UNSTARTED.json')
                        if not archive.exists():
                            put(archive, submitted)
                        if job in self.executor.preparation_pending:
                            del self.executor.preparation_pending[job]
                            self.executor._save_preparation_pending()
                        request = replacement
                        submitted = dict(request=request, source=request['preparation']['source']['commit'],
                                         automatic_retry=False)
        else:
            prior=self.inspect_prior_owner(command,directory,stage)
            put(directory/(stage+'.PRIOR_OWNER.json'),prior)
            if prior['failed']:
                raise Held('STAGE_FAILED:'+stage+':ORIGINAL_OWNER_STOP_OR_FAILURE:NO_AUTOMATIC_RETRY:'+str(prior))
            if prior['pending']:
                raise PreparationWait('DISTRIBUTED_PREPARATION_OPERATIONAL_HOLD:'+stage+':ORIGINAL_OWNER_RUNNING_OR_UNKNOWN:'+str(prior))
            try:request = self.build_request(completion, command, directory, stage, output)
            except (OSError,ValueError,subprocess.CalledProcessError)as error:
                raise PreparationWait('DISTRIBUTED_PREPARATION_OPERATIONAL_HOLD:'+stage+':INPUT_OR_SOURCE_METADATA_UNAVAILABLE:'+str(error))from error
            submitted = dict(request=request, source=request['preparation']['source']['commit'], automatic_retry=False)
        # A Q3 approval/budget does not open signal/compile scope. The round
        # must separately bound preparation tasks and retain leaf input caps.
        budget = self.settings.get('budgets', {}).get(STAGES[stage], 0)
        dependencies = self._dependencies(directory, stage)
        job_id = self.executor.submit_preparation(request, budget=budget, approved=self.approved,
            contract=request['original_contract'], dependencies=dependencies, expected_job_id=submitted.get('job_id'))
        if stage == 'DF_SIGNALS' and self.max_width_signal_locator(request['preparation']) is not None:
            # The queue may reuse an older physical owner even when the
            # transport manifest changed. Keep its exact request for intake.
            row = self.executor.store.row(job_id)
            pending = getattr(self.executor, 'preparation_pending', {}).get(job_id)
            owned = row['request'] if row else pending['request'] if pending else None
            if owned is not None and owned.get('request_id') != request.get('request_id'):
                archive = marker.with_name(stage+'.TRANSPORT_ALIAS.json')
                if (marker.exists() and submitted.get('job_id') not in (None, job_id)
                        and not archive.exists()):
                    put(archive, submitted)
                request = owned
                submitted = dict(request=owned, source=owned['preparation']['source']['commit'],
                                 automatic_retry=False)
        put(marker, dict(submitted, job_id=job_id))
        result = self.executor.preparation_result(job_id, wait=wait, adopt=False)
        if result is None and not wait:
            raise PreparationWait('DISTRIBUTED_PREPARATION_IN_PROGRESS:'+stage+':'+job_id, job_id=job_id)
        if not result or result.get('status') != 'PREPARATION_COMPLETE':
            terminal=result or{}
            while terminal.get('status')=='PREPARATION_DEPENDENCY_HOLD'and isinstance(terminal.get('predecessor'),dict):
                terminal=terminal['predecessor']
            if terminal.get('status') == 'PREPARATION_FAILED':
                raise Held('STAGE_FAILED:'+stage+':DISTRIBUTED_'+terminal['status']+':NO_AUTOMATIC_RETRY:'+str(result))
            if terminal.get('reason') == 'WORKER_RESULT_MISMATCH':
                raise Held('PREPARATION_RESULT_MISMATCH:'+stage+':'+job_id)
            hold = Held if terminal.get('status') in ('PREPARATION_HOLD', 'PREPARATION_BUDGET_OR_PERMISSION_HOLD',
                                                      'PREPARATION_STOPPED_BEFORE_RESERVATION') else PreparationWait
            raise hold('DISTRIBUTED_PREPARATION_OPERATIONAL_HOLD:' + stage + ':' + str(result))
        receipt = self._adopt_received_stage(completion, directory, stage, job_id, request, result,
                                             inherited_workspace=inherited_workspace)
        return read(output) if output and Path(output).exists() else receipt

    def import_received_stage(self, completion, marker, stage):
        """Restore an accepted stage from this run without submitting a request."""
        submitted = read(marker)
        job_id = submitted['job_id']
        directory = marker.parent
        if not self.received_stage(directory, stage, job_id):
            return False
        row = self.executor.store.row(job_id)
        result = self.executor.preparation_result(job_id, adopt=False)
        if not result or result.get('status') != 'PREPARATION_COMPLETE':
            return False
        self._adopt_received_stage(completion, directory, stage, job_id, row['request'], result)
        return True

    def _adopt_received_stage(self, completion, directory, stage, job_id, request, result, *, inherited_workspace=None):
        with self.lock:
            lock=self.stage_return_locks.setdefault((str(directory.resolve()),stage),threading.RLock())
        with lock:
            return self._adopt_received_stage_locked(completion,directory,stage,job_id,request,result,inherited_workspace=inherited_workspace)

    def _adopt_received_stage_locked(self, completion, directory, stage, job_id, request, result, *, inherited_workspace=None):
        from .pr_candidate_completion import Held, PreparationWait, put
        returned=directory/(stage+'.RETURN.json')
        if returned.is_file():
            receipt=read(returned)
            if (receipt.get('job_id')!=job_id or receipt.get('returncode')!=0 or receipt.get('signal') is not None
                    or receipt.get('source')!=request['preparation']['source']['commit']):
                raise Held('PREPARATION_RESULT_MISMATCH:'+stage+':'+job_id+':RETURN_OWNER')
            self.executor.adopt_preparation(job_id)
            return receipt
        if problem := df_freeze_output_problem(request, result):
            raise Held('PREPARATION_RESULT_MISMATCH:'+stage+':'+job_id+':'+problem)
        try:
            if inherited_workspace is None:
                self._import_result(result, request)
            else:
                self._import_result(result, request, inherited_workspace=inherited_workspace)
        except FileNotFoundError as error:
            put(directory/(stage+'.IMPORT_WAIT.json'), dict(job_id=job_id, missing_path=error.filename))
            raise PreparationWait('PREPARATION_RESULT_ARTIFACTS_MISSING:'+stage+':'+job_id+':'+str(error)) from error
        except ValueError as error:
            raise Held('PREPARATION_RESULT_MISMATCH:'+stage+':'+job_id+':'+str(error)) from error
        receipt = dict(returncode=0, signal=None, source=request['preparation']['source']['commit'], job_id=job_id,
                       transport='distributed', automatic_retry=False, preparation_calls=result.get('preparation_calls', 0))
        heads = [bundle['head'] for bundle in result.get('git_bundles', [])
                 if bundle.get('workspace', 'workspace') == 'workspace']
        if heads:
            receipt['git_heads'] = heads
        put(returned,receipt)
        self.executor.adopt_preparation(job_id)
        completion.log(event='DISTRIBUTED_STAGE_RETURN', stage=stage, job_id=job_id)
        return receipt
