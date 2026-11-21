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

from .rpe_preparation_artifacts import artifact_path


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

    @staticmethod
    def handles(stage):
        return stage in STAGES

    @staticmethod
    def _dependencies(directory, stage):
        paths = (directory/(name+'.DISTRIBUTED.json') for name in DEPENDENCIES.get(stage, ()))
        return [read(path)['job_id'] for path in paths if path.exists()]

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
                    action = ('CHECK_FAILED_ATTEMPT_ARTIFACTS' if artifacts else 'PRESERVE_FAILED_ATTEMPT')
            elif row['attempt_id']:
                action = 'RECOVER_ORIGINAL_ATTEMPT'
            else:
                action = 'RECHECK_RESERVATION'
            return dict(result, action=action)
        except (OSError, KeyError, TypeError, ValueError, sqlite3.Error):
            return unmatched

    def hold_evidence(self, directory, reason):
        prefix, _, payload = reason.partition(':')
        stage, _, detail = payload.partition(':')
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
        """Recover the same frozen request when a previous stop left it unclaimed."""
        prefix, _, payload = reason.partition(':')
        stage, _, detail = payload.partition(':')
        if prefix != 'DISTRIBUTED_PREPARATION_OPERATIONAL_HOLD' or stage not in STAGES:
            return False
        try:
            result = ast.literal_eval(detail)
            if not isinstance(result, dict) or result.get('status') != 'PREPARATION_STOPPED_BEFORE_RESERVATION':
                return False
            job = result['job_id']
            if self.executor._user_stop_requested() or self.executor.state.failure is not None:
                return False
            row = self.executor.store.row(job)
            if row is not None and (row['state'] != 'QUEUED' or row['attempt_id'] is not None):
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
        if prefix != 'DISTRIBUTED_PREPARATION_OPERATIONAL_HOLD' or stage not in STAGES:
            return False
        if not detail.startswith("{'status':"):
            return False
        try:
            result = ast.literal_eval(detail)
        except (SyntaxError, ValueError):
            return False
        return (isinstance(result, dict) and result.get('status') in (
                    'PREPARATION_STOPPED_BEFORE_RESERVATION', 'PREPARATION_RESULT_ARTIFACTS_MISSING',
                    'UNKNOWN', 'HOST_ELIGIBILITY_HOLD', 'PREPARATION_DEPENDENCY_HOLD')
                and self.received_stage(directory, stage, result.get('job_id')))

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
        constants = runpy.run_path(str(helper))
        origin = constants['PF_NATIVE_ORIGIN_CONFIG']
        for file, relative in ((adapter, 'scripts/run_pf_preparation_stage.py'),
                               (helper, 'src/trottertracks/rpe_preparation_paths.py')):
            destination = 'private/pf-transport/'+relative
            sha = self._blob(file.read_bytes())
            inputs[destination] = sha
            working_hashes.append(dict(path=destination, sha256=sha))
        bindings[str(script)] = '@WORK@/private/pf-transport/scripts/run_pf_preparation_stage.py'

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

    def _bundle(self, root, source, input_hashes=(), *, tracked_workspace=False):
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        bundles = []
        namespace = 'refs/rpe-preparation/input-bundles/'
        # A worker with only its fixed SOURCE cannot rely on a moving
        # controller HEAD. In this mode carry that history in the same bundle
        # as the inputs rather than making it a separate prerequisite.
        fixed_source_only = bool(input_hashes) and self.settings.get('bundle_fixed_source_only') is True
        # Send the shared checkout history once per worker cache. Retain that
        # history for provider code that resolves hard-coded Git references.
        if not fixed_source_only and head != source and subprocess.check_output(
            ['git', 'rev-list', '-1', head, '^' + source], cwd=root, text=True,
        ).strip():
            ref = namespace + 'base-' + head
            with self.lock:
                subprocess.run(['git', 'update-ref', ref, head], cwd=root, check=True)
            bundles.append(self._pack_bundle(root, ref, [source], (), not input_hashes))
        if not input_hashes:
            return bundles

        # One advertised endpoint reaches the declared producer commits, even
        # on separate branches, and preserves the same tracked workspace tree.
        commits = sorted({row['commit'] + '^{commit}' for row in input_hashes})
        resolved = subprocess.check_output(['git', 'rev-parse', *commits], cwd=root, text=True).split()
        parents = [head, *sorted(set(resolved) - {head})]
        tree, date = subprocess.check_output(
            ['git', 'show', '-s', '--format=%T %aI', head], cwd=root, text=True,
        ).split()
        args = ['git', '-c', 'user.name=HIROMU1015', '-c', 'user.email=u411612g@ecs.osaka-u.ac.jp',
                'commit-tree', tree]
        for parent in parents:
            args.extend(['-p', parent])
        tip = subprocess.check_output(
            args + ['-m', 'Make declared preparation input commits reachable'],
            cwd=root, text=True, env=dict(os.environ, GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date),
        ).strip()
        ref = namespace + 'inputs-' + tip
        with self.lock:
            subprocess.run(['git', 'update-ref', ref, tip], cwd=root, check=True)
        prerequisites = [source] if fixed_source_only else list(dict.fromkeys([source, head]))
        bundles.append(self._pack_bundle(root, ref, prerequisites, input_hashes, tracked_workspace))
        return bundles

    def _pack_bundle(self, root, ref, prerequisites, input_hashes, checkout, workspace_objects=()):
        locators = sorted({(row['commit'], row['path']) for row in input_hashes})
        key = (ref, tuple(prerequisites), tuple(locators), tuple(workspace_objects))
        with self.lock:
            sha = self.bundle_hashes.get(key)
        if sha and self.executor.artifact_cache.path(sha).is_file():
            return dict(sha256=sha, ref=ref, checkout_input_head=checkout)
        with tempfile.TemporaryDirectory(prefix='rpe-preparation-git-') as temporary:
            path = Path(temporary) / 'input.bundle'
            build_preparation_git_bundle(root, ref, prerequisites, locators, path, workspace_objects)
            sha = self._blob(path.read_bytes())
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
        if script.name in ('run_pr_pauli_candidate_signals.py','run_pr_pauli_candidate_prices.py')and'--no-push'not in command:
            command.append('--no-push')
        if script.is_relative_to(root):
            script_relative = script.relative_to(root).as_posix()
            source_files = {script_relative: hashlib.sha256(script.read_bytes()).hexdigest()}
            expected = subprocess.check_output(['git', 'show', source + ':' + script_relative], cwd=root)
            if expected != script.read_bytes():
                raise ValueError('Preparation leaf script differs from its declared SOURCE')
            for name in ('src/trottertracks/rpe_preparation_paths.py',
                         'scripts/run_pr_pauli_candidate_signals.py',
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
            if not script.is_relative_to(self.root) or not source_files:
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
        snapshot = self.settings.get('df_snapshot_directory')
        pf_terminal_signal = stage == 'PF_SIGNALS' and script.name == 'run_rpe_pf_horizon_signal.py'
        pf_signal = stage == 'PF_SIGNALS' and (script.name == 'run_pf_preparation_stage.py' or pf_terminal_signal)
        if pf_terminal_signal:
            if not snapshot:
                raise ValueError('Terminal PF signal requires the fixed H4 snapshot')
            caller_root = script.parents[1]
            caller = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=caller_root, text=True).strip()
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
        root_bundles=self._bundle(root,source,input_hashes,
                                 tracked_workspace=not script.is_relative_to(root))
        if script.is_relative_to(root) or pf_terminal_signal:
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
        if (stage == 'DF_SIGNALS' and script.name == 'run_pr_df_candidate_signal_jobs.py'
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

    def call(self, completion, command, directory, stage, output=None, science=True):
        from .pr_candidate_completion import Held, PreparationWait, put
        directory = Path(directory)
        marker = directory / (stage + '.DISTRIBUTED.json')
        inherited_workspace = None
        if marker.exists():
            submitted = read(marker)
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
        put(marker, dict(submitted, job_id=job_id))
        result = self.executor.preparation_result(job_id, wait=True, adopt=False)
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
        if problem := df_freeze_output_problem(request, result):
            raise Held('PREPARATION_RESULT_MISMATCH:'+stage+':'+job_id+':'+problem)
        try:
            if inherited_workspace is None:
                self._import_result(result, request)
            else:
                self._import_result(result, request, inherited_workspace=inherited_workspace)
        except FileNotFoundError as error:
            raise PreparationWait('PREPARATION_RESULT_ARTIFACTS_MISSING:'+stage+':'+job_id+':'+str(error)) from error
        except ValueError as error:
            raise Held('PREPARATION_RESULT_MISMATCH:'+stage+':'+job_id+':'+str(error)) from error
        self.executor.adopt_preparation(job_id)
        receipt = dict(returncode=0, signal=None, source=request['preparation']['source']['commit'], job_id=job_id,
                       transport='distributed', automatic_retry=False, preparation_calls=result.get('preparation_calls', 0))
        put(directory / (stage + '.RETURN.json'), receipt)
        completion.log(event='DISTRIBUTED_STAGE_RETURN', stage=stage, job_id=job_id)
        return read(output) if output and Path(output).exists() else receipt
