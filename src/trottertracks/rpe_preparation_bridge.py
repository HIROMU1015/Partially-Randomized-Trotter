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
import subprocess
import tempfile
import threading

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


def contract_runtime(values, script, configured):
    """Extract declared package/compiler identities without importing a provider."""
    result=deepcopy(configured);packages=result.setdefault('packages',{});compilers=set()
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


def fixed_input_hashes(root, documents, artifacts):
    """Hash every fixed Git locator in original JSON, including nested ones."""
    from .rpe_q3_coordinator import input_hashes
    working=[]
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
            matches=[(name,sha)for name,sha in artifacts.items()if name.endswith('/'+value['path'])]
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

    def _blob(self, raw):
        sha = hashlib.sha256(raw).hexdigest()
        self.executor.artifact_cache.put({sha: base64.b64encode(raw).decode('ascii')})
        return sha

    def _bundle(self, root, source, input_hashes=(), *, tracked_workspace=False):
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        bundles = []
        namespace = 'refs/rpe-preparation/input-bundles/'
        # Send the shared checkout history once per worker cache. Retain that
        # history for provider code that resolves hard-coded Git references.
        if head != source and subprocess.check_output(
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
        prerequisites = list(dict.fromkeys([source, head]))
        bundles.append(self._pack_bundle(root, ref, prerequisites, input_hashes, tracked_workspace))
        return bundles

    def _pack_bundle(self, root, ref, prerequisites, input_hashes, checkout):
        locators = sorted({(row['commit'], row['path']) for row in input_hashes})
        key = (ref, tuple(prerequisites), tuple(locators))
        with self.lock:
            sha = self.bundle_hashes.get(key)
        if sha and self.executor.artifact_cache.path(sha).is_file():
            return dict(sha256=sha, ref=ref, checkout_input_head=checkout)
        revisions = [ref, *('^' + commit for commit in prerequisites)]
        with tempfile.TemporaryDirectory(prefix='rpe-preparation-git-') as temporary:
            path = Path(temporary) / 'input.bundle'
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
            sha = self._blob(path.read_bytes())
        with self.lock:
            self.bundle_hashes[key] = sha
        return dict(sha256=sha, ref=ref, checkout_input_head=checkout)

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
        workspaces = {};runtime_inputs=[]
        pauli_inputs = []
        pauli_locators = []
        if script.name=='run_pr_df_candidate_prices.py' and '-c' in command:
            runtime_inputs.append(dict(df_price_view=command[command.index(str(script))+1]))
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
                    if pauli_locators:
                        for name in provider_files('run_pauli_signal_entry.py'):
                            raw = subprocess.check_output(['git','show',commit+':'+name],cwd=root)
                            files[name] = hashlib.sha256(raw).hexdigest()
                    workspaces[alias] = dict(source=dict(commit=commit, files=files), input_git_bundles=self._bundle(workspace,commit),checkout_input_head=False)
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
        original_documents=list(pauli_locators)
        for name,sha in inputs.items():
            if name.endswith('.json'):
                try:original_documents.append(json.loads(self.executor.artifact_cache.path(sha).read_bytes()))
                except (ValueError,UnicodeError):raise ValueError('Original preparation JSON is incomplete or invalid: '+name)
        input_hashes,working_hashes=fixed_input_hashes(root,original_documents,inputs)
        working_hashes.extend(pauli_inputs)
        required_views=collect_required_views(runtime_inputs)
        if stage in ('PAULI_COMPILE', 'PAULI_AUDIT') and script.name=='run_pr_pauli_candidate_prices.py':
            required_views.update(pauli_calibration_view(root))
        required_paths=collect_required_paths(runtime_inputs,self.settings.get('controller_paths'))
        with self.lock:
            for old, original in self.aliases.items():
                if original in bindings:bindings[old]=bindings[original]
        root_bundles=self._bundle(root,source,input_hashes,
                                 tracked_workspace=not script.is_relative_to(root))
        if script.is_relative_to(root):
            # This bundle supplies locators for frozen inputs. Checking out
            # its HEAD would substitute controller code for the provider.
            root_bundles=[dict(bundle,checkout_input_head=False)for bundle in root_bundles]
        # Pauli's publication workspace has a separate object database. It
        # reads the controller's newly committed configuration too; importing
        # only the prior provider branch would leave that locator unavailable.
        for workspace in workspaces.values():
            combined=root_bundles+workspace['input_git_bundles']
            workspace['input_git_bundles']=list({item['sha256']:item for item in combined}.values())
        requirements,compiler_bindings=contract_runtime(runtime_inputs,script,self.settings.get('runtime_requirements', {}).get(STAGES[stage], {}))
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
        if stage in ('DF_CALIBRATION', 'DF_HOLDOUT') and '--input' in command:
            frozen = read(Path(command[command.index('--input')+1]))
            if frozen.get('parallel_axes') is True:
                if frozen['source'] != source:
                    raise ValueError('Parallel DF price input SOURCE differs from its assigned runner')
                prep['parallel_df_price_axes'] = True
        self._validate_bundle_inputs(prep)
        return dict(schema='RPE_PREPARATION_REQUEST_V1', route=dict(kind='preparation'), preparation=prep,
                    original_contract=dict(source=source, stage=stage, automatic_retry=False))

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
        inputs = view.get('input_git_bundles', [])
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

    def _import_result(self, result, request):
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
        from .pr_candidate_completion import Held, put
        directory = Path(directory)
        marker = directory / (stage + '.DISTRIBUTED.json')
        if marker.exists():
            submitted = read(marker)
            request = submitted['request']
        else:
            prior=self.inspect_prior_owner(command,directory,stage)
            put(directory/(stage+'.PRIOR_OWNER.json'),prior)
            if prior['failed']:
                raise Held('STAGE_FAILED:'+stage+':ORIGINAL_OWNER_STOP_OR_FAILURE:NO_AUTOMATIC_RETRY:'+str(prior))
            if prior['pending']:
                raise Held('DISTRIBUTED_PREPARATION_OPERATIONAL_HOLD:'+stage+':ORIGINAL_OWNER_RUNNING_OR_UNKNOWN:'+str(prior))
            try:request = self.build_request(completion, command, directory, stage, output)
            except (OSError,ValueError,subprocess.CalledProcessError)as error:
                raise Held('DISTRIBUTED_PREPARATION_OPERATIONAL_HOLD:'+stage+':INPUT_OR_SOURCE_METADATA_UNAVAILABLE:'+str(error))from error
            submitted = dict(request=request, source=request['preparation']['source']['commit'], automatic_retry=False)
        # A Q3 approval/budget does not open signal/compile scope. The round
        # must separately bound preparation tasks and retain leaf input caps.
        budget = self.settings.get('budgets', {}).get(STAGES[stage], 0)
        dependencies=[]
        for dependency in DEPENDENCIES.get(stage,()):
            path=directory/(dependency+'.DISTRIBUTED.json')
            if path.exists():dependencies.append(read(path)['job_id'])
        job_id = self.executor.submit_preparation(request, budget=budget, approved=self.approved,
            contract=request['original_contract'],dependencies=dependencies)
        if submitted.get('job_id', job_id) != job_id:
            raise ValueError('Preparation task identity changed on resume')
        put(marker, dict(submitted, job_id=job_id))
        result = self.executor.preparation_result(job_id, wait=True, adopt=True)
        if not result or result.get('status') != 'PREPARATION_COMPLETE':
            terminal=result or{}
            while terminal.get('status')=='PREPARATION_DEPENDENCY_HOLD'and isinstance(terminal.get('predecessor'),dict):
                terminal=terminal['predecessor']
            if terminal.get('status')in('PREPARATION_FAILED','UNKNOWN'):
                raise Held('STAGE_FAILED:'+stage+':DISTRIBUTED_'+terminal['status']+':NO_AUTOMATIC_RETRY:'+str(result))
            raise Held('DISTRIBUTED_PREPARATION_OPERATIONAL_HOLD:' + stage + ':' + str(result))
        self._import_result(result, request)
        receipt = dict(returncode=0, signal=None, source=request['preparation']['source']['commit'], job_id=job_id,
                       transport='distributed', automatic_retry=False, preparation_calls=result.get('preparation_calls', 0))
        put(directory / (stage + '.RETURN.json'), receipt)
        completion.log(event='DISTRIBUTED_STAGE_RETURN', stage=stage, job_id=job_id)
        return read(output) if output and Path(output).exists() else receipt
