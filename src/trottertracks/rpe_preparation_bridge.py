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
from pathlib import Path
import subprocess
import tempfile
import threading

from .rpe_preparation_artifacts import artifact_path


STAGES = {
    'PAULI_SIGNALS': 'signal', 'DF_SIGNALS': 'signal',
    'DF_DESCRIPTOR': 'signal', 'A_BASE_P0': 'signal', 'PF_SIGNALS': 'signal',
    'PAULI_COMPILE': 'compile', 'DF_CALIBRATION': 'compile',
    'DF_HOLDOUT': 'compile', 'PF_COMPILE': 'compile',
    'PF_HOLDOUT_COMPILE': 'compile',
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


class DistributedPreparationBridge:
    def __init__(self, executor, root, settings=None, *, approved=False):
        self.executor = executor
        self.root = Path(root).resolve()
        self.settings = deepcopy(settings or {})
        self.approved = approved is True
        self.alias_path=Path(executor.run_dir)/'preparation'/'PATH_ALIASES.json'
        self.aliases=read(self.alias_path)if self.alias_path.exists()else{}
        self.lock=threading.RLock()

    @staticmethod
    def handles(stage):
        return stage in STAGES

    def _blob(self, raw):
        sha = hashlib.sha256(raw).hexdigest()
        self.executor.artifact_cache.put({sha: base64.b64encode(raw).decode('ascii')})
        return sha

    def _bundle(self, root, source):
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        namespace='refs/rpe-preparation/' + digest(str(self.executor.run_dir))[:16]
        imported=subprocess.check_output(['git','for-each-ref','--format=%(refname)',namespace],cwd=root,text=True).splitlines()
        if head == source and not imported:
            return []
        # Include the original provider branch even if its publication history
        # diverged. Git bundle prerequisites are checked by the destination;
        # no scientific checkout or branch HEAD is replaced here.
        with tempfile.TemporaryDirectory(prefix='rpe-preparation-git-') as temporary:
            path = Path(temporary) / 'input.bundle'
            subprocess.run(['git', 'bundle', 'create', str(path), 'HEAD', *imported, '^' + source], cwd=root,
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return [dict(sha256=self._blob(path.read_bytes()), ref='HEAD')]

    def build_request(self, completion, command, directory, stage, output=None):
        """Freeze a portable leaf request; never invoke its scientific body."""
        if stage not in STAGES:
            raise ValueError('No distributed preparation stage mapping: ' + stage)
        root = Path(getattr(completion, 'preparation_root', self.root)).resolve()
        directory = Path(directory).resolve()
        relative = directory.relative_to(root).as_posix()
        source = completion.source
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
                         'scripts/run_pr_pauli_candidate_signals.py'):
                item=root/name
                if item.is_file():
                    raw=subprocess.check_output(['git','show',source+':'+name],cwd=root)
                    if raw!=item.read_bytes():raise ValueError('Preparation helper differs from fixed SOURCE: '+name)
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
        # Original JSON remains byte-for-byte unchanged. Only runtime path
        # access is resolved by rpe_preparation_paths in the existing provider.
        for arg in command:
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
            if workspace:
                workspace = Path(workspace).resolve()
                if workspace != root:
                    alias = 'provider'
                    commit = source
                    files={}
                    for name in ('scripts/run_pauli_signal_entry.py','scripts/prepare_pauli_calibration.py',
                                 'scripts/run_pauli_split_cal.py','scripts/pauli_split_cal_audit_request.py',
                                 'src/trottertracks/pauli_signal_entry.py','src/trottertracks/pauli_calibration_request.py'):
                        item=self.root/name
                        if item.is_file():
                            raw=subprocess.check_output(['git','show',commit+':'+name],cwd=self.root)
                            if raw!=item.read_bytes():raise ValueError('Provider workspace SOURCE differs: '+name)
                            files[name]=hashlib.sha256(raw).hexdigest()
                    if not files:raise ValueError('Preparation workspace has no fixed provider source')
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
        with self.lock:
            for old, original in self.aliases.items():
                if original in bindings:bindings[old]=bindings[original]
        root_bundles=self._bundle(root,source)
        # Pauli's publication workspace has a separate object database. It
        # reads the controller's newly committed configuration too; importing
        # only the prior provider branch would leave that locator unavailable.
        for workspace in workspaces.values():
            combined=root_bundles+workspace['input_git_bundles']
            workspace['input_git_bundles']=list({item['sha256']:item for item in combined}.values())
        requirements,compiler_bindings=contract_runtime(runtime_inputs,script,self.settings.get('runtime_requirements', {}).get(STAGES[stage], {}))
        prep = dict(stage=STAGES[stage], provider=stage, command=list(command),
                    source=dict(commit=source, files=source_files), source_workspace=True,
                    controller_root=str(root),
                    input_git_bundles=root_bundles, workspaces=workspaces,
                    path_bindings=bindings, inputs=inputs, outputs=list(dict.fromkeys(outputs)),
                    optional_outputs=optional_outputs,
                    runtime_requirements=requirements,compiler_bindings=compiler_bindings)
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
            path = directory / artifact_path(bundle['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifacts[bundle['path']]:
                raise ValueError('Returned scientific Git bundle changed')
            destination = workspaces[bundle.get('workspace', 'workspace')]
            # The central optimizer's GitReader can be a different repository
            # from an owned provider clone. Both must resolve the original
            # returned commit; neither working HEAD is changed.
            for repository in dict.fromkeys([destination,self.root]):
                subprocess.run(['git', 'fetch', '--no-tags', str(path), bundle.get('ref', 'HEAD')], cwd=repository,
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                tip=subprocess.check_output(['git','rev-parse','FETCH_HEAD'],cwd=repository,text=True).strip()
                namespace='refs/rpe-preparation/'+digest(str(self.executor.run_dir))[:16]
                subprocess.run(['git','update-ref',namespace+'/'+artifacts[bundle['path']],tip],cwd=repository,check=True)
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
            request = self.build_request(completion, command, directory, stage, output)
            submitted = dict(request=request, source=completion.source, automatic_retry=False)
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
        receipt = dict(returncode=0, signal=None, source=completion.source, job_id=job_id,
                       transport='distributed', automatic_retry=False, preparation_calls=result.get('preparation_calls', 0))
        put(directory / (stage + '.RETURN.json'), receipt)
        completion.log(event='DISTRIBUTED_STAGE_RETURN', stage=stage, job_id=job_id)
        return read(output) if output and Path(output).exists() else receipt
