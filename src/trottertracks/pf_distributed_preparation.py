"""Split the selected D PF protocol over the existing shared worker queue.

The original provider plans, preflights, freezes and registers the selected
configuration. Numerical signal/compile and C receipt admission use typed tasks.
All producer publication is in a new owned clone, never its fixed source view.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import json
import os
import subprocess
import sys
from tempfile import TemporaryDirectory


def read(path):return json.loads(Path(path).read_text())
def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def git(root,*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()


def run_preflight(command, **options):
    """Limit a failed metadata probe to its candidate and tolerate a departing PID."""
    from .pr_candidate_completion import Held, PreparationWait

    for attempt in range(2):
        try:
            return subprocess.run(command, check=True, stderr=subprocess.PIPE, **options)
        except subprocess.CalledProcessError as error:
            stderr = (error.stderr or b'').decode(errors='replace')
            vanished = 'ProcessLookupError: [Errno 3]' in stderr and 'in active_price_jobs' in stderr
            if vanished and attempt == 0:
                continue
            detail = '\n'.join(stderr.splitlines()[-8:]) or str(error)
            failure = PreparationWait if vanished else Held
            raise failure('PF_METADATA_PREFLIGHT_HOLD:'+detail) from error


def selected_workspace_request(request, workspace):
    destination = workspace/'docs/research/distributed_selected_pf'
    return dict(request, output_root=str(destination), registry_file=str(destination/'REGISTRY.json'))


def place_saved_price_inputs(root, workspace, prices):
    """Transfer selected guard receipts without traversing their Git history."""
    from .pr_candidate_completion import PreparationWait
    from .rpe_live_q3_readiness import copy_git_locators

    refs = set()
    admissions = set()
    for row in prices:
        evidence = row['price_evidence']
        admission = evidence['admission']
        if not admission['path'].endswith('/GUARD_CHECK.json'):
            continue
        for entry in (evidence['source'], admission):
            refs.add((entry['commit'], entry['path']))
        key = admission['commit'], admission['path'], tuple(admission.get('selector', ()))
        admissions.add(key)
    try:
        for commit, path, selector in admissions:
            guard = json.loads(git(root, 'show', commit+':'+path))
            for index in selector:
                guard = guard[index]
            refs.update((item['evidence']['commit'], item['evidence']['path'])
                        for item in guard.get('guard', guard)['checked'])
        copy_git_locators(root, workspace, refs)
    except subprocess.CalledProcessError as error:
        raise PreparationWait('PF_SAVED_PRICE_GIT_INPUT_HOLD: '+str(error)) from error


def recover_signal_publication(workspace, commit, owner):
    """Read the accepted signal bundle and its fixed input delta from the old run."""
    from .pf_saved_prices import bound_owner_input, distributed_owner_return
    from .rpe_preparation_artifacts import ArtifactCache, artifact_path

    stage = 'PF_SIGNALS'
    owner = Path(owner)
    marker = read(owner/(stage+'.DISTRIBUTED.json'))
    if not bound_owner_input(marker, owner/(stage+'.INPUT.json')):
        raise ValueError('Accepted PF signal input differs from the prior request')
    state, returned = distributed_owner_return(owner, marker, stage)
    if state != 'ENDED' or commit not in returned['git_heads']:
        raise ValueError('No accepted PF signal publication for '+commit)
    job = Path(returned['accepted_job'])
    outcome = read(job/'OUTCOME.json')
    [bundle] = [entry for entry in outcome['git_bundles']
                if entry.get('workspace', 'workspace') == 'workspace' and entry['head'] == commit]
    run = job.parents[2]
    cache = ArtifactCache(run/'distributed/artifacts')
    source_ref = bundle.get('ref', 'HEAD')
    result_ref = 'refs/rpe-preparation/recovered/'+commit

    def fetch(path, ref, target):
        subprocess.run(['git', 'fetch', '--no-tags', '--no-write-fetch-head', str(path), ref+':'+target],
                       cwd=workspace, check=True, capture_output=True)

    def has_commit(sha):
        return subprocess.run(['git', 'cat-file', '-e', sha+'^{commit}'], cwd=workspace,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

    def fetch_input(row):
        sha, ref = row['sha256'], row.get('ref', 'HEAD')
        path = cache.path(sha)
        target = result_ref+'/inputs/'+sha
        if not cache.missing([sha]):
            fetch(path, ref, target)
            return
        # Retention may have compacted the input bundle after the run ended.
        # Reconstruct only the declared request's exact bytes, in temporary
        # storage, so the original run's archive and cache remain untouched.
        from .rpe_preparation_retention import restore

        for parent in run.parents:
            archives = parent/'archives/preparation-inputs'
            if not archives.is_dir():
                continue
            for recipe in sorted(archives.glob('*/'+sha+'.recipe.json.gz')):
                with TemporaryDirectory(prefix='rpe-pf-signal-input-') as temporary:
                    recovered = ArtifactCache(temporary)
                    restore(marker['request'], sha, archive=recipe.parent, cache_directory=recovered.root)
                    fetch(recovered.path(sha), ref, target)
                return
        raise FileNotFoundError('Missing accepted PF signal input bundle '+sha)

    if input_head := bundle.get('input_head'):
        if not has_commit(input_head):
            inputs = [row for row in marker['request']['preparation'].get('input_git_bundles', ())
                      if not row.get('workspace_objects')]
            for row in inputs:
                fetch_input(row)
                if has_commit(input_head):
                    break
            if not has_commit(input_head):
                raise ValueError('The saved inputs do not provide the accepted PF signal parent')
    if 'path' in bundle:
        name = artifact_path(bundle['path']).as_posix()
        sha = outcome['artifacts'][name]
        if cache.missing([sha]):
            raise FileNotFoundError('Missing accepted PF signal output bundle '+sha)
        path = cache.path(sha)
        heads = git(workspace, 'bundle', 'list-heads', str(path), source_ref).splitlines()
        if len(heads) != 1 or heads[0].split()[0] != commit:
            raise ValueError('Accepted PF signal bundle publication changed')
        # Keep inputs under a sibling ref: Git cannot store a ref and a
        # directory of refs with that same name.
        fetch(path, source_ref, result_ref+'/output')
    elif input_head != commit:
        raise ValueError('Accepted PF signal publication is missing its output bundle')


def reuse_signal(root, workspace, folder, plan, saved, *, waves=()):
    """Carry the original completed signal into an explicitly scoped continuation."""
    from .pr_candidate_completion import Held
    relative=folder.relative_to(workspace).as_posix()
    commit=saved['commit']
    source_root = root

    def blob(name):
        return subprocess.check_output(['git','show',commit+':'+relative+'/'+name],cwd=source_root,
                                       stderr=subprocess.PIPE)
    try:
        exists = subprocess.run(['git', 'cat-file', '-e', commit+'^{commit}'], cwd=root,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if not exists:
            from .pf_saved_prices import archived_signal_owner

            owner = saved.get('owner') or archived_signal_owner(waves, commit)
            if owner:
                recover_signal_publication(workspace, commit, owner)
                source_root = workspace
        prior=json.loads(blob('PLAN.json'))
        status=json.loads(blob('signal_run01/STATUS.json'))
        handoff=json.loads(blob('signal_run01/MEAN_HANDOFF.json'))
        if (prior['configuration']!=plan['configuration'] or
                prior['prepared']['pending']['requests']!=plan['prepared']['pending']['requests'] or
                handoff['configuration_id']!=plan['configuration']['id'] or
                handoff['target']!=plan['prepared']['target'] or
                status['status']!='COMPLETE_THIRTEEN_ROUNDS' or len(handoff['means'])!=13):
            raise ValueError('Original completed signal does not match the selected PF plan')
        files=['STATUS.json','MEAN_HANDOFF.json',*(f'round_{m}.json' for m in range(13))]
        original={name:blob('signal_run01/'+name) for name in files}
        if source_root == root:
            subprocess.run(['git','fetch','--no-tags','--no-write-fetch-head',str(root),commit],cwd=workspace,
                           check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        else:
            subprocess.run(['git','fetch','--no-tags','--no-write-fetch-head',str(workspace),
                            commit+':refs/rpe-preparation/recovered/'+commit+'/output'],cwd=root,
                           check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        output=folder/'signal_run01';output.mkdir(exist_ok=True)
        for name, raw in original.items():
            path=output/name
            if path.exists():
                if path.read_bytes()!=raw:raise ValueError('Existing PF signal differs from the original artifact')
            else:
                with path.open('xb') as stream:stream.write(raw)
        put(folder/'SAVED_SIGNAL_REUSE.json',dict(configuration_id=handoff['configuration_id'],
            original_source=dict(commit=commit,path=relative+'/signal_run01/MEAN_HANDOFF.json'),
            reused_means=13,new_means=0))
    except (OSError, KeyError, IndexError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        raise Held('PF_SAVED_SIGNAL_REUSE_HOLD:'+str(error)) from error


def reuse_calibration(root, workspace, folder, plan, rounds, receipt, *, phase='calibration'):
    """Restore exact completed calibration or holdout compile artifacts."""
    if phase not in ('calibration', 'holdout'):
        raise ValueError('Unsupported PF compile reuse phase: '+phase)
    from .pr_candidate_completion import Held
    relative=(folder/phase/'run01').relative_to(workspace).as_posix()
    commit=receipt['raw_commit']
    try:
        selected_axes = plan.get('selected_axis_compile') is True
        expected = ({str(cell['m']): cell['axes'] for cell in plan['new_compile_cells'] if cell['m'] in rounds}
                    if selected_axes else {str(m): ['X','Y'] for m in rounds})
        received = receipt.get('axes_by_round', {str(m): ['X','Y'] for m in rounds})
        if (receipt['status']!='COMPILE_RECEIPTS_READY' or receipt['phase']!=phase or
                receipt['relative']!=relative or receipt['rounds']!=rounds or expected != received):
            raise ValueError('Collected calibration differs from the selected PF leaf')
        def blob(path):
            return subprocess.check_output(['git','show',commit+':'+path],cwd=root)
        origin=json.loads(blob(folder.relative_to(workspace).as_posix()+'/PLAN.json'))
        result=json.loads(blob(relative+'/RESULT.json'))
        wrappers = result['results']
        axes_match = (all(len(row['wrappers']) == 2 for row in wrappers) if not selected_axes else
                      {row['cell']['m'] for row in wrappers} == set(rounds)
                      and all(set(row['wrappers']) == {
                          {'X': 'cosine', 'Y': 'sine'}[axis] for axis in expected[str(row['cell']['m'])]
                      } for row in wrappers))
        if (origin['configuration']!=plan['configuration'] or
                origin['prepared']['pending']['requests']!=plan['prepared']['pending']['requests'] or
                result['child_exit_code']!=0 or len(wrappers)!=len(rounds)
                or not axes_match or not all(row['ok'] for row in wrappers)):
            raise ValueError('Collected native calibration does not match the selected PF plan')
        names=git(root,'ls-tree','-r','--name-only',commit,'--',relative).splitlines()
        subprocess.run(['git','fetch','--no-tags','--no-write-fetch-head',str(root),commit],cwd=workspace,check=True,
                       stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        for name in names:
            path=workspace/Path(name)
            path.relative_to(folder/phase/'run01')
            path.parent.mkdir(parents=True,exist_ok=True)
            raw=blob(name)
            if path.exists():
                if path.read_bytes()!=raw:raise ValueError('Existing calibration artifact differs')
            else:
                with path.open('xb') as stream:stream.write(raw)
        put(folder/('PF_COMPILE.RESULT.json' if phase=='calibration' else 'PF_HOLDOUT_COMPILE.RESULT.json'),receipt)
    except (OSError,KeyError,ValueError,subprocess.CalledProcessError) as error:
        raise Held('PF_SAVED_CALIBRATION_REUSE_HOLD:'+str(error)) from error
    return receipt


def track_collected_prices(workspace, compiled):
    """Make the worker's byte-identical collected files tracked for the C guard."""
    relative=Path(compiled['relative'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Collected PF price path leaves its workspace')
    git(workspace,'add','--sparse','--',str(relative))
    if git(workspace,'diff','--cached','--name-only','--',str(relative)):
        git(workspace,'-c','user.name=HIROMU1015','-c','user.email=u411612g@ecs.osaka-u.ac.jp',
            'commit','-m','Track recovered PF raw price inputs without replacing their original SOURCE',
            '--',str(relative))


def stage_original_audit(root, workspace, folder, value, audit):
    """Stage a projection of an original publication for the unchanged C guard."""
    from .pf_saved_prices import raw_compile_documents
    from .pr_candidate_completion import Held

    receipt = audit['compiled']
    axes = audit['axes_by_round']
    target = folder/'recovered_prices'/receipt['raw_commit']
    raw = target/'raw'
    relative = raw.relative_to(workspace).as_posix()
    wanted = {(int(m), a) for m, names in axes.items() for a in names}
    try:
        result, prices, sidecars, proofs = raw_compile_documents(root, value, receipt)
        selected = [(p, deepcopy(s)) for p, s in zip(prices, sidecars, strict=True)
                    if (p['m'], p['axis']) in wanted]
        if {(p['m'], p['axis']) for p, _ in selected} != wanted:
            raise ValueError('Original PF audit axes differ from the fixed recovery')
        origin = dict(commit=receipt['raw_commit'], path=receipt['relative']+'/RESULT.json')
        for _, side in selected:
            # The projected raw proof is byte-equivalent in content to the
            # original. Its locator follows the fixed C binding convention.
            side['implementation_evidence']['path'] = relative+'/IMPLEMENTATION_EVIDENCE.json'
        documents = {
            'RESULT.json': dict(original_compile=origin, axes_by_round=axes,
                                child_exit_code=result['child_exit_code']),
            'PRICES.json': [p for p, _ in selected],
            'SIDECARS_UNBOUND.json': [s for _, s in selected],
            'IMPLEMENTATION_EVIDENCE.json': proofs,
        }
        subprocess.run(['git', 'fetch', '--no-tags', '--no-write-fetch-head', str(root), origin['commit']],
                       cwd=workspace, check=True, capture_output=True)
        for name, body in documents.items():
            path = raw/name
            if not path.exists():
                put(path, body)
            elif read(path) != body:
                raise ValueError('Preserve original PF audit projection: '+str(path))
        track_collected_prices(workspace, {'relative': relative})
        commit = git(workspace, 'log', '-1', '--format=%H', '--', relative)
        staged = dict(status='COMPILE_RECEIPTS_READY', phase=receipt['phase'], route=receipt['route'],
                      raw_commit=commit, relative=relative, axes_by_round=axes,
                      rounds=sorted(int(m) for m in axes), original_compile=origin)
        # The existing input collector binds each provisional sidecar to
        # this durable raw publication.
        put(target/'ORIGINAL_COMPILE.RESULT.json', staged)
        return target, staged
    except (OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        raise Held('PF_SAVED_CALIBRATION_REUSE_HOLD:'+str(error)) from error


def saved_for_configuration(settings, field, configuration_id):
    """Reuse a completed leaf only for its canonical configuration identity."""
    plural = {'saved_signal': 'saved_signals',
              'saved_holdout_compile': 'saved_holdout_compiles',
              'saved_calibration': 'saved_calibrations',
              'saved_calibration_audit': 'saved_calibration_audits'}[field]
    values = settings.get(plural, {})
    if not isinstance(values, dict):
        raise ValueError('PF '+plural+' must map canonical configuration IDs to saved receipts')
    return values.get(configuration_id, settings.get(field))


def complete_selected(root,spec,run,request,entry,output,runtime,*,inherited_registry=None):
    """Original finite PF contract; no independent candidate/shot selection."""
    from .pr_candidate_completion import PreparationWait
    if not Path(entry).is_file():
        raise PreparationWait('Fixed PF provider entry is unavailable: '+str(entry))
    original=Path(entry).resolve().parents[1];run=Path(run)
    owner_input=run/'ORIGINAL_OWNER_REQUEST.json';owner_output=run/'ORIGINAL_OWNER_CHECK.json'
    outer=next((p for p in run.parents if p.name=='pf_auto'),run)
    wave_root = next((p for p in reversed(run.parents) if p.name == 'pf_auto'), outer)
    signal_index=wave_root/'cache'/'pf-saved-signal-index'
    metadata_root = Path(__file__).resolve().parents[2]
    metadata_env = os.environ.copy()
    snapshot = getattr(runtime, 'settings', {}).get('df_snapshot_directory')
    if snapshot:
        metadata_env['RPE_PF_PREFLIGHT_SNAPSHOT'] = str(snapshot)
    price_args = []
    settings = spec['pf_completion']
    exploratory = settings.get('exploratory_cost') is True
    if exploratory and inherited_registry is None:
        request = dict(request, exploratory_cost=True)
    saved_signals = {}
    from .pf_saved_prices import prior_waves
    earlier_waves = prior_waves(wave_root.parent, spec)
    if inherited_registry is not None:
        # Resume the exact owner check and plan already bound to this workspace.
        request = read(owner_input)
    elif settings.get('recover_saved_price_rounds', True):
        from .pf_saved_prices import candidate_roots, completed_partial, price_indexes
        price_index = run/'SAVED_PRICE_INDEX.json'
        if not price_index.exists():
            put(price_index, dict(root=str(root), locators=price_indexes([wave_root, *earlier_waves], settings)))
        price_args = ['--saved-price-index', str(price_index)]
        if not owner_input.exists():
            owners = set(request.get('prior_owner_directories', ()))
            consumed = {tuple(axis) for axis in request.get('previously_audited_price_axes', ())}
            for item in request['selected']:
                cfg = item['configuration_body']['id']
                for waves in (wave_root, *earlier_waves):
                    for base in candidate_roots(waves):
                        leaf = base/cfg/'pf_auto/pf_completion'
                        if leaf == run:
                            continue
                        provider = leaf/'provider_workspace/docs/research/distributed_selected_pf'/cfg
                        if not provider.is_dir():
                            continue
                        audited = completed_partial({}, leaf, cfg) if len(request['selected']) == 1 else None
                        if audited is None:
                            owners.add(str(provider))
                        else:
                            consumed.update(tuple(axis) for axis in audited)
                            result = provider/'PF_SIGNALS.RESULT.json'
                            if result.is_file():
                                saved = dict(commit=read(result)['publication'])
                                if (provider/'PF_SIGNALS.DISTRIBUTED.json').is_file():
                                    saved['owner'] = str(provider)
                                saved_signals[cfg] = saved
            request = dict(request, prior_owner_directories=sorted(owners))
            if consumed:
                request['previously_audited_price_axes'] = [list(axis) for axis in sorted(consumed)]
            if saved_signals:
                request['independent_saved_signals'] = saved_signals
        else:
            # Bind the prior scope of this scientific leaf at its first precheck.
            fixed_owner = read(owner_input)
            request = dict(request, prior_owner_directories=fixed_owner.get('prior_owner_directories', []))
            if 'previously_audited_price_axes' in fixed_owner:
                request['previously_audited_price_axes'] = fixed_owner['previously_audited_price_axes']
            if 'independent_saved_signals' in fixed_owner:
                request['independent_saved_signals'] = fixed_owner['independent_saved_signals']
    if not owner_input.exists():
        put(owner_input,request)
    elif read(owner_input) != request:
        raise ValueError('Original PF owner request changed on resume')
    registry_output = inherited_registry or request['registry_file']
    if not owner_output.exists() or not (run/'PROVIDER_SOURCE.json').exists():
        run_preflight([spec['pf_completion'].get('interpreter',sys.executable),'-B',
            str(metadata_root/'scripts/check_pf_preparation_owner.py'),'--provider-root',str(original),
            '--request',str(owner_input),'--output',str(owner_output),'--signal-index',str(signal_index),
            *price_args],env=metadata_env)
    owner=read(owner_output)
    discovered = [ref for row in owner['rows'] for ref in row.get('discovered_price_receipts', ())]
    if discovered and price_args:
        from .pf_saved_prices import unique_locators
        resolved_index = run/'RESOLVED_PRICE_INDEX.json'
        if not resolved_index.exists():
            index_data = read(price_index)
            index_data['locators'] = unique_locators([*index_data['locators'], *discovered])
            put(resolved_index, index_data)
        price_args = ['--saved-price-index', str(resolved_index)]
    held=[row for row in owner['rows']if row['status'] in ('PRIOR_OWNER_HOLD','PRICE_EVIDENCE_REQUIRED')]
    if held:
        if not Path(registry_output).exists():put(registry_output,dict(schema='D_PF_EVIDENCE_REGISTRY_V1',configurations={}))
        put(output,dict(schema='D_PF_EVIDENCE_COMPLETION_V1',items=held,registry_file=registry_output))
        return
    reused=[row for row in owner['rows']if row['status']=='REUSE_COMPLETE_ORIGINAL_OWNER']
    if len(reused)==len(owner['rows']):
        registry=read(request['registry_file'])if Path(request['registry_file']).exists()else dict(schema='D_PF_EVIDENCE_REGISTRY_V1',configurations={})
        for row in reused:registry['configurations'][row['configuration_id']]=row['entry']
        subprocess.run(['git','fetch','--no-tags','--no-write-fetch-head',str(original),'HEAD'],cwd=root,check=True,
            stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        put(registry_output,registry);put(output,dict(schema='D_PF_EVIDENCE_COMPLETION_V1',items=reused,registry_file=registry_output))
        return
    workspace=run/'provider_workspace';marker=run/'PROVIDER_SOURCE.json'
    if not marker.exists():
        source=git(original,'rev-parse','HEAD')
        if workspace.exists():raise ValueError('Unidentified PF producer workspace; preserve it')
        subprocess.run(['git','clone','--shared','--no-checkout',str(original),str(workspace)],check=True,
            stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        subprocess.run(['git','sparse-checkout','set','scripts','src','configs','references'],cwd=workspace,check=True)
        subprocess.run(['git','checkout','-b','selected-preparation',source],cwd=workspace,check=True,
            stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        put(marker,dict(root=str(original),source=source))
    fixed=read(marker)
    if fixed['root']!=str(original):raise ValueError('PF provider changed on resume')
    selected=selected_workspace_request(request, workspace)
    inp=workspace/'docs/research/distributed_selected_pf/REQUEST.json'
    plan_output=run/'PLAN_RESPONSE.json'
    if not inp.exists():put(inp,selected)
    elif inherited_registry is not None:
        from .pr_candidate_completion import Held
        [item] = read(plan_output)['items']
        original_workspace = Path(item['output']).parents[3]
        if read(inp) != selected_workspace_request(request, original_workspace):
            raise Held('INHERITED_PF_SELECTED_CONTRACT_CHANGED')
    elif read(inp)!=selected:raise ValueError('PF selected contract changed')
    if not plan_output.exists():
        subprocess.run([spec['pf_completion'].get('interpreter',sys.executable),'-B',
            str(metadata_root/'scripts/run_pf_preparation_stage.py'),'--provider-root',str(workspace),
            '--input',str(inp),'--output',str(plan_output),'--plan-signal-index',str(signal_index),*price_args],
            cwd=workspace,check=True)
    response=read(plan_output);items=[]
    owner_rows = {row['configuration_id']: row for row in owner['rows']}
    for item in response['items']:
        if item['status']=='REUSED_COMPLETED_REGISTRY':items.append(item);continue
        original_folder=Path(item['output'])
        folder=(workspace/'docs/research/distributed_selected_pf'/item['configuration_id']
                if inherited_registry is not None else original_folder)
        value=read(folder/'PLAN.json')
        owner_row = owner_rows[item['configuration_id']]
        deferred = owner_row.get('deferred_price_rounds', {})
        original_calibration = owner_row.get('original_calibration')
        original_audits = owner_row.get('original_audits', ())
        signal_holds = owner_row.get('original_signal_holds', ())
        scoped = exploratory and 'original_owner_directories' in owner_row
        if price_args:
            # Only the admitted prices actually selected by this plan need
            # object delivery to tasks running in the owned producer clone.
            place_saved_price_inputs(root, workspace, value['direct_prices'])
        if (folder/'STOP.json').exists():raise ValueError('Preserve consumed PF STOP; no automatic retry')
        source=dict(commit=fixed['source'],caller='scripts/prepare_pf_candidate.py')
        def local(phase,**fields):
            phase_in=folder/(phase+'.METADATA_INPUT.json');phase_out=folder/(phase+'.METADATA_RESULT.json')
            put(phase_in,dict(value=value,relative_output=str(folder.relative_to(workspace)),
                phase=phase,source=source,**fields))
            if not phase_out.exists():
                # Fresh interpreter: the D provider's package must not mix
                # with the controller's already imported trottertracks.
                command = [spec['pf_completion'].get('interpreter',sys.executable),'-B',
                    str(metadata_root/'scripts/run_pf_preparation_stage.py'),'--provider-root',str(workspace),
                    '--input',str(phase_in),'--output',str(phase_out)]
                if phase == 'preflight':
                    run_preflight(command,cwd=workspace,env=metadata_env)
                else:
                    subprocess.run(command,cwd=workspace,check=True,env=metadata_env)
            return read(phase_out)
        source=local('preflight', **({'deferred_price_rounds': deferred} if deferred else {}))['source']
        provider_files = ['scripts/prepare_pf_candidate.py','src/trottertracks/pf_evidence_preparation.py',
                          'scripts/run_pf_m0_prices.py','scripts/run_pf_p7_h_prices.py','scripts/run_truncated_p7_prices.py']
        if value.get('selected_axis_compile'):
            provider_files += ['scripts/h46_long_time_native.py','scripts/pf_price_stages.py',
                               'scripts/pf_native_price_fast.py','scripts/run_full_pf_shortlist_prices.py']
        context=SimpleNamespace(directory=folder,settings=spec.get('pf_completion',{}),
            source=fixed['source'],preparation_root=workspace,log=lambda **row:None,
            preparation_source_manifest=dict(commit=fixed['source'],files={name:__import__('hashlib').sha256(
                subprocess.check_output(['git','show',fixed['source']+':'+name],cwd=workspace)).hexdigest()
                for name in provider_files}))
        if inherited_registry is not None:
            context.inherited_preparation_root = original_folder.parents[3]
        def stage_input_for(phase, **fields):
            return dict(value=value, relative_output=str(folder.relative_to(workspace)),
                        phase=phase, source=source, **fields)

        def call(phase, tag, *, stage_folder=folder, **fields):
            stage_in=stage_folder/(tag+'.INPUT.json');stage_out=stage_folder/(tag+'.RESULT.json')
            stage_input = stage_input_for(phase, **fields)
            stage_input['relative_output'] = stage_folder.relative_to(workspace).as_posix()
            if inherited_registry is not None and stage_in.exists():
                if read(stage_in) != stage_input:
                    raise Held('INHERITED_PF_STAGE_INPUTS_CHANGED:'+tag)
            else:
                put(stage_in, stage_input)
            cmd=[spec['pf_completion'].get('interpreter',sys.executable),'-B',
                str(Path(root)/'scripts/run_pf_preparation_stage.py'),'--provider-root',str(workspace),
                '--input',str(stage_in),'--output',str(stage_out)]
            recovered_audit = getattr(runtime, 'recovered_audit', None)
            if phase == 'audit' and recovered_audit is not None:
                result = recovered_audit(stage_folder, tag, stage_input, fields['compiled'])
                if result is not None:
                    return result
            options = {'wait': False} if scoped and phase in ('compile', 'audit') else {}
            return runtime.call(context,cmd,stage_folder,tag,stage_out,science=True,**options)
        def compile_inputs(cohort, cells):
            fields = dict(cohort=cohort, rounds=[cell['m'] for cell in cells])
            if value.get('selected_axis_compile'):
                fields['axes_by_round'] = {str(cell['m']): cell['axes'] for cell in cells}
            return fields

        def compile_stage(tag, cohort, cells):
            return call('compile', tag, **compile_inputs(cohort, cells))

        def exploratory_prices(values):
            covered = {(r['m'], r['axis']) for r in values}
            need = {(m, a) for m in value['calibration_rounds'] for a in ('X', 'Y')}
            frozen = None
            if need and need <= covered and not owner_row.get('original_model_hold'):
                try:
                    frozen = local('freeze', prices=values)['model']
                except subprocess.CalledProcessError as error:
                    put(folder/'EXPLORATORY_PRICE_ISSUE.json', dict(stage='PF_MODEL_FREEZE', reason=str(error)))
            return local('exploratory-prices', prices=values, model=frozen)['prices']

        def prices():
            values=deepcopy(value['direct_prices'])
            cells=[r for r in value['new_compile_cells']if r['role']!='HOLDOUT']
            calibration=[r['m'] for r in cells]
            if calibration:
                saved=saved_for_configuration(spec['pf_completion'],'saved_calibration',value['configuration']['id'])
                compiled=(reuse_calibration(root,workspace,folder,value,calibration,saved) if saved else
                          compile_stage('PF_COMPILE','calibration',cells))
                track_collected_prices(workspace,compiled)
                saved_audit=saved_for_configuration(spec['pf_completion'],'saved_calibration_audit',value['configuration']['id'])
                if saved_audit:
                    prices=saved_audit['prices']
                    expected={(cell['m'],axis) for cell in cells for axis in cell.get('axes',('X','Y'))}
                    if (saved_audit['status']!='AUDIT_PASS' or len(prices)!=len(expected) or
                            {(p['m'],p['axis']) for p in prices}!=expected or
                            any(p['gate']!='C_IMPLEMENTATION_GUARD_PASS' or
                                p['price_evidence']['source']['path']!=compiled['relative']+'/PRICES.json'
                                for p in prices)):
                        raise ValueError('Collected audit does not match the selected calibration')
                    commits=sorted({p['price_evidence']['source']['commit'] for p in prices})
                    subprocess.run(['git','fetch','--no-tags','--no-write-fetch-head',str(root),*commits],cwd=workspace,
                                   check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                    put(folder/'SAVED_CALIBRATION_AUDIT_REUSE.json',saved_audit)
                else:
                    prices=call('audit','PF_AUDIT',compiled=compiled)['prices']
                values+=prices
            if exploratory:
                return exploratory_prices(values)
            frozen=call('freeze','PF_MODEL_FREEZE',prices=values)['model']
            holdout_cells=[r for r in value['new_compile_cells']if r['role']=='HOLDOUT']
            holdout=[r['m'] for r in holdout_cells]
            if holdout:
                recover = getattr(runtime, 'recovered_holdout_prices', None)
                recovered = recover(folder, stage_input_for('compile', **compile_inputs('holdout', holdout_cells))) if recover else None
                if recovered is not None:
                    commits = sorted({row['price_evidence']['source']['commit'] for row in recovered})
                    subprocess.run(['git','fetch','--no-tags','--no-write-fetch-head',str(root),*commits],
                                   cwd=workspace,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                    return call('finish','PF_FINISH',prices=values+recovered,model=frozen)['prices']
                saved=saved_for_configuration(spec['pf_completion'],'saved_holdout_compile',value['configuration']['id'])
                compiled=(reuse_calibration(root,workspace,folder,value,holdout,saved,phase='holdout') if saved else
                          compile_stage('PF_HOLDOUT_COMPILE','holdout',holdout_cells))
                track_collected_prices(workspace,compiled)
                values+=call('audit','PF_HOLDOUT_AUDIT',compiled=compiled)['prices']
            return call('finish','PF_FINISH',prices=values,model=frozen)['prices']
        original_signal = owner_row.get('original_signal')
        saved_signal = (saved_for_configuration(settings, 'saved_signal', value['configuration']['id'])
                        or original_signal
                        or request.get('independent_saved_signals', {}).get(value['configuration']['id']))
        if original_signal and saved_signal['commit'] == original_signal['commit']:
            saved_signal = original_signal

        def signals():
            if saved_signal:
                reuse_signal(root,workspace,folder,value,saved_signal,waves=(wave_root, *earlier_waves))
                return None
            return call('signal','PF_SIGNALS')

        def available_prices(get_prices):
            from .pr_candidate_completion import Held
            try:
                return get_prices()
            except Held as error:
                if not exploratory:
                    raise
                put(folder/'EXPLORATORY_PRICE_ISSUE.json', dict(stage='PF_CALIBRATION', reason=str(error)))
                return exploratory_prices(deepcopy(value['direct_prices']))

        if deferred or signal_holds:
            from .pf_saved_prices import independent_plan
            # Preserve calibration/holdout roles while admitting independently clear work.
            cells = [cell for cell in independent_plan(value,deferred)['new_compile_cells']
                     if cell['role'] != 'HOLDOUT']
            def audits(result):
                from .pr_candidate_completion import Held

                for audit in original_audits:
                    try:
                        target, compiled = stage_original_audit(root, workspace, folder, value, audit)
                        result.extend(call('audit', 'PF_AUDIT', stage_folder=target, compiled=compiled)['prices'])
                    except Held as error:
                        put(folder/'recovered_prices'/audit['compiled']['raw_commit']/'AUDIT_PENDING.json',
                            dict(reason=str(error)))
                if original_calibration:
                    compiled = reuse_calibration(root, workspace, folder, value,
                        original_calibration['rounds'], original_calibration)
                    put(folder/'SAVED_CALIBRATION_REUSE.json', original_calibration)
                    track_collected_prices(workspace, compiled)
                    result.extend(call('audit', 'PF_AUDIT', compiled=compiled)['prices'])
                elif cells:
                    compiled = compile_stage('PF_COMPILE', 'calibration', cells)
                    track_collected_prices(workspace, compiled)
                    result.extend(call('audit', 'PF_AUDIT', compiled=compiled)['prices'])
            can_signal = saved_signal or not signal_holds
            audited = []
            def audit_prices():
                from .pr_candidate_completion import Held
                try:
                    audits(audited)
                except Held as error:
                    put(folder/'EXPLORATORY_PRICE_ISSUE.json', dict(stage='PF_CALIBRATION', reason=str(error)))
                return exploratory_prices(deepcopy(value['direct_prices']) + audited)
            with ThreadPoolExecutor(max_workers=1) as pool:
                signal = pool.submit(signals) if can_signal and (
                    exploratory or value.get('new_signal_means', 0) and not saved_signal) else None
                if exploratory:
                    prices_source = available_prices(audit_prices)
                else:
                    audits(audited)
                saved_signal_result = signal.result() if signal is not None else None
            if exploratory and can_signal:
                registered = local('register', prices=prices_source,
                    registry=str(Path(selected['registry_file']).relative_to(workspace)),
                    binding_view=selected.get('binding_view'))
                registered.update(deferred_price_rounds=deferred, original_audits=original_audits,
                                  audited_independent_prices=audited,
                                  original_owner_directories=sorted({str(folder),
                                      *owner_row.get('original_owner_directories', [])}))
                items.append(registered)
                continue
            commits = {row['price_evidence']['source']['commit'] for row in audited}
            if saved_signal_result:
                commits.add(saved_signal_result['publication'])
            if commits:
                subprocess.run(['git','fetch','--no-tags','--no-write-fetch-head',str(workspace),*sorted(commits)],
                               cwd=root,check=True,capture_output=True)
            items.append(dict(status='PRICE_EVIDENCE_REQUIRED', configuration_id=item['configuration_id'],
                              reason=('Original PF signal owner requires reconciliation' if signal_holds and not can_signal
                                      else 'Outstanding PF price receipts require reconciliation'),
                              deferred_price_rounds=deferred, audited_independent_prices=audited,
                              original_calibration=original_calibration,
                              original_signal_holds=signal_holds, original_audits=original_audits,
                              original_owner_directories=sorted({str(folder),
                                  *owner_row.get('original_owner_directories', [])}),
                              independent_signal=saved_signal_result or saved_signal))
            continue
        if saved_signal:
            signals()
            price_source=available_prices(prices)
        else:
            # The native PF contract has no signal -> calibration dependency.
            with ThreadPoolExecutor(max_workers=2)as pool:
                signal=pool.submit(signals);price=pool.submit(prices)
                signal.result();price_source=available_prices(price.result)
        registered = local('register',prices=price_source,
            registry=str(Path(selected['registry_file']).relative_to(workspace)),
            binding_view=selected.get('binding_view'))
        if scoped:
            registered['original_owner_directories'] = sorted({str(folder), *owner_row['original_owner_directories']})
        items.append(registered)
    # The usual registry loader consumes the original producer locators. Fetch
    # the owned clone's objects, without changing the controller's HEAD.
    subprocess.run(['git','fetch','--no-tags','--no-write-fetch-head',str(workspace),'HEAD'],cwd=root,check=True,
                   stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    registry_path = Path(selected['registry_file'])
    registry = read(registry_path) if registry_path.exists() else dict(schema='D_PF_EVIDENCE_REGISTRY_V1',configurations={})
    put(registry_output,registry)
    put(output,dict(response,items=items,registry_file=registry_output))


def terminal_model_prices(parent, config, method, table, model, physical_id):
    """Extend the parent's adopted width-specific affine model without refitting."""
    from .pf_optimizer_connection import AXES
    from .price_model_validation import predict
    from .saved_rpe_cost_catalog import enc, frac

    scope = 'CONDITIONAL_COMPARISON_MODEL_WITH_EXPLICIT_LONG_Q_UNKNOWN'
    kind = 'CONDITIONAL_AFFINE_PRICE_REFERENCE'
    gate = 'MODEL_REFERENCE_LONG_Q_UNCERTAINTY_UNKNOWN'
    exploratory = table.get('scope') == 'EXPLORATORY_CALIBRATION_ONLY'
    if (table.get('scope') not in (scope, 'EXPLORATORY_CALIBRATION_ONLY') or table.get('configuration_id') != parent['id']
            or table.get('qualified_for_candidate') is not False
            or model['configuration_id'] != parent['id'] or model['holdout_refit'] is not False
            or model['long_q_error_bound'] is not None):
        raise ValueError('Terminal PF has no matching adopted long-q model')
    terminal = config['rounds'][-1]
    prior = parent['rounds'][-1]
    width = frac(parent['spec']['delta_pf'])
    q = terminal['q_PF']
    if (terminal['m'] != config['spec']['M'] or terminal['structure'] != prior['structure']
            or frac(terminal['T']) != q * width or frac(terminal['actual_delta_pf']) != width
            or q <= prior['q_PF']):
        raise ValueError('Terminal PF round differs from the adopted model schedule')
    prefix = {(row['m'], row['axis']): row for row in table['axes']}
    holdouts = {row['axis']: row for row in table['holdout']}
    prices = []
    for axis in AXES:
        axis_model = model['models'][axis]
        domain = axis_model['domain']
        evidence = dict(source=table['model'], axis=axis)
        reference = prefix[prior['m'], axis]
        if exploratory:
            if (axis_model['model'] != 'PRICE_LINEAR_FEATURE_MODEL_V1'
                    or axis_model['features'] != ['one', 'q'] or domain['route'] != method
                    or domain['axis'] != axis or domain['profile'] != parent['spec']['profile']):
                raise ValueError('Terminal PF calibration model differs from '+axis)
            estimate = predict(axis_model, dict(one=1, q=q))
            if estimate <= 0:
                raise ValueError('Terminal PF model produced a nonpositive price for '+axis)
            prices.append(dict(m=terminal['m'], axis=axis, physical_id=physical_id,
                unit_price_RZ=enc(estimate), kind='EXPLORATORY_AFFINE_PRICE_REFERENCE',
                price_evidence=evidence, gate='CALIBRATION_ONLY', q=q, T=terminal['T'],
                actual_delta_pf=terminal['actual_delta_pf'], unvalidated_extrapolation=True,
                model_error_bound_RZ=None))
            continue
        holdout = holdouts[axis]
        observed = prefix[model['holdout_round'], axis]
        if (axis_model['model'] != 'PRICE_LINEAR_FEATURE_MODEL_V1'
                or axis_model['features'] != ['one', 'q']
                or domain['route'] != method or domain['axis'] != axis
                or domain['kind'] != 'DIRECT_COMPILED_DETERMINISTIC'
                or domain['profile'] != parent['spec']['profile']
                or reference['kind'] != kind or reference['gate'] != gate
                or reference['price_evidence'] != evidence or reference['q'] != prior['q_PF']
                or frac(reference['actual_delta_pf']) != width
                or holdout['m'] != model['holdout_round'] or holdout['q'] >= q
                or observed['gate'] != 'C_IMPLEMENTATION_GUARD_PASS'
                or observed['price_evidence'] != holdout['receipt']):
            raise ValueError('Terminal PF model, admitted prefix or holdout mismatch for '+axis)
        estimate = predict(axis_model, dict(one=1, q=q))
        if estimate <= 0:
            raise ValueError('Terminal PF model produced a nonpositive price for '+axis)
        prices.append(dict(m=terminal['m'], axis=axis, physical_id=physical_id,
            unit_price_RZ=enc(estimate), kind=kind, price_evidence=evidence, gate=gate,
            q=q, T=terminal['T'], actual_delta_pf=terminal['actual_delta_pf'],
            unvalidated_extrapolation=True, model_error_bound_RZ=None))
    return prices


def complete_terminal_prices(root, runtime, parent_entries, *, stop_requested, on_ready, selected=None):
    """Register 14-round evidence using the same price rule as the admitted parent."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from hashlib import sha256
    from .pf_optimizer_connection import AXES, physical_ids
    from .pr_candidate_completion import Held
    from .rpe_release_registry import write

    if selected is None:
        selected = runtime.settings.get('pf_terminal_price_leaves', {})
    state_path = runtime.executor.run_dir/'pf_auto'/'TERMINAL_PRICES.json'
    states = read(state_path) if state_path.is_file() else {}

    def fixed(root, loc):
        raw = subprocess.check_output(['git','show',loc['commit']+':'+loc['path']],cwd=root,stderr=subprocess.PIPE)
        value = json.loads(raw)
        for key in loc.get('selector', ()):
            value = value[key]
        return value

    def run(parent_id, leaf):
        workspace = Path(leaf['workspace']).resolve()
        if not workspace.is_relative_to(runtime.executor.run_dir.resolve()):
            raise ValueError('Terminal PF workspace leaves the current run')
        folder = workspace/leaf['relative_output']
        cfg = fixed(workspace, leaf['configuration'])
        parent_entry = parent_entries[parent_id]
        parent = fixed(root, parent_entry['configuration'])
        if (parent['id'] != parent_id or cfg['spec']['M'] != 13
                or dict(cfg['spec'], M=12) != parent['spec']):
            raise ValueError('Terminal PF scope differs from the admitted parent')
        ids = physical_ids(root if leaf.get('terminal_reference') else workspace, cfg)
        if ids[13] != leaf['physical_id'] or {m:ids[m] for m in range(13)} != physical_ids(root, parent):
            raise ValueError('Terminal PF physical schedule differs from its fixed receipt')
        parent_table = fixed(root, parent_entry['prices'])
        axes = parent_table['axes']
        if ({(x['m'],x['axis']) for x in axes if ids[x['m']]==x['physical_id']}
                != {(m,axis) for m in range(13) for axis in AXES} or len(axes)!=26):
            raise ValueError('Terminal PF needs 26 matching prefix prices')
        if stop_requested():
            return dict(status='WAITING_STOP')
        plan_path = folder/'PLAN.json'
        if sha256(plan_path.read_bytes()).hexdigest() != leaf['plan_sha256']:
            raise ValueError('Terminal PF plan changed')
        if read(plan_path)['configuration'] != cfg:
            raise ValueError('Terminal PF plan differs from the fixed configuration')
        if leaf['entry']['price_scope'] != parent_table.get('scope'):
            raise ValueError('Terminal PF price scope differs from the admitted parent')
        if not parent_table.get('model'):
            raise ValueError('Terminal PF requires the adopted parent price model')
        model = fixed(root, parent_table['model'])
        prices = terminal_model_prices(parent, cfg, parent_entry['method'], parent_table, model, ids[13])
        for row in axes:
            if 'configuration_id' in row:
                row['configuration_id'] = cfg['id']
        out = folder/'AXIS_PRICES_MODEL.json'
        table = dict(parent_table, configuration_id=cfg['id'], axes=axes+prices)
        if out.exists():
            if read(out) != table:
                raise ValueError('Previously registered terminal PF price table differs')
        else:
            put(out, table)
        relative = str(out.relative_to(workspace))
        git(workspace, 'add', '--sparse', '--', relative)
        if git(workspace,'diff','--cached','--name-only','--',relative):
            git(workspace,'commit','-m','Record PF terminal prices from the adopted parent model','--',relative)
        tip=git(workspace,'log','-1','--format=%H','--',relative)
        subprocess.run(['git','fetch','--no-tags','--no-write-fetch-head',str(workspace),
                        tip+':refs/rpe-preparation/pf-terminal/'+cfg['id']],
                       cwd=root,check=True,capture_output=True)
        entry=dict(leaf['entry'], prices=dict(commit=tip,path=relative))
        return dict(status='READY', configuration_id=cfg['id'], entry=entry, parent_prices=parent_entry['prices'],
                    verified_terminal_axes=dict(m=13, physical_id=ids[13], axes=list(AXES)))

    def process(item):
        parent_id, leaf = item
        try:
            return parent_id, run(parent_id, leaf)
        except (Held, ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
            return parent_id, dict(status='WAITING', reason=str(error))

    pending = []
    for key, leaf in selected.items():
        parent_entry = parent_entries.get(key, {})
        if not parent_entry.get('prices'):
            continue
        previous = states.get(key, {})
        locs = {field:parent_entry.get(field) for field in ('configuration','signal_packet')}
        if (previous.get('status') == 'READY' and previous.get('parent_prices') == parent_entry['prices']
                and previous['entry'].get('horizon_parent') == locs):
            on_ready(previous['configuration_id'], previous['entry'])
        else:
            # Older receipts lack the parent table locator. Recreate its checked
            # model quote against the immutable output before attaching it.
            pending.append((key, leaf))
    if not pending:
        return
    with ThreadPoolExecutor(max_workers=len(pending),thread_name_prefix='pf-terminal-prices') as pool:
        for future in as_completed(pool.submit(process,item) for item in pending):
            parent_id, state = future.result()
            prior = states.get(parent_id, {})
            if state['status'] != 'READY':
                if prior.get('status') == 'READY':
                    state['previous_adoption'] = prior
                elif 'previous_adoption' in prior:
                    state['previous_adoption'] = prior['previous_adoption']
            states[parent_id] = state
            write(state_path,states)
            if state['status']=='READY':
                on_ready(state['configuration_id'], state['entry'])
