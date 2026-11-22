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


def read(path):return json.loads(Path(path).read_text())
def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def git(root,*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()


def selected_workspace_request(request, workspace):
    destination = workspace/'docs/research/distributed_selected_pf'
    return dict(request, output_root=str(destination), registry_file=str(destination/'REGISTRY.json'))


def reuse_signal(root, workspace, folder, plan, saved):
    """Carry the original completed signal into an explicitly scoped continuation."""
    from .pr_candidate_completion import Held
    relative=folder.relative_to(workspace).as_posix()
    commit=saved['commit']
    def blob(name):
        return subprocess.check_output(['git','show',commit+':'+relative+'/'+name],cwd=root)
    try:
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
        subprocess.run(['git','fetch','--no-tags','--no-write-fetch-head',str(root),commit],cwd=workspace,check=True,
                       stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
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
    except (OSError,KeyError,ValueError,subprocess.CalledProcessError) as error:
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
    saved_signals = {}
    if inherited_registry is not None:
        # Resume the exact owner check and plan already bound to this workspace.
        request = read(owner_input)
    elif settings.get('recover_saved_price_rounds', True):
        from .pf_saved_prices import candidate_roots, completed_partial, price_indexes, prior_waves
        price_index = run/'SAVED_PRICE_INDEX.json'
        earlier_waves = prior_waves(wave_root.parent, spec)
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
                                saved_signals[cfg] = dict(commit=read(result)['publication'])
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
        subprocess.run([spec['pf_completion'].get('interpreter',sys.executable),'-B',
            str(metadata_root/'scripts/check_pf_preparation_owner.py'),'--provider-root',str(original),
            '--request',str(owner_input),'--output',str(owner_output),'--signal-index',str(signal_index),
            *price_args],check=True,env=metadata_env)
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
        deferred = owner_rows[item['configuration_id']].get('deferred_price_rounds', {})
        original_calibration = owner_rows[item['configuration_id']].get('original_calibration')
        if price_args:
            # Only the admitted prices actually selected by this plan need
            # object delivery to tasks running in the owned producer clone.
            for commit in sorted({row['price_evidence']['source']['commit'] for row in value['direct_prices']
                                  if row['price_evidence']['admission']['path'].endswith('/GUARD_CHECK.json')}):
                subprocess.run(['git','fetch','--no-tags','--no-write-fetch-head',str(root),commit],
                               cwd=workspace,check=True,capture_output=True)
        if (folder/'STOP.json').exists():raise ValueError('Preserve consumed PF STOP; no automatic retry')
        source=dict(commit=fixed['source'],caller='scripts/prepare_pf_candidate.py')
        def local(phase,**fields):
            phase_in=folder/(phase+'.METADATA_INPUT.json');phase_out=folder/(phase+'.METADATA_RESULT.json')
            put(phase_in,dict(value=value,relative_output=str(folder.relative_to(workspace)),
                phase=phase,source=source,**fields))
            if not phase_out.exists():
                # Fresh interpreter: the D provider's package must not mix
                # with the controller's already imported trottertracks.
                subprocess.run([spec['pf_completion'].get('interpreter',sys.executable),'-B',
                    str(metadata_root/'scripts/run_pf_preparation_stage.py'),'--provider-root',str(workspace),
                    '--input',str(phase_in),'--output',str(phase_out)],cwd=workspace,check=True,env=metadata_env)
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

        def call(phase,tag,**fields):
            stage_in=folder/(tag+'.INPUT.json');stage_out=folder/(tag+'.RESULT.json')
            stage_input = stage_input_for(phase, **fields)
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
                result = recovered_audit(folder, tag, stage_input, fields['compiled'])
                if result is not None:
                    return result
            return runtime.call(context,cmd,folder,tag,stage_out,science=True)
        def compile_inputs(cohort, cells):
            fields = dict(cohort=cohort, rounds=[cell['m'] for cell in cells])
            if value.get('selected_axis_compile'):
                fields['axes_by_round'] = {str(cell['m']): cell['axes'] for cell in cells}
            return fields

        def compile_stage(tag, cohort, cells):
            return call('compile', tag, **compile_inputs(cohort, cells))

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
        saved_signal = (saved_for_configuration(settings, 'saved_signal', value['configuration']['id'])
                        or request.get('independent_saved_signals', {}).get(value['configuration']['id']))

        def signals():
            if saved_signal:
                reuse_signal(root,workspace,folder,value,saved_signal)
                return None
            return call('signal','PF_SIGNALS')

        if deferred:
            from .pf_saved_prices import independent_plan
            # Preserve calibration/holdout roles while admitting independently clear work.
            cells = [cell for cell in independent_plan(value,deferred)['new_compile_cells']
                     if cell['role'] != 'HOLDOUT']
            with ThreadPoolExecutor(max_workers=1) as pool:
                signal = pool.submit(signals) if value.get('new_signal_means', 0) and not saved_signal else None
                audited = []
                if cells or original_calibration:
                    if original_calibration:
                        compiled = reuse_calibration(root, workspace, folder, value,
                            original_calibration['rounds'], original_calibration)
                        put(folder/'SAVED_CALIBRATION_REUSE.json', original_calibration)
                    else:
                        compiled = compile_stage('PF_COMPILE', 'calibration', cells)
                    track_collected_prices(workspace, compiled)
                    audited = call('audit', 'PF_AUDIT', compiled=compiled)['prices']
                saved_signal_result = signal.result() if signal is not None else None
            commits = {row['price_evidence']['source']['commit'] for row in audited}
            if saved_signal_result:
                commits.add(saved_signal_result['publication'])
            if commits:
                subprocess.run(['git','fetch','--no-tags','--no-write-fetch-head',str(workspace),*sorted(commits)],
                               cwd=root,check=True,capture_output=True)
            items.append(dict(status='PRICE_EVIDENCE_REQUIRED', configuration_id=item['configuration_id'],
                              reason='Outstanding PF price receipts require reconciliation',
                              deferred_price_rounds=deferred, audited_independent_prices=audited,
                              original_calibration=original_calibration,
                              independent_signal=saved_signal_result or saved_signal))
            continue
        if saved_signal:
            signals()
            price_source=prices()
        else:
            # The native PF contract has no signal -> calibration dependency.
            with ThreadPoolExecutor(max_workers=2)as pool:
                signal=pool.submit(signals);price=pool.submit(prices)
                signal.result();price_source=price.result()
        items.append(local('register',prices=price_source,
            registry=str(Path(selected['registry_file']).relative_to(workspace)),
            binding_view=selected.get('binding_view')))
    # The usual registry loader consumes the original producer locators. Fetch
    # the owned clone's objects, without changing the controller's HEAD.
    subprocess.run(['git','fetch','--no-tags','--no-write-fetch-head',str(workspace),'HEAD'],cwd=root,check=True,
                   stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    registry_path = Path(selected['registry_file'])
    registry = read(registry_path) if registry_path.exists() else dict(schema='D_PF_EVIDENCE_REGISTRY_V1',configurations={})
    put(registry_output,registry)
    put(output,dict(response,items=items,registry_file=registry_output))


def complete_terminal_prices(root, runtime, parent_entries, *, stop_requested, on_ready):
    """Run the fixed terminal price leaves and register their full 14-round evidence."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from hashlib import sha256
    from .pf_optimizer_connection import AXES, physical_ids
    from .pr_candidate_completion import Held
    from .rpe_release_registry import write

    selected = runtime.settings.get('pf_terminal_price_leaves', {})
    adapter = Path(root)/'scripts/run_pf_preparation_stage.py'
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
        ids = physical_ids(workspace, cfg)
        if ids[13] != leaf['physical_id'] or {m:ids[m] for m in range(13)} != physical_ids(root, parent):
            raise ValueError('Terminal PF physical schedule differs from its fixed receipt')
        axes = deepcopy(fixed(root, parent_entry['prices'])['axes'])
        if ({(x['m'],x['axis']) for x in axes if ids[x['m']]==x['physical_id']}
                != {(m,axis) for m in range(13) for axis in AXES} or len(axes)!=26):
            raise ValueError('Terminal PF needs 26 matching prefix prices')
        for row in axes:
            if 'configuration_id' in row:
                row['configuration_id'] = cfg['id']
        if stop_requested():
            return dict(status='WAITING_STOP')
        plan_path = folder/'PLAN.json'
        if sha256(plan_path.read_bytes()).hexdigest() != leaf['plan_sha256']:
            raise ValueError('Terminal PF compile plan changed')
        value = read(plan_path)
        manifest = leaf['provider']
        source = leaf['execution_source']
        if value['configuration'] != cfg or {cell['m']:cell['axes'] for cell in value['new_compile_cells']} != {13:['X','Y']}:
            raise ValueError('Terminal PF request must compile only the fixed X/Y pair')
        context = SimpleNamespace(directory=folder, settings={'selected_axis_compile':True},
            source=manifest['commit'], preparation_root=workspace,
            preparation_source_manifest=manifest, log=lambda **row:None)
        def stage(kind, name, fields):
            input_path = folder/(name+'.INPUT.json')
            output = folder/(name+'.RESULT.json')
            body = dict(value=value, relative_output=leaf['relative_output'], phase=kind, source=source, **fields)
            if input_path.exists() and read(input_path) != body:
                raise ValueError('Terminal PF stage request changed on resume')
            if not input_path.exists():
                put(input_path, body)
            command=[leaf['interpreter'],'-B',str(adapter),'--provider-root',str(workspace),
                '--input',str(input_path),'--output',str(output)]
            return runtime.call(context, command, folder, name, output, science=True)
        compiled = stage('compile', 'PF_HOLDOUT_COMPILE', dict(cohort='terminal', rounds=[13], axes_by_round={'13':['X','Y']}))
        track_collected_prices(workspace, compiled)
        audited = stage('audit', 'PF_HOLDOUT_AUDIT', dict(compiled=compiled))
        prices = audited['prices']
        if (audited['status'] != 'AUDIT_PASS' or len(prices)!=2
                or {(row['m'],row['axis'],row['physical_id'],row['gate']) for row in prices} !=
                    {(13,axis,ids[13],'C_IMPLEMENTATION_GUARD_PASS') for axis in AXES}):
            raise ValueError('Terminal audit did not confirm both fixed axes')
        out = folder/'AXIS_PRICES.json'
        table = dict(axes=axes+prices)
        if out.exists():
            if read(out) != table:
                raise ValueError('Previously registered terminal PF price table differs')
        else:
            put(out, table)
        relative = str(out.relative_to(workspace))
        git(workspace, 'add', '--sparse', '--', relative)
        if git(workspace,'diff','--cached','--name-only','--',relative):
            git(workspace,'commit','-m','Record admitted PF terminal X/Y prices','--',relative)
        tip=git(workspace,'log','-1','--format=%H','--',relative)
        subprocess.run(['git','fetch','--no-tags','--no-write-fetch-head',str(workspace),
                        tip+':refs/rpe-preparation/pf-terminal/'+cfg['id']],
                       cwd=root,check=True,capture_output=True)
        entry=dict(leaf['entry'], prices=dict(commit=tip,path=relative))
        return dict(status='READY', configuration_id=cfg['id'], entry=entry,
                    verified_terminal_axes=dict(m=13, physical_id=ids[13], axes=list(AXES)))

    def process(item):
        parent_id, leaf = item
        try:
            return parent_id, run(parent_id, leaf)
        except (Held, ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
            return parent_id, dict(status='WAITING', reason=str(error))

    pending = [(key, item) for key,item in selected.items() if key in parent_entries and
               states.get(key,{}).get('status')!='READY']
    if not pending:
        return
    with ThreadPoolExecutor(max_workers=len(pending),thread_name_prefix='pf-terminal-prices') as pool:
        for future in as_completed(pool.submit(process,item) for item in pending):
            parent_id, state = future.result()
            states[parent_id] = state
            write(state_path,states)
            if state['status']=='READY':
                on_ready(state['configuration_id'], state['entry'])
