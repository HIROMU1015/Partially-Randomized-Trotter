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
import subprocess
import sys


def read(path):return json.loads(Path(path).read_text())
def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def git(root,*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()


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


def complete_selected(root,spec,run,request,entry,output,runtime):
    """Original finite PF contract; no independent candidate/shot selection."""
    original=Path(entry).resolve().parents[1];run=Path(run)
    owner_input=run/'ORIGINAL_OWNER_REQUEST.json';owner_output=run/'ORIGINAL_OWNER_CHECK.json'
    put(owner_input,request)
    subprocess.run([spec['pf_completion'].get('interpreter',sys.executable),'-B',
        str(Path(root)/'scripts/check_pf_preparation_owner.py'),'--provider-root',str(original),
        '--request',str(owner_input),'--output',str(owner_output)],check=True)
    owner=read(owner_output)
    held=[row for row in owner['rows']if row['status']=='PRIOR_OWNER_HOLD']
    if held:
        if not Path(request['registry_file']).exists():put(request['registry_file'],dict(schema='D_PF_EVIDENCE_REGISTRY_V1',configurations={}))
        put(output,dict(schema='D_PF_EVIDENCE_COMPLETION_V1',items=held,registry_file=request['registry_file']))
        return
    reused=[row for row in owner['rows']if row['status']=='REUSE_COMPLETE_ORIGINAL_OWNER']
    if len(reused)==len(owner['rows']):
        registry=read(request['registry_file'])if Path(request['registry_file']).exists()else dict(schema='D_PF_EVIDENCE_REGISTRY_V1',configurations={})
        for row in reused:registry['configurations'][row['configuration_id']]=row['entry']
        subprocess.run(['git','fetch','--no-tags',str(original),'HEAD'],cwd=root,check=True,
            stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        put(request['registry_file'],registry);put(output,dict(schema='D_PF_EVIDENCE_COMPLETION_V1',items=reused,registry_file=request['registry_file']))
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
    selected=deepcopy(request)
    selected['output_root']=str(workspace/'docs/research/distributed_selected_pf')
    selected['registry_file']=str(workspace/'docs/research/distributed_selected_pf/REGISTRY.json')
    inp=workspace/'docs/research/distributed_selected_pf/REQUEST.json'
    plan_output=run/'PLAN_RESPONSE.json'
    if not inp.exists():put(inp,selected)
    elif read(inp)!=selected:raise ValueError('PF selected contract changed')
    if not plan_output.exists():
        subprocess.run([spec['pf_completion'].get('interpreter',sys.executable),'-B',
            str(workspace/'scripts/prepare_pf_candidate.py'),'--request',str(inp),'--output',str(plan_output),
            '--mode','plan'],cwd=workspace,check=True)
    response=read(plan_output);items=[]
    for item in response['items']:
        if item['status']=='REUSED_COMPLETED_REGISTRY':items.append(item);continue
        folder=Path(item['output']);value=read(folder/'PLAN.json')
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
                    str(Path(root)/'scripts/run_pf_preparation_stage.py'),'--provider-root',str(workspace),
                    '--input',str(phase_in),'--output',str(phase_out)],cwd=workspace,check=True)
            return read(phase_out)
        source=local('preflight')['source']
        context=SimpleNamespace(directory=folder,settings=spec.get('pf_completion',{}),
            source=fixed['source'],preparation_root=workspace,log=lambda **row:None,
            preparation_source_manifest=dict(commit=fixed['source'],files={name:__import__('hashlib').sha256(
                subprocess.check_output(['git','show',fixed['source']+':'+name],cwd=workspace)).hexdigest()
                for name in ['scripts/prepare_pf_candidate.py','src/trottertracks/pf_evidence_preparation.py',
                    'scripts/run_pf_m0_prices.py','scripts/run_pf_p7_h_prices.py','scripts/run_truncated_p7_prices.py']}))
        def call(phase,tag,**fields):
            stage_in=folder/(tag+'.INPUT.json');stage_out=folder/(tag+'.RESULT.json')
            put(stage_in,dict(value=value,relative_output=str(folder.relative_to(workspace)),
                             phase=phase,source=source,**fields))
            cmd=[spec['pf_completion'].get('interpreter',sys.executable),'-B',
                str(Path(root)/'scripts/run_pf_preparation_stage.py'),'--provider-root',str(workspace),
                '--input',str(stage_in),'--output',str(stage_out)]
            return runtime.call(context,cmd,folder,tag,stage_out,science=True)
        def prices():
            values=deepcopy(value['direct_prices'])
            calibration=[r['m']for r in value['new_compile_cells']if r['role']!='HOLDOUT']
            if calibration:
                compiled=call('compile','PF_COMPILE',cohort='calibration',rounds=calibration)
                values+=call('audit','PF_AUDIT',compiled=compiled)['prices']
            frozen=call('freeze','PF_MODEL_FREEZE',prices=values)['model']
            holdout=[r['m']for r in value['new_compile_cells']if r['role']=='HOLDOUT']
            if holdout:
                compiled=call('compile','PF_HOLDOUT_COMPILE',cohort='holdout',rounds=holdout)
                values+=call('audit','PF_HOLDOUT_AUDIT',compiled=compiled)['prices']
            return call('finish','PF_FINISH',prices=values,model=frozen)['prices']
        saved_signal=spec['pf_completion'].get('saved_signal')
        if saved_signal:
            reuse_signal(root,workspace,folder,value,saved_signal)
            price_source=prices()
        else:
            # The native PF contract has no signal -> calibration dependency.
            with ThreadPoolExecutor(max_workers=2)as pool:
                signal=pool.submit(call,'signal','PF_SIGNALS');price=pool.submit(prices)
                signal.result();price_source=price.result()
        items.append(local('register',prices=price_source,
            registry=str(Path(selected['registry_file']).relative_to(workspace))))
    # The usual registry loader consumes the original producer locators. Fetch
    # the owned clone's objects, without changing the controller's HEAD.
    subprocess.run(['git','fetch','--no-tags',str(workspace),'HEAD'],cwd=root,check=True,
                   stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    put(request['registry_file'],read(selected['registry_file']))
    put(output,dict(response,items=items,registry_file=request['registry_file']))
