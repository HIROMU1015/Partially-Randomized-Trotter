"""Request-driven Pauli calibration -> unchanged audit -> owned PriceRules registry.

No candidate selection, signal or Q3. Default mode prepares metadata. --execute
runs a single supported same-step S2 calibration through A/B2's fixed leaves.
Each actual step is acquired and audited separately, then registered with its
covered rounds; an existing rule is never silently replaced.
"""
from copy import deepcopy
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.rpe_preparation_paths import runtime_request,resolve as resolve_runtime_path
spec=importlib.util.spec_from_file_location('_pauli_owned_signal_ops',ROOT/'scripts/run_pr_pauli_candidate_signals.py')
OPS=importlib.util.module_from_spec(spec);spec.loader.exec_module(OPS)
A_SOURCE='02e7cc6a127954fa7c446e7383d037f0617d50d3'
AUDIT_SOURCE='54b4353b9'
A_FILES=('src/trottertracks/pauli_calibration_request.py','scripts/prepare_pauli_calibration.py','scripts/run_pauli_split_cal.py')
SCOPED_AUDIT_FILES=('scripts/pauli_ld2_step_audit.py','scripts/pauli_ld3_cal_audit.py')
AUDIT_ADAPTER='scripts/pauli_split_cal_audit_request.py'
AUDIT_FILES=SCOPED_AUDIT_FILES+(AUDIT_ADAPTER,)  # legacy immutable audit contract


def selected(workspace,ref):
    data=json.loads(subprocess.check_output(['git','show',ref['commit']+':'+ref['path']],cwd=workspace))
    for key in ref.get('selector',[]):data=data[key]
    return data


def canonical_preparation(config,source):
    return dict(configuration=config,configuration_source=source,
        dictionary_source=config['spec']['split']['tail_dictionary']['source'],
        schedule=[{k:r[k]for k in ('m','q_RPE','T','q_PF','actual_delta_pf','R_outer','structure','mapping','law_binding')}
                  |{'physical_id':r['candidate']['id']}for r in config['rounds']])


def command(workspace,cmd,log,marker=None):
    if marker is not None:OPS.write(marker,dict(status='STARTING',command=cmd,UTC=OPS.utc(),automatic_retry=False))
    start=time.monotonic()
    with Path(log).open('a')as stream:
        child=subprocess.Popen(cmd,cwd=workspace,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
                               stdout=stream,stderr=subprocess.STDOUT)
        if marker is not None:OPS.write(marker,dict(status='RUNNING',pid=child.pid,command=cmd,UTC=OPS.utc(),automatic_retry=False))
        code=child.wait()
    if marker is not None:OPS.write(marker.with_name(marker.stem+'_RETURN.json'),dict(returncode=code if code>=0 else None,
        signal=-code if code<0 else None,wall_seconds=time.monotonic()-start,UTC=OPS.utc(),automatic_retry=False))
    return code


def requote(request,config,implementation,out):
    script='''import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/'src'))
from trottertracks.pr_price_rules import PriceRules,GitReader
reg=json.loads(Path(sys.argv[2]).read_text());cfg=json.loads(Path(sys.argv[3]).read_text());imp=json.loads(sys.argv[4])
r=PriceRules(GitReader(sys.argv[1]),reg).quote(dict(configuration=cfg,implementation=imp,policy='MODEL'))
Path(sys.argv[5]).write_text(json.dumps(r,indent=2)+'\\n')
'''
    cfg=out.with_name('REQUOTE_CONFIGURATION.json');OPS.fixed_file(cfg,config)
    subprocess.run([request.get('quote_interpreter',sys.executable),'-B','-c',script,request['d2_view'],
        str(Path(request['registry_file']).resolve()),str(cfg),json.dumps(implementation),str(out)],check=True,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    return OPS.read(out)


def public_files(base):
    return [p for p in base.rglob('*')if p.is_file()and 'work' not in p.relative_to(base).parts]


def _publish_compile_binding(workspace, output):
    """Publish the fixed inputs and binding; raw results stay in the artifact queue."""
    from pathlib import PurePosixPath

    base = os.environ['RPE_PAULI_BINDING_PUBLIC_BASE']
    allowed = {'CONFIGURATION.json', 'PREPARATION.json', 'INPUT_QUOTE.json', 'CAL_PREPARATION.json',
               'EXECUTION_OPS.json', 'GROUP_CALIBRATION_REQUEST.json', 'BINDING.json',
               'PREPARE.log', 'PREPARE_EXECUTION.log', 'METADATA.log', 'PREFLIGHT.log', 'FREEZE.log'}
    prefix = PurePosixPath('docs/research/d2_automatic_price_20261116')
    destination = PurePosixPath(output.relative_to(workspace).as_posix())
    history = base + '..HEAD'
    changed = OPS.git(workspace, 'log', '--format=', '--name-only', history).splitlines()
    paths = {PurePosixPath(path) for path in changed if path}
    if (not destination.is_relative_to(prefix) or not paths or
            OPS.git(workspace, 'rev-list', '--merges', history) or
            not any(path.name == 'BINDING.json' for path in paths) or
            any(not PurePosixPath(path).is_relative_to(destination) or
                PurePosixPath(path).name not in allowed for path in paths)):
        raise ValueError('Pauli publication contains files outside the compile binding')
    OPS.publish(workspace)


def _run_one(request,*,execute=False,push=True,phase='all',publish_binding=False):
    if phase not in ('all','compile','audit'):raise ValueError('Unknown preparation phase')
    request=runtime_request(request,('workspace','private_root','d2_view','registry_file','interpreter','quote_interpreter'))
    workspace=Path(request['workspace']).resolve();relative=Path(request['directory'])
    if relative.is_absolute()or'..'in relative.parts:raise ValueError('OWNED_REPOSITORY_RELATIVE_DIRECTORY_REQUIRED')
    out=(workspace/relative).resolve();out.relative_to(workspace)
    config=request.get('configuration')or selected(workspace,request['configuration_source'])
    quote=request['price_quote'];cid=config['id'];prefix=config['spec']['L_D']
    if quote.get('configuration_id')!=cid or quote.get('policy')!='MODEL':raise ValueError('CANONICAL_MODEL_QUOTE_REQUIRED')
    if quote.get('complete'):
        return dict(schema='PR_PAULI_CANDIDATE_PRICES_V1',status='PRICES_READY',quote=quote,configuration_id=cid,
                    new_paths=0,new_compiles=0,reused_complete_quote=True)
    py=request.get('interpreter','/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python')
    with OPS.owned_checkout_lock(workspace):
        for source,files in ((A_SOURCE,A_FILES),(AUDIT_SOURCE,SCOPED_AUDIT_FILES if request.get('scoped_registration')else AUDIT_FILES)):
            for name in files:
                if(workspace/name).read_bytes()!=subprocess.check_output(['git','show',source+':'+name],cwd=workspace):
                    raise ValueError('FIXED_CALIBRATION_OR_AUDIT_SOURCE_DIFFERS:'+name)
        if request.get('scoped_registration'):
            adapter_source=request.get('execution_source')
            if not adapter_source:raise ValueError('FIXED_SCOPED_AUDIT_ADAPTER_SOURCE_REQUIRED')
            if(workspace/AUDIT_ADAPTER).read_bytes()!=subprocess.check_output(['git','show',adapter_source+':'+AUDIT_ADAPTER],cwd=workspace):
                raise ValueError('FIXED_AUDIT_METADATA_SOURCE_DIFFERS')
        out.mkdir(parents=True,exist_ok=True)
        prep=request.get('prepared_item')or canonical_preparation(config,request.get('configuration_source'))
        if prep.get('configuration')!=config:raise ValueError('PREPARED_CONFIGURATION_DIFFERS')
        cfg=out/'CONFIGURATION.json';pf=out/'PREPARATION.json';qf=out/'INPUT_QUOTE.json';calfile=out/'CAL_PREPARATION.json'
        OPS.fixed_file(cfg,config);OPS.fixed_file(pf,prep);OPS.fixed_file(qf,quote)
        statefile=out/'PRICE_STATE.json';state=OPS.read(statefile)if statefile.exists()else{}
        result=dict(schema='PR_PAULI_CANDIDATE_PRICES_V1',configuration_id=cid,workspace=str(workspace),directory=str(relative),
                    new_paths=0,new_compiles=0,automatic_retry=False)
        if state.get('status')in('PRICES_READY','SCOPED_STEP_MODEL_READY'):
            return dict(state,resumed=True,new_compiles_this_invocation=0,new_paths_this_invocation=0)
        if str(state.get('status','')).startswith(('HELD','STOP')):
            return dict(state,resumed=True,reason=state.get('reason','NO_AUTOMATIC_RETRY'),
                        new_compiles_this_invocation=0,new_paths_this_invocation=0)
        if phase=='audit'and state.get('status')!='COMPILE_READY_FOR_AUDIT':
            return dict(result,status='HELD_AUDIT_COMPILE_DEPENDENCY_MISSING',
                        reason='Audit never acquires missing compile artifacts',new_compiles_this_invocation=0)
        if phase=='compile'and state.get('status')=='COMPILE_READY_FOR_AUDIT':
            return dict(state,resumed=True,new_compiles_this_invocation=0,new_paths_this_invocation=0)
        prepare=[py,'-B','scripts/prepare_pauli_calibration.py','--preparation',str(pf),'--price-result',str(qf),'--output',str(calfile)]
        if not calfile.exists():
            if command(workspace,prepare,out/'PREPARE.log')!=0:return dict(result,status='HELD_CALIBRATION_PREPARATION_FAILED')
        cal=OPS.read(calfile);groups=cal.get('calibration_requests',[])
        result.update(preparation_status=cal.get('status'),needs=cal.get('needs',[]),calibration_requests=groups,
                      implementation=cal.get('implementation'),preserved_price_rows=cal.get('reused_prices',[]))
        selected_id=request.get('group_request_id')
        if selected_id:
            groups=[g for g in groups if g['request_id']==selected_id]
            if len(groups)!=1:raise ValueError('SELECTED_CALIBRATION_GROUP_NOT_FOUND')
        if request.get('scoped_registration')and not execute:
            return dict(result,status='CALIBRATION_PLANNED_NOT_EXECUTED')
        if len(groups)>1:
            return dict(result,status='NEEDS_PRICE_RULE_MULTISTEP_REGISTRATION',
                        reason='Separate actual-step coefficients cannot be registered as one physical-family rule; no cells executed')
        if cal.get('needs')or len(groups)!=1:return dict(result,status='HELD_CALIBRATION_RULE_OR_INPUT',reason=cal.get('status'))
        group=groups[0]
        if group.get('saved_calibration_source')or group.get('status')=='REUSE_SAVED_CALIBRATION':
            return dict(result,status='REUSE_SAVED_CALIBRATION_NEEDS_MODEL_LOCATOR',
                        saved_calibration_source=group.get('saved_calibration_source'),
                        reason='No new path; supply the saved audit/model locator for registry re-use')
        registered={r.get('registered_rule')for r in quote.get('rows',[])if r.get('registered_rule')}
        if registered and not request.get('scoped_registration'):
            return dict(result,status='NEEDS_EXISTING_RULE_EXTENSION_WITH_PRESERVED_SOURCE',registered_rules=sorted(registered),
                        reason='Do not overwrite an existing same-family rule or attach one step coefficients to another step')
        if not execute:return dict(result,status='CALIBRATION_PLANNED_NOT_EXECUTED')
        if not state:
            execution=dict(branch=OPS.git(workspace,'branch','--show-current'),base=str(relative/'calibration'),
                           private=str(Path(request['private_root']).resolve()/'calibration'))
            OPS.write(out/'EXECUTION_OPS.json',execution)
            # The first metadata request has no output overrides, enabling saved-cell detection.
            if request.get('scoped_registration'):
                # Reuse A's exact group builder. Its --execution bulk mode
                # intentionally cannot assign one output directory to several
                # actual steps, so execute this already-selected group only.
                script='''import json,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path[:0]=[str(root/'src'),str(root/'scripts')]
from trottertracks import pauli_calibration_request as P
cfg=json.loads(Path(sys.argv[2]).read_text());group=json.loads(Path(sys.argv[3]).read_text());ops=json.loads(Path(sys.argv[4]).read_text())
spec,item,old=P.build_spec(root,cfg,json.loads(sys.argv[5]),execution=ops,covered_rounds=group['covered_rounds'])
if old is not None:raise ValueError('SAVED_CALIBRATION_MUST_NOT_BE_REACQUIRED')
Path(sys.argv[6]).write_text(json.dumps(dict(schema='A_PAULI_SPLIT_CAL_REQUEST_V1',driver_spec=spec),indent=2)+'\\n')
'''
                selected_file=out/'SELECTED_GROUP.json';OPS.fixed_file(selected_file,group)
                requestfile=out/'GROUP_CALIBRATION_REQUEST.json'
                commandline=[py,'-B','-c',script,str(workspace),str(cfg),str(selected_file),str(out/'EXECUTION_OPS.json'),
                    json.dumps(request.get('configuration_source')),str(requestfile)]
                if command(workspace,commandline,out/'PREPARE_EXECUTION.log')!=0:
                    return dict(result,status='HELD_EXECUTION_REQUEST_PREPARATION_FAILED')
                spec=OPS.read(requestfile)['driver_spec'];base=workspace/spec['base']
            else:
                if command(workspace,prepare+['--execution',str(out/'EXECUTION_OPS.json')],out/'PREPARE_EXECUTION.log')!=0:
                    return dict(result,status='HELD_EXECUTION_REQUEST_PREPARATION_FAILED')
                cal=OPS.read(calfile);group=cal['calibration_requests'][0];spec=group['driver_request']['driver_spec']
                requestfile=Path(group['driver_request_file']);base=workspace/spec['base']
            state=dict(result,status='EXECUTION_INPUTS_FIXED',driver_spec=spec,request_file=str(requestfile),implementation=cal['implementation'])
            inputs_commit=OPS.commit(workspace,[cfg,pf,qf,calfile,out/'EXECUTION_OPS.json',requestfile,
                out/'PREPARE.log',out/'PREPARE_EXECUTION.log'],'Fix selected Pauli calibration request')
            state['inputs_commit']=inputs_commit;OPS.publish(workspace,push);OPS.write(statefile,state)
        spec=state['driver_spec'];base=workspace/spec['base'];requestfile=Path(resolve_runtime_path(state['request_file']))
        driver=[py,'-B','scripts/run_pauli_split_cal.py','--request',str(requestfile)]
        if not(base/'BINDING.json').exists():
            for label,args in (('METADATA',['--metadata']),('PREFLIGHT',['--input-preflight',str(Path(request['private_root'])/'preflight')]),('FREEZE',['--freeze'])):
                if command(workspace,driver+args,out/(label+'.log'))!=0:
                    state.update(status='HELD_CALIBRATION_'+label);OPS.write(statefile,state);return state
        if 'binding_commit'not in state:
            state['binding_commit']=OPS.commit(workspace,[base/'BINDING.json',out/'METADATA.log',out/'PREFLIGHT.log',out/'FREEZE.log'],
                'Freeze selected Pauli calibration binding')
            if publish_binding:
                _publish_compile_binding(workspace, out)
            else:
                OPS.publish(workspace,push)
            OPS.write(statefile,state)
        marker=out/'CALIBRATION_STARTED.json';terminal=base/'run01/RESULT.json'
        if terminal.exists():
            if marker.exists()and OPS.live(OPS.read(marker),requestfile):
                return dict(state,status='IN_PROGRESS',reason='ORIGINAL_CALIBRATION_RUNNING')
            raw=OPS.read(terminal)
        elif marker.exists()or(base/'run01').exists()or Path(spec['private']).exists():
            if marker.exists()and OPS.live(OPS.read(marker),requestfile):
                return dict(state,status='IN_PROGRESS',reason='ORIGINAL_CALIBRATION_RUNNING')
            return dict(state,status='HELD_INCOMPLETE_PRIOR_CALIBRATION',reason='NO_AUTOMATIC_RETRY')
        else:
            code=command(workspace,driver+['--execute',state['binding_commit']],out/'EXECUTE.log',marker)
            if not terminal.exists():return dict(state,status='HELD_MISSING_CALIBRATION_TERMINAL',returncode=code)
            raw=OPS.read(terminal)
        if raw.get('binding',{}).get('commit')!=state['binding_commit']:
            raise ValueError('RAW_CALIBRATION_BINDING_SOURCE_DIFFERS')
        state.update(new_compiles=raw.get('compiles',0),new_paths=len(raw.get('summaries',[])),
                     execution_source=raw.get('source'),raw_status=raw.get('status'))
        if 'raw_commit'not in state:
            state['raw_commit']=OPS.commit(workspace,[base/'run01',marker,marker.with_name(marker.stem+'_RETURN.json'),out/'EXECUTE.log'],
                'Publish selected Pauli calibration raw receipts')
            OPS.publish(workspace,push);OPS.write(statefile,state)
        if raw.get('status')!='FINITE_LD%d_CAL_PATHS_COMPLETE'%prefix:
            state.update(status='HELD_CALIBRATION_GATE_OR_EXECUTION');OPS.write(statefile,state);return state
        if phase=='compile':
            state.update(status='COMPILE_READY_FOR_AUDIT',audit_executed=False)
            OPS.write(statefile,state);return state
        audit=[py,'-B','scripts/pauli_split_cal_audit_request.py','--request',str(requestfile),
               '--config',state['inputs_commit']+':'+str(relative/'CONFIGURATION.json')]
        audit_result=base/'audit_run01/AUDIT_RESULT.json';pricefile=base/'PRICE_26AXIS.json'
        if not pricefile.exists():
            if (out/'AUDIT_STARTED.json').exists():
                return dict(state,status='HELD_INCOMPLETE_PRIOR_AUDIT',reason='PRESERVE_PARTIAL_AUDIT_NO_AUTOMATIC_RETRY')
            OPS.write(out/'AUDIT_STARTED.json',dict(UTC=OPS.utc(),raw=state['raw_commit']))
            for label,args in (('BUILD',['--build',state['raw_commit']]),('RUN',['--run','--workers',str(request.get('audit_workers',2))]),('ANALYZE',['--analyze'])):
                if command(workspace,audit+args,out/('AUDIT_'+label+'.log'))!=0:
                    state.update(status='HELD_AUDIT_'+label);OPS.write(statefile,state);return state
        if not audit_result.exists():return dict(state,status='HELD_INCOMPLETE_AUDIT_TERMINAL')
        price=OPS.read(pricefile)
        if 'price_commit'not in state:
            state['price_commit']=OPS.commit(workspace,public_files(base)+[out/'AUDIT_STARTED.json',out/'AUDIT_BUILD.log',
                 out/'AUDIT_RUN.log',out/'AUDIT_ANALYZE.log'],'Publish selected Pauli decomposition audit and model')
            OPS.publish(workspace,push);OPS.write(statefile,state)
        state['price_source']=dict(commit=state['price_commit'],path=spec['base']+'/PRICE_26AXIS.json')
        if price.get('status')!='PRICE_RULE_SUPPORTED_MODEL_REFERENCE':
            state.update(status='HELD_AUDIT_RULE_NOT_SUPPORTED',reason=price.get('status'));OPS.write(statefile,state);return state
        if request.get('scoped_registration'):
            package=dict(kind='PR_PAULI_STEP_AUDIT',configuration_source=request['configuration_source'],
                implementation=state['implementation'],preparation_group=group,price_source=state['price_source'],
                audit_source=dict(commit=state['price_commit'],path=spec['base']+'/audit_run01/AUDIT_RESULT.json'),
                audit_inputs_source=dict(commit=state['price_commit'],path=spec['base']+'/INPUTS_AUDIT.json'),
                binding_source=dict(commit=state['binding_commit'],path=spec['base']+'/BINDING.json'),
                raw_source=dict(commit=state['raw_commit'],path=spec['base']+'/run01/RESULT.json'))
            state.update(status='SCOPED_STEP_MODEL_READY',evidence_packages=[package],covered_rounds=group['covered_rounds'],
                         legacy_registry_modified=False)
            OPS.write(statefile,state);return state
        registry_path=Path(request['registry_file'])
        with registry_path.with_suffix(registry_path.suffix+'.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            registry=OPS.read(registry_path)
            name='PAULI_LD%d'%prefix
            if any(e['name']==name for e in registry['entries']):name+='_'+cid[:12]
            if not any(e.get('price_source')==state['price_source']for e in registry['entries']):
                R=spec['cell']['R'];registry['entries'].append(dict(name=name,
                    configuration_source=dict(commit=state['inputs_commit'],path=str(relative/'CONFIGURATION.json')),
                    implementation=state['implementation'],direct_sources=[],price_source=state['price_source'],model_source=state['price_source'],
                    mode='PAULI_AUDIT',calibration_domain=dict(q=[8,8],R=[R,R]),calibration_R=[R],
                    caveats=['Candidate-specific S0-r1 decomposition; all round prices are finite-cohort model applications'],
                    calibration_evidence=dict(state['price_source'],selector=['calibration_cell','rows']),
                    registration=dict(by='D2 common candidate preparation',binding=state['binding_commit'],raw=state['raw_commit'])))
                OPS.write(registry_path,registry)
        quote=requote(request,config,state['implementation'],out/'OUTPUT_QUOTE.json')
        state.update(status='PRICES_READY'if quote.get('complete')else'HELD_REGISTERED_RULE_QUOTE_INCOMPLETE',quote=quote,
                     registry_file=str(registry_path),registered_rule=name)
        OPS.write(statefile,state);return state


def run(request,*,execute=False,push=True,phase='all',publish_binding=False):
    """Selected groups use existing acquisition/audit, then caller registration.

    Each step has its own stage namespace. No one-step model is registered as a
    legacy 26-axis rule. Repeated completed groups return their fixed receipts.
    """
    if publish_binding and (not execute or push or phase != 'compile'):
        raise ValueError('Isolated Pauli publication only supports distributed compile bindings')
    if not request.get('scoped_registration')or request.get('group_request_id'):
        return (_run_one(request,execute=execute,push=push)if phase=='all'
                else _run_one(request,execute=execute,push=push,phase=phase,publish_binding=publish_binding))
    plan=_run_one(request,execute=False,push=False)
    if plan.get('status')=='PRICES_READY'or not execute:return plan
    if plan.get('needs')or not plan.get('calibration_requests'):
        return dict(plan,status='WAITING_PRICE_EVIDENCE')
    packages=[];results=[]
    groups=plan['calibration_requests']
    separate_branches=publish_binding and len(groups)>1
    branches=[]
    if separate_branches:
        workspace=Path(runtime_request(request,('workspace',))['workspace']).resolve()
        attempt_branch=OPS.git(workspace,'branch','--show-current')
        public_base=os.environ['RPE_PAULI_BINDING_PUBLIC_BASE']
        if (not attempt_branch.startswith('codex/rpe-pauli-binding-')
                or OPS.git(workspace,'rev-parse','HEAD')!=public_base):
            raise ValueError('The Pauli preparation attempt does not own the fixed public base')
    try:
        for index,group in enumerate(groups):
            if separate_branches:
                branch=f'{attempt_branch}-group-{index+1}'
                OPS.git(workspace,'switch','-c',branch,public_base)
            child=dict(request,group_request_id=group['request_id'],
                directory=str(Path(request['directory'])/'step_groups'/group['request_id'].split(':')[-1][:24]),
                private_root=str(Path(request['private_root'])/'step_groups'/group['request_id'].split(':')[-1][:24]))
            result=(_run_one(child,execute=True,push=push)if phase=='all'
                    else _run_one(child,execute=True,push=push,phase=phase,publish_binding=publish_binding))
            results.append(result)
            if result.get('status')not in('SCOPED_STEP_MODEL_READY','PRICES_READY','COMPILE_READY_FOR_AUDIT'):
                return dict(plan,status=result['status'],group_results=results,evidence_packages=packages)
            if separate_branches:
                branches.append(branch)
            packages.extend(result.get('evidence_packages',[]))
    finally:
        if separate_branches:
            # Only each group's input and binding commits are public. Join
            # the group result histories on the private, collectable attempt.
            OPS.git(workspace,'switch',attempt_branch)
            if branches:
                OPS.git(workspace,'merge','--no-edit','--no-ff',*branches)
    return dict(schema='PR_PAULI_CANDIDATE_PRICES_V1',status='STEP_COMPILES_READY_FOR_AUDIT'if phase=='compile'else'STEP_MODELS_READY',
                configuration_id=request['configuration']['id'],group_results=results,evidence_packages=packages,
                legacy_registry_modified=False,automatic_retry=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--no-push',action='store_true')
    p.add_argument('--phase',choices=('all','compile','audit'),default='all')
    p.add_argument('--publish-binding',action='store_true')
    a=p.parse_args(argv)
    try:result=run(OPS.read(a.request),execute=a.execute,push=not a.no_push,phase=a.phase,publish_binding=a.publish_binding)
    except (ValueError,OSError,KeyError,subprocess.CalledProcessError)as exc:
        result=dict(schema='PR_PAULI_CANDIDATE_PRICES_V1',status='HELD_ORCHESTRATION_FAILURE',reason=repr(exc),automatic_retry=False)
    OPS.write(a.output,result);print(json.dumps(result,ensure_ascii=False,default=str))
    return 0 if result['status']in('PRICES_READY','CALIBRATION_PLANNED_NOT_EXECUTED','SCOPED_STEP_MODEL_READY','STEP_MODELS_READY','WAITING_PRICE_EVIDENCE','COMPILE_READY_FOR_AUDIT','STEP_COMPILES_READY_FOR_AUDIT')else 2


if __name__=='__main__':raise SystemExit(main())
