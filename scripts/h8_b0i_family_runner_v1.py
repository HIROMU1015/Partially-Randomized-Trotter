"""H8 B0+I: fixed twelve-case screen, then bounded per-case native/compile waves.

No real operation is permitted by the checked-in draft. A future explicit
grant binds this SOURCE, profile, inputs, resources and an unused output.
"""
from copy import deepcopy
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


def load(name, filename):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(filename))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


adapter_module=load('h8_b0i_adapter','h8_native_b0i_adapter_v1.py')
g=adapter_module.g
family=load('h8_b0i_accounting','h8_b0i_family_accounting_v1.py')
SCOPE=adapter_module.b.SCOPE
LIMITS=family.LIMITS
POLICY='ALL_SCREEN_THEN_FIRST_SELECTED_CORRECTNESS_THEN_CASE_WAVES_V1'


def validate_contract(grant):
    grant.require(SCOPE);c=grant.contract
    candidates=[dict(prefix=p,q=q,formula='2nd',T=.8) for p,q in family.REGISTRY]
    if (c.get('candidates')!=candidates or c.get('new_candidates')!=candidates or
            c.get('operation_limits')!=LIMITS or c.get('execution_order')!=POLICY or
            c.get('workers')!=2 or c.get('control')!='symmetric_directional' or
            c.get('precision_domain')!=dict(epsilon=list(family.EPS),factors=list(family.FACTORS)) or
            c.get('preparation_policy')!=adapter_module.b.RECIPE_POLICY or
            c.get('automatic_retry') is not False or c.get('next_stage_authorized') is not False or
            c.get('matched_B0')!=dict(approved=False,automatic_build=False)):
        raise g.Stop('H8_FIXED_SCIENTIFIC_OPERATION_SCOPE')
    if c.get('compiler_options')!=dict(basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17):
        raise g.Stop('H8_FIXED_COMPILER')
    if c.get('control_tolerance')!=1e-8 or c['budget'].get('wall_seconds') is not None:
        raise g.Stop('H8_EXPLICIT_TOLERANCE_WALL_POLICY')
    cfg=c.get('parallel_compile',{})
    if (cfg.get('enabled') is not True or cfg.get('max_workers')!=2 or
            cfg.get('internal_threads')!=1 or cfg.get('automatic_retry') is not False or
            cfg.get('backend')!='OWNED_FORK_AFTER_CASE_CORRECTNESS_V1'):
        raise g.Stop('H8_TWO_AXIS_FORK_SCOPE')
    if c.get('external_resource_monitor')!=load('h8_external_config','h8_b0i_family_external_monitor_v1.py').CONFIG:
        raise g.Stop('H8_INDEPENDENT_OBSERVER_REQUIRED')
    for p,q in family.REGISTRY:
        expected=g.d0.digest(adapter_module.b.recipe(c,p,q))
        if c['binding']['b0_fingerprints'].get(f'{p}/{q}')!=expected:
            raise g.Stop('H8_FIXED_PREPARATION_RECIPE')
    if not grant.synthetic:
        if (c['native_provider'].get('SOURCE')!='c31ef424175c02379eafebe8fc6c7548c1d9aabe' or
                c['binding'].get('source')!=c['native_provider']['SOURCE']):
            raise g.Stop('H8_PINNED_SCIENTIFIC_PROVIDER_REQUIRED')
        cpus=cfg.get('worker_cpus',[]);driver=cfg.get('driver_cpu')
        if (len(cpus)!=2 or len(set(cpus))!=2 or any(type(x) is not int or x<0 for x in cpus) or
                type(driver) is not int or driver<0 or driver in cpus or
                set(c['allowed_cpus'])!=set(cpus+[driver]) or os.sched_getaffinity(0)!={driver}):
            raise g.Stop('H8_FRESH_CPU_BINDING_REQUIRED')
        minimum=3*c['budget']['address_space_bytes']+c['launcher_budget']['address_space_bytes']+16*2**30
        if c['budget']['fresh_memory_available_bytes']<minimum:raise g.Stop('H8_ALL_ROLE_ADMISSION')
        if (not c.get('execution_plan_sealed') or not c.get('absolute_launch_command') or
                g.d0.digest(c['profile'])!=c['expected_profile_digest'] or
                not c['profile'].get('runtime_file_hashes') or not c['native_provider'].get('source_hashes') or
                os.environ.get('QISKIT_PARALLEL')!='FALSE'):
            raise g.Stop('H8_SEALED_PROFILE_PROVIDER_AND_COMMAND_REQUIRED')


class Calls:
    def __init__(self,limits,journal):self.limits=limits;self.used=dict.fromkeys(limits,0);self.journal=journal
    def take(self,name,count=1):
        if type(count) is not int or count<=0 or name not in self.limits or self.used[name]+count>self.limits[name]:
            raise g.Stop('H8_OPERATION_BUDGET:'+name)
        self.used[name]+=count
        self.journal(dict(operation=name,reserved_before_call=count,used=deepcopy(self.used),refund=0))


def private_temp(grant):
    grant.require(SCOPE);root=grant.output/'temp';root.mkdir(mode=0o700)
    for name,directory in [('TMPDIR',root),('MPLCONFIGDIR',root/'mpl'),
            ('XDG_CACHE_HOME',root/'cache'),('NUMBA_CACHE_DIR',root/'numba')]:
        if directory!=root:directory.mkdir(mode=0o700)
        os.environ[name]=str(directory)
    tempfile.tempdir=str(root);g.seed_font_cache(grant,root/'mpl')


def pipeline(grant,verified,mapping,monitor,artifacts,calls,factory,compile_batch):
    grant.require(SCOPE);api=None;cases=[];results=[]
    try:
        monitor.phase='A_H8_ALL_TWELVE_SCREEN'
        api=factory(grant,verified,mapping,calls,monitor.tick)
        for prefix,q in family.REGISTRY:
            calls.take('tail_scalar');calls.take('analytic_overlay')
            case=api.prepare(prefix,q);request=case['request'];signal=deepcopy(request['signal'])
            if (signal.get('reference') is None or signal['allowance'].get('kind')!='EMPIRICAL' or
                    signal['allowance'].get('certified') is not False):raise g.Stop('H8_EMPIRICAL_REFERENCE_REQUIRED')
            signal['cell']=dict(id=f'B0I_p{prefix}_q{q}_2nd',method='B0I',prefix=prefix,q=q,formula='2nd',T=.8)
            rows=[dict(epsilon=e,factor=f,accounting=g.d0.accounting(signal,e,f)) for e in family.EPS for f in family.FACTORS]
            selected=family.screen(rows,g);case.update(prefix=prefix,q=q,accounting=rows,selected=selected)
            artifacts.write(f'case_{prefix}_{q}_screen.json',dict(prefix=prefix,q=q,precision=rows,
                scalar=request['scalar_receipt'],signal=signal,preparation_binding=case.get('preparation_binding'),
                compile_required=selected,cost_status='UNMEASURED'),monitor.tick)
            cases.append(case)
        selected=[c for c in cases if c['selected']]
        artifacts.write('compile_plan_after_screen_v1.json',dict(selected_cases=[[c['prefix'],c['q']] for c in selected],
            new_compile_axes=2*len(selected),maximum_new_compile_axes=24,
            first_native_correctness_case=None if not selected else [selected[0]['prefix'],selected[0]['q']],
            first_case_rule='FIRST_SELECTED_IN_FIXED_PREFIX_Q_REGISTRY_ORDER',all_screen_before_any_native=True,
            execution_order=POLICY),monitor.tick)
        # One case in parent RAM. Every case must pass all probes before either
        # of its two worker compilations. A later failure retains earlier costs
        # but can never produce a COMPLETE comparison.
        for case in cases:
            axes=None
            if case['selected']:
                monitor.phase=f"B_H8_NATIVE_CORRECTNESS_{case['prefix']}_{case['q']}"
                if case.get('independent') is None:raise g.Stop('H8_SAVED_INDEPENDENT_REFERENCE_REQUIRED')
                calls.take('native_evolution_build');calls.take('measured_wrapper_build',2)
                built=api.build(case);validation=api.validate(case,built)
                if validation.get('status')!='PASS' or set(validation.get('fingerprints',{}))!={'cosine','sine'}:
                    raise g.Stop('H8_NATIVE_CORRECTNESS_BEFORE_EITHER_AXIS')
                artifacts.write(f"case_{case['prefix']}_{case['q']}_correctness.json",validation,monitor.tick)
                case.update(built=built,validation=validation);monitor.phase=f"C_H8_COMPILE_{case['prefix']}_{case['q']}"
                rows=compile_batch(grant,[case],api,monitor,artifacts,calls)
                expected={(case['prefix'],case['q'],axis) for axis in ('cosine','sine')}
                if set(rows)!=expected:raise g.Stop('H8_CASE_TWO_AXIS_COVERAGE')
                axes={}
                for axis in ('cosine','sine'):
                    row=rows[(case['prefix'],case['q'],axis)];fp=validation['fingerprints'][axis]
                    adapter_module.metrics_checked(row['metrics'])
                    if (row['fingerprint']!=fp or row.get('input_fingerprint_check')!=dict(before=fp,after=fp) or
                            row['compiler_hash']!=grant.contract['binding']['compiler']):raise g.Stop('H8_COMPILED_IDENTITY')
                    axes[axis]=row
                case.pop('built');del built,rows;gc.collect()
            results.append(dict(prefix=case['prefix'],q=case['q'],accounting=case['accounting'],axes=axes,
                acquisition='NEW_MEASURED' if axes is not None else 'SKIPPED_INELIGIBLE_UNMEASURED',
                origin_SOURCE=grant.contract['SOURCE']))
        n=len(selected)
        expected=dict(snapshot_load=1,prefix_preparation=3,tail_scalar=12,analytic_overlay=12,
            native_evolution_build=n,measured_wrapper_build=2*n,probe_wrapper_copy=2*n,control_probe=4*n,compile=2*n)
        if calls.used!=expected:raise g.Stop('H8_ADAPTIVE_CALL_COVERAGE')
        monitor.phase='D_H8_CONFIRMATION_UNION6_COMPARISON'
        accounting=family.cost_rows(results,g)
        comparison=family.compare(accounting,g.d0.strict_json(verified['d0_candidate_accounting.json']),grant.contract,g,monitor.tick)
        api.verify_bindings()
        artifacts.write('full_family_results_v1.json',dict(SOURCE=grant.contract['SOURCE'],cases=results,
            accounting=accounting,comparison=comparison,calls=calls.used,carry=grant.contract['carry'],
            compiler=grant.contract['binding']['compiler'],new_signal_exponential_actions=0,
            native_reference='STORED_INDEPENDENT_FORWARD_OVERLAP',numerical_certificate=False,
            next_stage_authorized=False),monitor.tick)
    finally:
        cases.clear()
        if api is not None:api.close()


def run(contract,authorization,*,contract_sha,authorization_sha,output,root,mapping_loader,
        synthetic=False,monitor_factory=g.Monitor,factory=adapter_module.NativeAdapter,compile_batch=None):
    grant=g.authorization_gate(contract,authorization,contract_sha=contract_sha,authorization_sha=authorization_sha,
        output=output,scope=SCOPE,synthetic=synthetic)
    validate_contract(grant)
    artifacts=g.Artifacts(grant);monitor=None;first=None;status=2;sequence=[0]
    def journal(value):
        sequence[0]+=1;artifacts.write(f'operation_{sequence[0]:03d}.json',value,monitor.tick if monitor else lambda:None)
    calls=Calls(contract['operation_limits'],journal)
    try:
        if not synthetic:
            external=load('h8_external_monitor','h8_b0i_family_external_monitor_v1.py')
            monitor_factory=external.monitor_class(g)
        monitor=monitor_factory(grant)
        if not synthetic:monitor.install();private_temp(grant)
        g.runtime_gate(grant,root);mapping=mapping_loader();verified=g.load_inputs(grant,mapping)
        if compile_batch is None:
            if synthetic:raise g.Stop('H8_ARTIFICIAL_REQUIRES_MOCK_BACKEND')
            parallel=load('h8_parallel','h8_b0i_family_parallel_compile_v1.py')
            compile_batch=lambda *args:parallel.compile_batch(*args,g=g,metrics_check=adapter_module.metrics_checked)
        pipeline(grant,verified,mapping,monitor,artifacts,calls,factory,compile_batch)
        monitor.phase='D_H8_FINAL_BINDING';g.load_inputs(grant,mapping);g.runtime_gate(grant,root);monitor.tick(force=True);status=0
    except Exception as error:
        first=getattr(monitor,'first_stop',None) or dict(reason=type(error).__name__+':'+str(error)[:1000],phase=getattr(monitor,'phase','PRELAUNCH'))
    finally:
        if monitor is not None:monitor.close()
        try:artifacts.write('terminal_v1.json',dict(status='COMPLETE_MANDATORY_STOP' if status==0 else 'FAIL_CLOSED_STOP',
            first_stop=first,intended_exit_code=status,native_exit_code=None,pid=g.identity(),
            calls_reserved_before_operations=calls.used,child_CPU_seconds=getattr(monitor,'child_cpu_seconds',0.),
            cleanup_signal_deferral=getattr(monitor,'cleanup_signal_deferral',None),
            output_file_receipts=deepcopy(artifacts.receipts),
            automatic_retry=False,next_stage_authorized=False),terminal=True)
        finally:artifacts.close()
    return status


def record_wait(contract,authorization,*,contract_sha,authorization_sha,output,status):
    grant=g.authorization_gate(contract,authorization,contract_sha=contract_sha,authorization_sha=authorization_sha,
        output=output,scope=SCOPE,consumed=True)
    ticket=g.d0.strict_json(g.read_bytes(g.marker_path(grant.output)))
    if (ticket['contract_sha256']!=contract_sha or ticket['authorization_sha256']!=authorization_sha or
            ticket['SOURCE']!=contract['SOURCE'] or ticket['output']!=str(output) or
            ticket['parent_identity']!=g.identity(os.getppid())):raise g.Stop('H8_NATIVE_WAIT_BINDING')
    old=ticket['identity'];current=g.identity(old['pid'])
    if current is not None and all(current[k]==old[k] for k in ('pid','start_ticks','boot_id','uid')):
        raise g.Stop('H8_DRIVER_STILL_PRESENT')
    if type(status) is not int or not 0<=status<=255:raise g.Stop('H8_NATIVE_WAIT_STATUS')
    value=dict(shell_wait_status=status,driver_identity=old,original_identity_absent=True,
        terminal_available=(grant.output/'terminal_v1.json').exists(),same_parent_wait=True,
        exact_exit_timestamp=None,automatic_retry=False)
    with (grant.output/'native_exit_v1.json').open('x') as f:json.dump(value,f);f.write('\n');f.flush();os.fsync(f.fileno())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('contract','authorization','inputs','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--record-wait-status',type=int);args=parser.parse_args()
    raw=g.read_bytes(args.contract);auth=g.read_bytes(args.authorization);c=g.d0.strict_json(raw);a=g.d0.strict_json(auth)
    if c.get('synthetic_only'):raise g.Stop('H8_CLI_REJECTS_ARTIFICIAL')
    kw=dict(contract_sha=g.sha(raw),authorization_sha=g.sha(auth),output=Path(args.output))
    if args.record_wait_status is not None:record_wait(c,a,status=args.record_wait_status,**kw);return 0
    return run(c,a,root=Path(__file__).resolve().parents[1],mapping_loader=lambda:g.d0.strict_json(g.read_bytes(args.inputs)),**kw)


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as error:
        print('H8_PRELAUNCH_STOP:'+type(error).__name__+':'+str(error)[:1000],file=sys.stderr);sys.exit(2)
