"""H6 B0+I 3-case/6-axis bounded one-shot; drafts cannot load science."""
from copy import deepcopy
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

adapter_module = load('h6_pilot_adapter', 'h6_b0i_pilot_adapter_v1.py')
g = adapter_module.g
SCOPE = adapter_module.b.SCOPE
PILOT = ((5,1), (10,2), (15,8))
LIMITS = dict(snapshot_load=1, analytic_overlay=3, native_evolution_build=3,
    measured_wrapper_build=6, probe_wrapper_copy=6, control_probe=12, compile=6)


def validate_contract(grant):
    grant.require(SCOPE); c = grant.contract
    parallel=c.get('parallel_compile',{}).get('enabled') is True
    expected = [dict(prefix=p, q=q, formula='2nd', T=.8) for p,q in PILOT]
    if (c.get('candidates') != expected or c.get('control') != 'symmetric_directional' or
            c.get('operation_limits') != LIMITS or c.get('matched_B0', {}).get('automatic_build') is not False or
            c.get('matched_B0', {}).get('approved') is not False or
            c.get('precision_domain') != dict(epsilon=[.05,.01,.005,.001,.0005], factors=[1.,10.]) or
            (c.get('workers') != 1 and not parallel) or c.get('automatic_retry') is not False):
        raise g.Stop('PILOT_REGISTRY_OPERATION_SCOPE')
    if parallel:
        cfg=c['parallel_compile'];workers=c['workers']
        if (type(workers) is not int or not 2<=workers<=6 or cfg.get('max_workers')!=workers or
                cfg.get('backend')!='OWNED_FORK_AFTER_ALL_AB_V1' or cfg.get('internal_threads')!=1 or
                cfg.get('automatic_retry') is not False or cfg.get('parent_death_signal')!='SIGKILL'):
            raise g.Stop('FIXED_PARALLEL_COMPILE_SCOPE')
        if not grant.synthetic:
            cpus=cfg['worker_cpus'];driver=cfg['driver_cpu']
            if (len(cpus)!=workers or any(type(cpu) is not int or cpu<0 for cpu in cpus) or
                    len(set(cpus))!=workers or type(driver) is not int or driver<0 or driver in cpus or
                    set(c['allowed_cpus'])!=set(cpus+[driver]) or os.sched_getaffinity(0)!={driver}):
                raise g.Stop('FRESH_PARALLEL_CPU_BINDING_REQUIRED')
            minimum=(workers+1)*c['budget']['address_space_bytes']+c['launcher_budget']['address_space_bytes']+16*2**30
            if c['budget']['fresh_memory_available_bytes']<minimum:
                raise g.Stop('PARALLEL_ADMISSION_MUST_INCLUDE_ALL_ROLES')
    if c.get('compiler_options') != dict(basis_gates=['rz','sx','x','cx'], optimization_level=1, seed_transpiler=17):
        raise g.Stop('FIXED_COMPILER_OPTIONS')
    if c.get('control_tolerance') != 1e-8: raise g.Stop('FIXED_CONTROL_TOLERANCE')
    if not grant.synthetic:
        if c.get('execution_plan_sealed') is not True or not c.get('absolute_launch_command'):
            raise g.Stop('SEALED_FRESH_LAUNCH_BINDING_REQUIRED')
        if os.environ.get('QISKIT_PARALLEL') != 'FALSE': raise g.Stop('QISKIT_PARALLEL_DISABLED_REQUIRED')
        evidence = c['scientific_review']; root = g.home_path(c['runtime_root'])
        if (evidence['commit'] != '3d910ee1de4597b8895e8506500553dbad5a5085' or
                evidence['path'] != 'docs/research/track_a_h8_d0_integrated_scientific_review_20261015.md'):
            raise g.Stop('FIXED_SCIENTIFIC_REVIEW_REQUIRED')
        raw = g.read_bytes(root/evidence['path'])
        import subprocess
        fixed = subprocess.check_output(['git','-C',str(root),'show',evidence['commit']+':'+evidence['path']])
        if raw != fixed or g.sha(raw) != evidence['sha256']:
            raise g.Stop('SCIENTIFIC_REVIEW_BINDING')
        if c['native_provider']['SOURCE'] != '87b6b9f46b7e8ba9abe132ee21478c114b6f46f7':
            raise g.Stop('SCIENTIFIC_PROVIDER_SOURCE')
        if not c['native_provider'].get('source_hashes') or not c['profile']['runtime_file_hashes']:
            raise g.Stop('MATERIALIZED_PROVIDER_PROFILE_REQUIRED')
        adapter_module.identity_binding.validate_evidence(c,g,root)


class Calls:
    def __init__(self, limits, journal):
        self.limits = limits; self.used = dict.fromkeys(limits, 0); self.journal = journal
    def take(self, name, count=1):
        if name not in self.limits or type(count) is not int or count <= 0 or self.used[name]+count > self.limits[name]:
            raise g.Stop('OPERATION_BUDGET:'+name)
        self.used[name] += count
        self.journal(dict(operation=name, reserved_before_call=count, used=deepcopy(self.used), refund=0))


def prepare_private_temp(grant):
    """Run only after authorization/exclusive output; prevent tempfile fallback."""
    grant.require(SCOPE)
    temporary=grant.output/'temp';temporary.mkdir(mode=0o700)
    paths=dict(TMPDIR=temporary,MPLCONFIGDIR=temporary/'mpl',XDG_CACHE_HOME=temporary/'cache')
    for key,path in paths.items():
        if path!=temporary:path.mkdir(mode=0o700)
        os.environ[key]=str(path)
    tempfile.tempdir=str(temporary)


def pipeline(grant, verified, mapping, monitor, artifacts, calls, factory, compile_batch=None):
    monitor.phase = 'A_INPUT_SCALAR_SIGNAL'
    api = factory(grant, verified, mapping, calls, monitor.tick)
    cases = []
    try:
        for prefix,q in PILOT:
            calls.take('analytic_overlay'); case = api.prepare(prefix,q)
            request = case['request']; signal = deepcopy(request['signal'])
            signal['cell'] = dict(id=f'B0I_p{prefix}_q{q}_2nd',method='B0I',prefix=prefix,q=q,
                formula='2nd',T=.8)
            if signal['reference'] is None or signal['allowance']['certified'] is not False:
                raise g.Stop('EMPIRICAL_SIGNAL_REFERENCE_REQUIRED')
            rows = [dict(epsilon=e, factor=f, accounting=g.d0.accounting(signal,e,f))
                for e in grant.contract['precision_domain']['epsilon'] for f in grant.contract['precision_domain']['factors']]
            case.update(prefix=prefix,q=q,accounting=rows)
            artifacts.write(f'case_{prefix}_{q}_scalar_signal.json',dict(prefix=prefix,q=q,
                scalar=request['scalar_receipt'],signal=signal,blueprint=request['blueprint'],
                preparation_binding=case.get('preparation_binding'),
                independent_saved_reference=case['independent'],precision=rows),monitor.tick)
            cases.append(case)
        monitor.phase = 'B_NATIVE_CORRECTNESS'
        for case in cases:
            calls.take('native_evolution_build'); calls.take('measured_wrapper_build',2)
            built = api.build(case); validation = api.validate(case,built)
            if validation.get('status') != 'PASS' or set(validation.get('fingerprints',{})) != {'cosine','sine'}:
                raise g.Stop('ALL_AXES_NATIVE_CORRECTNESS_REQUIRED')
            case.update(built=built, validation=validation)
            artifacts.write(f"case_{case['prefix']}_{case['q']}_correctness.json",validation,monitor.tick)
        # Every case passed A/B before the first compile reservation.
        monitor.phase = 'C_COMPILE_ACCOUNTING'
        compiled_rows=compile_batch(grant,cases,api,monitor,artifacts,calls) if compile_batch else None
        results = []
        for case in cases:
            axes = {}
            for axis in ('cosine','sine'):
                monitor.tick(force=True)
                wrapper = case['built']['wrappers'][axis]; fingerprint = case['validation']['fingerprints'][axis]
                if api.fingerprint(wrapper) != fingerprint: raise g.Stop('WRAPPER_CHANGED_BEFORE_COMPILE')
                if compiled_rows is None:
                    calls.take('compile'); row = api.compile(wrapper,fingerprint)
                else:row=compiled_rows[(case['prefix'],case['q'],axis)]
                adapter_module.metrics_checked(row['metrics'])
                if row['fingerprint'] != fingerprint or row['compiler_hash'] != grant.contract['binding']['compiler']:
                    raise g.Stop('COMPILED_IDENTITY_MISMATCH')
                axes[axis] = row
                if compiled_rows is None:
                    artifacts.write(f"case_{case['prefix']}_{case['q']}_{axis}_cost.json",row,monitor.tick)
            costs = []
            for precision in case['accounting']:
                account = precision['accounting']; eligible = account['cost_acquisition_eligible']
                values = {m:sum(account['axes'][channel]['shots']*axes[axis]['metrics'][m]
                    for axis,channel in g.d0.AXIS_CHANNEL.items()) for m in adapter_module.METRICS} if eligible else None
                costs.append(dict(epsilon=precision['epsilon'],factor=precision['factor'],
                    status='DERIVED_SAVED_SCALAR' if eligible else 'INELIGIBLE',metrics=values))
            results.append(dict(prefix=case['prefix'],q=case['q'],axes=axes,accounting=costs,
                legacy_B0_reference=api.legacy_b0(case['prefix'],case['q']),
                matched_B0=dict(status='NOT_ACQUIRED',deltas=None,automatic_build=False),
                population_winner_certified=False,numerical_certificate=False))
            case.pop('built'); monitor.tick(force=True)
        if calls.used != LIMITS: raise g.Stop('EXECUTED_OPERATION_COVERAGE')
        monitor.phase = 'D_FINAL_PROVIDER_BINDING'
        api.verify_bindings()
        artifacts.write('pilot_results_v1.json',dict(cases=results,calls=calls.used,
            SOURCE=grant.contract['SOURCE'],compiler=grant.contract['binding']['compiler'],
            old_science_binding=grant.contract['binding'],carry=grant.contract['carry'],
            new_signal_exponential_actions=0,matched_B0_added=False,next_stage_authorized=False),monitor.tick)
    finally:
        cases.clear(); api.close()


def run(contract, authorization, *, contract_sha, authorization_sha, output, root,
        mapping_loader, synthetic=False, monitor_factory=g.Monitor, factory=adapter_module.NativeAdapter,
        compile_batch=None):
    grant = g.authorization_gate(contract,authorization,contract_sha=contract_sha,
        authorization_sha=authorization_sha,output=output,scope=SCOPE,synthetic=synthetic)
    validate_contract(grant)
    artifacts = g.Artifacts(grant); monitor = None; first_stop = None; result = 2; journal_count = [0]; journal_bytes = [0]
    preparation_diagnostic_write_error=None
    def journal(value):
        journal_bytes[0] += len(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode())+1
        if journal_bytes[0] > contract.get('journal_cap_bytes',64*2**20): raise g.Stop('OPERATION_JOURNAL_CAP')
        journal_count[0] += 1
        artifacts.write(f'operation_{journal_count[0]:03d}.json',value,monitor.tick if monitor else lambda:None)
    calls = Calls(contract['operation_limits'], journal)
    try:
        monitor = monitor_factory(grant)
        if not synthetic:
            monitor.install();prepare_private_temp(grant)
        g.runtime_gate(grant,root); mapping = mapping_loader(); verified = g.load_inputs(grant,mapping)
        if contract.get('parallel_compile',{}).get('enabled') is True and compile_batch is None:
            if synthetic:raise g.Stop('ARTIFICIAL_PARALLEL_REQUIRES_MOCK_BACKEND')
            parallel=load('h6_owned_parallel_compile','h6_b0i_parallel_compile_v1.py')
            compile_batch=lambda *args:parallel.compile_batch(*args,g=g,metrics_check=adapter_module.metrics_checked)
        pipeline(grant,verified,mapping,monitor,artifacts,calls,factory,compile_batch)
        monitor.phase = 'D_FINAL_BINDING_CLEANUP'
        g.load_inputs(grant,mapping); g.runtime_gate(grant,root); monitor.tick(force=True); result = 0
    except Exception as error:
        first_stop = dict(reason=type(error).__name__+':'+str(error)[:1000],phase=monitor.phase if monitor else 'PRELAUNCH')
        if monitor is not None and monitor.first_stop is not None: first_stop = monitor.first_stop
        diagnostic=getattr(error,'preparation_diagnostic',None)
        if diagnostic is not None:
            try:artifacts.write('preparation_mismatch_v1.json',diagnostic,terminal=True)
            except Exception as evidence_error:
                preparation_diagnostic_write_error=type(evidence_error).__name__+':'+str(evidence_error)[:300]
    finally:
        if monitor is not None: monitor.close()
        try:
            artifacts.write('terminal_v1.json',dict(status='COMPLETE_MANDATORY_STOP' if result==0 else 'FAIL_CLOSED_STOP',
                first_stop=first_stop,intended_exit_code=result,native_exit_code=None,pid=g.identity(),
                scientific_child_processes_max=contract['workers'] if contract.get('parallel_compile',{}).get('enabled') else 0,
                worker_model='OWNED_FORK_COMPILE_AXES' if contract.get('parallel_compile',{}).get('enabled') else 'ONE_FOREGROUND_PROCESS_NO_POOL',
                calls_reserved_before_operations=calls.used,automatic_retry=False,next_stage_authorized=False,
                preparation_diagnostic_write_error=preparation_diagnostic_write_error,
                cleanup=dict(timer_removed=True,artifact_FDs_closed_on_return=True,
                    child_wait_reap='SEE_PARALLEL_CLEANUP_RECEIPT' if contract.get('parallel_compile',{}).get('enabled') else 'NO_SCIENTIFIC_CHILDREN'),
                resource_observations=getattr(monitor,'history',[]),output_file_receipts=deepcopy(artifacts.receipts)),terminal=True)
        finally: artifacts.close()
    return result


def record_wait(contract, authorization, *, contract_sha, authorization_sha, output, status):
    grant = g.authorization_gate(contract,authorization,contract_sha=contract_sha,
        authorization_sha=authorization_sha,output=output,scope=SCOPE,consumed=True)
    ticket = g.d0.strict_json(g.read_bytes(g.marker_path(grant.output)))
    if (ticket['contract_sha256']!=contract_sha or ticket['authorization_sha256']!=authorization_sha or
            ticket['SOURCE']!=contract['SOURCE'] or ticket['output']!=str(grant.output) or
            ticket['parent_identity']!=g.identity(os.getppid())): raise g.Stop('WAIT_PARENT_RUN_BINDING')
    old = ticket['identity']; current = g.identity(old['pid'])
    if current is not None and all(current[k]==old[k] for k in ('pid','start_ticks','boot_id','uid')):
        raise g.Stop('DRIVER_IDENTITY_STILL_PRESENT')
    if type(status) is not int or not 0<=status<=255: raise g.Stop('NATIVE_WAIT_STATUS_REQUIRED')
    terminal = grant.output/'terminal_v1.json'
    value = dict(shell_wait_status=status,driver_identity=old,original_identity_absent=True,
        terminal_available=terminal.exists(),same_parent_wait=True,exact_exit_timestamp=None,automatic_retry=False)
    fd = os.open(grant.output/'native_exit_v1.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try: os.write(fd,json.dumps(value).encode()+b'\n'); os.fsync(fd)
    finally: os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('contract','authorization','inputs','output'): parser.add_argument('--'+key,required=True)
    parser.add_argument('--record-wait-status',type=int); args = parser.parse_args()
    raw = g.read_bytes(args.contract); auth = g.read_bytes(args.authorization)
    contract = g.d0.strict_json(raw); authorization = g.d0.strict_json(auth)
    if contract.get('synthetic_only'): raise g.Stop('PRODUCTION_CLI_REJECTS_ARTIFICIAL')
    kw = dict(contract_sha=g.sha(raw),authorization_sha=g.sha(auth),output=Path(args.output))
    if args.record_wait_status is not None:
        record_wait(contract,authorization,status=args.record_wait_status,**kw); return 0
    return run(contract,authorization,root=Path(__file__).resolve().parents[1],
        mapping_loader=lambda:g.d0.strict_json(g.read_bytes(args.inputs)),**kw)

if __name__=='__main__':
    try: sys.exit(main())
    except Exception as error:
        print('H6_PRELAUNCH_STOP:'+type(error).__name__+':'+str(error)[:1000],file=sys.stderr);sys.exit(2)
