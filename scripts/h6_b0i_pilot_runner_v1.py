"""H6 B0+I 3-case/6-axis bounded one-shot; drafts cannot load science."""
from copy import deepcopy
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

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
    expected = [dict(prefix=p, q=q, formula='2nd', T=.8) for p,q in PILOT]
    if (c.get('candidates') != expected or c.get('control') != 'symmetric_directional' or
            c.get('operation_limits') != LIMITS or c.get('matched_B0', {}).get('automatic_build') is not False or
            c.get('matched_B0', {}).get('approved') is not False or
            c.get('precision_domain') != dict(epsilon=[.05,.01,.005,.001,.0005], factors=[1.,10.]) or
            c.get('workers') != 1 or c.get('automatic_retry') is not False):
        raise g.Stop('PILOT_REGISTRY_OPERATION_SCOPE')
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


class Calls:
    def __init__(self, limits, journal):
        self.limits = limits; self.used = dict.fromkeys(limits, 0); self.journal = journal
    def take(self, name, count=1):
        if name not in self.limits or type(count) is not int or count <= 0 or self.used[name]+count > self.limits[name]:
            raise g.Stop('OPERATION_BUDGET:'+name)
        self.used[name] += count
        self.journal(dict(operation=name, reserved_before_call=count, used=deepcopy(self.used), refund=0))


def pipeline(grant, verified, mapping, monitor, artifacts, calls, factory):
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
        results = []
        for case in cases:
            axes = {}
            for axis in ('cosine','sine'):
                monitor.tick(force=True)
                wrapper = case['built']['wrappers'][axis]; fingerprint = case['validation']['fingerprints'][axis]
                if api.fingerprint(wrapper) != fingerprint: raise g.Stop('WRAPPER_CHANGED_BEFORE_COMPILE')
                calls.take('compile'); row = api.compile(wrapper,fingerprint)
                adapter_module.metrics_checked(row['metrics'])
                if row['fingerprint'] != fingerprint or row['compiler_hash'] != grant.contract['binding']['compiler']:
                    raise g.Stop('COMPILED_IDENTITY_MISMATCH')
                axes[axis] = row
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
        mapping_loader, synthetic=False, monitor_factory=g.Monitor, factory=adapter_module.NativeAdapter):
    grant = g.authorization_gate(contract,authorization,contract_sha=contract_sha,
        authorization_sha=authorization_sha,output=output,scope=SCOPE,synthetic=synthetic)
    validate_contract(grant)
    artifacts = g.Artifacts(grant); monitor = None; first_stop = None; result = 2; journal_count = [0]; journal_bytes = [0]
    def journal(value):
        journal_bytes[0] += len(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode())+1
        if journal_bytes[0] > contract.get('journal_cap_bytes',64*2**20): raise g.Stop('OPERATION_JOURNAL_CAP')
        journal_count[0] += 1
        artifacts.write(f'operation_{journal_count[0]:03d}.json',value,monitor.tick if monitor else lambda:None)
    calls = Calls(contract['operation_limits'], journal)
    try:
        monitor = monitor_factory(grant)
        if not synthetic: monitor.install()
        g.runtime_gate(grant,root); mapping = mapping_loader(); verified = g.load_inputs(grant,mapping)
        pipeline(grant,verified,mapping,monitor,artifacts,calls,factory)
        monitor.phase = 'D_FINAL_BINDING_CLEANUP'
        g.load_inputs(grant,mapping); g.runtime_gate(grant,root); monitor.tick(force=True); result = 0
    except Exception as error:
        first_stop = dict(reason=type(error).__name__+':'+str(error)[:1000],phase=monitor.phase if monitor else 'PRELAUNCH')
        if monitor is not None and monitor.first_stop is not None: first_stop = monitor.first_stop
    finally:
        if monitor is not None: monitor.close()
        try:
            artifacts.write('terminal_v1.json',dict(status='COMPLETE_MANDATORY_STOP' if result==0 else 'FAIL_CLOSED_STOP',
                first_stop=first_stop,intended_exit_code=result,native_exit_code=None,pid=g.identity(),
                scientific_child_processes=0,worker_model='ONE_FOREGROUND_PROCESS_NO_POOL',
                calls_reserved_before_operations=calls.used,automatic_retry=False,next_stage_authorized=False,
                cleanup=dict(timer_removed=True,artifact_FDs_closed_on_return=True,child_wait_reap='NO_SCIENTIFIC_CHILDREN'),
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
