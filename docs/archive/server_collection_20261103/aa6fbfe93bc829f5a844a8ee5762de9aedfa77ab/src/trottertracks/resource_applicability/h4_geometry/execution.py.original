"""Two future one-shot stages. No execution authorization is emitted here."""
import io
import json
import os
from pathlib import Path
import time
import threading
from .identity import require, Stop, sha, fingerprint
from .gates import authorize, checkout_gate, DISTANCES, OUTPUT, Permit
from .resources import OutputBudget, WallBudget, Monitor, admission, observe_memory, limit_owned_address_space
from .ledger import Ledger, wire, json_bytes


def _generate_worker(permit,distance):
    from .inputs import generate_input,freeze_input
    arrays=generate_input(permit,distance,lambda:None)
    return freeze_input(permit,distance,arrays)


def _compile_worker(circuit,options):
    from qiskit import transpile
    from .circuits import metrics
    return metrics(transpile(circuit,**options))


class OwnedRun:
    def __init__(self,permit,authorization,*,prior_wall=0,handoff=False):
        observation=observe_memory()
        self.workers=admission(observation['available'],observation['observed_at'],permit.plan['requested_workers'],
                               authorization['allowed_cpus'],observation['process_cpus'])
        # No affinity edits. A narrower CPU permission requires a separately scoped launch context.
        require(observation['process_cpus'] <= set(authorization['allowed_cpus']), 'process can use unpermitted CPU')
        self.monitor=Monitor(self.workers,observation)
        self.wall=WallBudget(prior_wall)
        limit_owned_address_space()
        self.budget=OutputBudget(OUTPUT,handoff=handoff)
        self.finished=threading.Event();self.failure=None
        self.stage=permit.stage
        self.wall_index=0
        self.thread=threading.Thread(target=self._watch,name='owned-h4-monitor',daemon=True)
        self.thread.start()
        try:
            from .workers import OwnedPool
            self.pool=OwnedPool(self.workers,permit,self.monitor,self.budget)
        except BaseException:
            self.finished.set();self.thread.join(timeout=2)
            self.monitor.stop_children()
            self.budget.close();raise

    def _watch(self):
        try:
            while not self.finished.wait(1):
                self.monitor.poll()
                used=self.wall.consumed()
                self.budget.write(self.stage+'-wall-%06d.json'%self.wall_index,json_bytes({'consumed_seconds':used}))
                self.wall_index+=1
        except BaseException as exc:
            self.failure=exc
            self.monitor.stop_children()
            # The driver is owned by this run. Interrupt Python, not another job.
            import _thread
            _thread.interrupt_main()

    def pulse(self):
        require(self.failure is None,'owned monitoring failed STOP')
        require(getattr(getattr(self,'pool',None),'failure',None) is None,'owned pool failed STOP')
        self.wall.consumed()

    def wait(self,future):
        from concurrent.futures import TimeoutError
        while True:
            self.pulse()
            try:
                return future.result(timeout=1)
            except TimeoutError:
                pass

    def wait_any(self,futures):
        from concurrent.futures import wait,FIRST_COMPLETED
        while True:
            self.pulse()
            done,_=wait(futures,timeout=1,return_when=FIRST_COMPLETED)
            if done:
                self.pulse()
                return done

    def abort(self):
        self.finished.set()
        self.monitor.stop_children()

    def close(self):
        self.finished.set();self.thread.join(timeout=2)
        self.monitor.stop_children()
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.budget.close()
        require(self.failure is None,'monitor STOP; no retry/resume')


def generation_stage(permit,authorization,options):
    require(permit.stage=='input_generation','generation stage')
    run=OwnedRun(permit,authorization)
    records={};identities={}
    try:
        futures={d:run.pool.submit(_generate_worker,permit,d) for d in DISTANCES}
        for d in DISTANCES:
            data,record,identity=run.wait(futures[d]);run.pulse()
            run.budget.write(record['file'],data)
            records[d],identities[d]=record,identity
        freeze={'stage':'INPUTS_FROZEN_STOP','run_id':permit.plan['run_id'],
                'source_commit':permit.plan['source_commit'],'generation_review_digest':permit.review_digest,
                'inputs':records,'identities':identities,'consumed_seconds':run.wall.consumed(),
                'research_decision':None,'next_stage_authorized':False,'mandatory_stop':True}
        run.budget.write('generation-freeze.json',json_bytes(freeze))
        return freeze  # no signal continuation and no plan/authorization emitted
    finally:
        run.close()


def input_boundary(permit):
    require(permit.stage=='signal_compile','separate signal permit')
    freeze_path=Path(OUTPUT)/'generation-freeze.json'
    require(not freeze_path.is_symlink(),'freeze symlink')
    freeze=json.loads(freeze_path.read_bytes())
    require(fingerprint('h4-generation-freeze-v1',freeze)==permit.plan['generation_freeze_digest'],'generation freeze binding')
    require(freeze['stage']=='INPUTS_FROZEN_STOP' and freeze['inputs']==permit.plan['inputs'] and
            freeze['source_commit']==permit.plan['source_commit'] and freeze['mandatory_stop'] is True,'frozen inputs')
    return freeze


def load_new_input(permit,distance,freeze):
    require(permit.stage=='signal_compile','signal input gate')
    import numpy as np
    from .inputs import array_identity
    entry=permit.plan['inputs'][distance]
    require(entry['file']=='input-'+distance+'.npz','input path scope')
    path=Path(OUTPUT)/entry['file']
    require(not path.is_symlink(),'input symlink')
    data=path.read_bytes();require(sha(data)==entry['bytes_sha256'],'frozen input bytes')
    with np.load(io.BytesIO(data),allow_pickle=False) as archive:
        arrays={k:archive[k] for k in archive.files}
    expected=freeze['identities'][distance]
    require({k:array_identity(v) for k,v in sorted(arrays.items())}==expected['arrays'],'complete input array identities')
    require(fingerprint('h4-input-freeze-v1',expected)==entry['input'],'input fingerprint')
    return arrays


def candidate_wrapper_jobs(run,identity,seeds,preparation,template):
    """Generate one evolution and its two axes lazily in trajectory order."""
    from .signal import sample_events
    from .circuits import build_evolution,wrapper,numerical_fingerprint
    for index,seed in enumerate(seeds):
        run.pulse()
        events=sample_events(preparation['components'],template,seed)[0] if seed is not None else [[] for _ in range(template['q'])]
        evolution=build_evolution(preparation,template,events)
        for axis in ('cosine','sine'):
            run.pulse();circuit=wrapper(evolution,axis)
            numerical=numerical_fingerprint(circuit,axis)
            yield wire(identity,axis,seed,index if seed is not None else None,numerical),circuit


def signal_stage(permit,authorization,options):
    require(permit.stage=='signal_compile','separate signal stage')
    # The freeze and file access occur only after authorize + checkout_gate.
    freeze=input_boundary(permit)
    run=OwnedRun(permit,authorization,prior_wall=freeze['consumed_seconds'],handoff=True)
    ledger=None
    from .signal import prepare,corrected_signal,candidate_identity,trajectory_seeds,display_map
    from .parallel import compile_wrappers
    from .inputs import validate_frozen_state
    signals=[];reuse={}
    try:
        ledger=Ledger(run.budget)
        for distance in DISTANCES:
            run.pulse();arrays=load_new_input(permit,distance,freeze)
            validate_frozen_state(arrays)
            import numpy as np
            inp={'geometry':distance,**{k:permit.plan['inputs'][distance][k] for k in ('input','H','DF','state')}}
            for template in permit.plan['templates']:
                run.pulse()
                identity=candidate_identity(inp,template,permit.plan['source_commit'],
                    permit.plan['compiler_fingerprint'],permit.plan['environment_fingerprint'])
                preparation,det,tail=prepare(arrays,template)
                signal_record=corrected_signal(det,tail,preparation['constant'],arrays['qiskit_state'],template)
                target=complex(np.exp(-1j*float(arrays['energy'])*template['T']))
                seeds=trajectory_seeds(identity) if template['method'] in ('B2','B3') else [None]
                records=compile_wrappers(run,ledger,candidate_wrapper_jobs(run,identity,seeds,preparation,template),
                                         _compile_worker,options,reuse,expected_count=2*len(seeds))
                all_metrics=[[records[2*i]['metrics'],records[2*i+1]['metrics']] for i in range(len(seeds))]
                costs=[[m['rz_count'] for m in pair] for pair in all_metrics]
                saved={'normalization':signal_record['normalization'],
                    'bias':{'cosine':abs(signal_record['corrected'].real-target.real),'sine':abs(signal_record['corrected'].imag-target.imag)},
                    'paired_costs':costs}
                result={'geometry':distance,'template_id':template['template_id'],'identity':identity,
                        'candidate_fingerprint':fingerprint('h4-candidate-v1',identity),
                        'signal':{'corrected':[signal_record['corrected'].real,signal_record['corrected'].imag],
                                  'raw':[signal_record['raw'].real,signal_record['raw'].imag],'normalization':saved['normalization']},
                        'saved':saved,'paired_metrics':all_metrics,'display':display_map(saved),
                        'research_decision':None,'next_stage_authorized':False}
                signals.append(result)
                run.budget.write('signal-'+distance+'-'+template['template_id']+'.json',json_bytes(result))
        accounting=ledger.audit()
        require(len(signals)==1308 and accounting['logical_wrappers']==74784 and accounting['actual_invocations']<=74784,'fixed campaign accounting')
        final={'status':'MAP_COMPLETE_STOP','signal_records':len(signals),**accounting,
               'ledger_head_digest':ledger.chain,'consumed_seconds':run.wall.consumed(),
               'research_decision':None,'next_stage_authorized':False,'mandatory_stop':True}
        run.budget.write('map-complete.json',json_bytes(final))
        return final
    finally:
        if ledger is not None:
            ledger.close()
        run.close()


def launch(stage,plan,authorization,review,*,explicit_launch=False):
    permit=authorize(stage,plan,authorization,review,explicit_launch=explicit_launch)
    contract,options=checkout_gate(permit)
    if stage=='input_generation':
        return generation_stage(permit,authorization,options)
    return signal_stage(permit,authorization,options)
