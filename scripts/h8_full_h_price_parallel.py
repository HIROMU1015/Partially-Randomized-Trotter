"""H8 same-candidate/same-seed technical transfer, with untouched legacy SOURCE."""
import hashlib
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import subprocess
import h8_full_h_pure_rte as s
import h8_full_h_price as legacy
import h8_full_h_parallel_aer as fast

PARENT=s.PUBLIC
PUBLIC=PARENT/'parallel_v1'
PRIVATE=PARENT/'PRIVATE/parallel_v1'
ORIGINAL_INSTALL=s.install
ORIGINAL_PRICE_JOB=legacy.price_job


def sha_stream(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def verify_old_sources():
    sources=s.read(PARENT/'execution_sources.json')
    for path,want in sources['source_files'].items():
        s.require(sha_stream(s.ROOT/path)==want,'LEGACY_SOURCE_CONSERVATION:'+path)
    s.require(sha_stream(PARENT/'selection.json')==sources['selection_sha256'],'SAME_FIXED_SELECTION')
    return sources


def install():
    native=ORIGINAL_INSTALL()
    from trotterlib.df_trotter import circuit
    circuit.simulate_statevector=fast.simulate_statevector
    return native


def choose_threads():
    from pf_common_run_contract import resources
    obs=resources();measure=s.read(PARENT/'performance_check.json')
    allowed=max(1,math.floor(obs['idle_cpu']))
    feasible=[x for x in measure['results'] if x['threads']<=allowed]
    # Measured workload choice, not an arbitrary small resource cap.
    selected=min(feasible,key=lambda x:x['seconds'])['threads'] if feasible else 1
    return selected,obs



def spawned_phase_pool(*args,**kwargs):
    # Avoid forking a process after a multithreaded Aer call.
    kwargs['mp_context']=mp.get_context('spawn')
    return ProcessPoolExecutor(*args,**kwargs)


def initialize_remaining():
    s.PUBLIC=PUBLIC;s.PRIVATE=PRIVATE;s.install=install
    # Source/input loader only; no SCF, state or target reference regeneration.
    legacy.CTX=s.context('price_worker_'+str(os.getpid()))


def spawned_price_pool(*args,**kwargs):
    kwargs.update(mp_context=mp.get_context('spawn'),initializer=initialize_remaining)
    return ProcessPoolExecutor(*args,**kwargs)


def price_job(job):
    import h46_long_time_native as native
    import h8_compiler_phase_bookkeeping as phase
    # The isolated CZ diagnostic passed its unchanged local Operator check.
    # This affects only diagnostic coverage, not lowering/compiler or metrics.
    phase.NONBASIS=phase.NONBASIS|{'cz'}
    phase.ProcessPoolExecutor=spawned_phase_pool
    directory=Path(job[-1]);threads,obs=choose_threads()
    action_dir=directory.parent/(directory.name+'_actions')
    fast.configure(threads,action_dir)
    original_compose=native.compose
    identity={}
    def guarded_compose(*args,**kwargs):
        if job[3]==0:
            previous=PARENT/'PRIVATE/cost'/s.COHORT/job[0]['id']/'0/events_PRIVATE.json'
            current=directory/'events_PRIVATE.json'
            old_hash=sha_stream(previous);new_hash=sha_stream(current)
            s.require(old_hash==new_hash,'EXACT_SAME_SAVED_EVENT_FILE')
            identity.update(previous=str(previous.relative_to(s.ROOT)),
                previous_sha256=old_hash,current_sha256=new_hash,byte_identical=True,
                new_scientific_sample=False,previous_incomplete_not_counted=True)
            s.save(PUBLIC/'representative_event_identity.json',identity)
        return original_compose(*args,**kwargs)
    native.compose=guarded_compose
    try:
        value=ORIGINAL_PRICE_JOB(job)
        value.update(technical_run='parallel_v1',action_verifier='EXACT_LOCAL_OPERATORS_AER_PARALLEL',
            action_threads=threads,resource_observation=obs,event_identity=identity,
            main_scientific_conditions_unchanged=True)
        s.save(directory/'parallel_supplement.json',value)
        return value
    finally:native.compose=original_compose


def setup():
    s.require(not PUBLIC.exists() and not PRIVATE.exists(),'FRESH_PARALLEL_OUTPUT')
    sources=verify_old_sources()
    s.require(s.read(PARENT/'parallel_validation/parallel_validation.json')['status']=='PASS','PARALLEL_PREFLIGHT_PASS')
    s.require(s.read(PARENT/'performance_check_parallel.json')['status']=='PASS','PARALLEL_SYNTHETIC_PASS')
    s.require(not Path('/proc/2285682').exists(),'PREVIOUS_DRIVER_MUST_BE_DRAINED')
    PUBLIC.mkdir(parents=True)
    s.PUBLIC=PUBLIC;s.PRIVATE=PRIVATE
    for name in ['validation.json','signal_vector_validation.json','selection.json','signals.json',
        'conditional_product_compile_validation.json','provider_sources.json','technical_history.json',
        'performance_check.json','performance_check_parallel.json']:
        # Exact scientific metadata bytes, no reselection or regenerated signal.
        (PUBLIC/name).write_bytes((PARENT/name).read_bytes())
    s.install=install;legacy.price_job=price_job;legacy.ProcessPoolExecutor=spawned_price_pool
    files=dict(sources['source_files'])
    for name in ['scripts/h8_full_h_parallel_aer.py','scripts/h8_full_h_price_parallel.py']:
        files[name]=sha_stream(s.ROOT/name)
    record=dict(sources,price_SOURCE=subprocess.check_output(['git','rev-parse','HEAD'],cwd=s.ROOT,text=True).strip(),
        previous_price_SOURCE=sources['price_SOURCE'],source_files=files,technical_run='parallel_v1',
        changes=['measured Aer threading','cached exact <=3q action matrices','per-action vector checkpoints',
            'isolated CZ diagnostic coverage using the existing Operator check',
            'spawned phase/remaining workers to avoid fork after multithreaded Aer'],
        thread_benchmark_sha256=sha_stream(PARENT/'performance_check.json'),
        parallel_validation=dict(path=str((PARENT/'parallel_validation/parallel_validation.json').relative_to(s.ROOT)),
            sha256=sha_stream(PARENT/'parallel_validation/parallel_validation.json')),
        unchanged=['Hamiltonian','state','reference','scalar','components','sampling law','seed namespace',
            'candidate','cohort','n9','normalization','shot policy','priced native circuit','compiler','tolerances'],
        previous_partial_trials_in_statistical_samples=False)
    s.save(PUBLIC/'execution_sources.json',record)


def main():
    setup()
    try:legacy.main()
    except BaseException as exc:
        s.save(PUBLIC/'price_STOP.json',dict(reason=str(exc),type=type(exc).__name__,automatic_retry=False))
        raise


if __name__=='__main__':main()
