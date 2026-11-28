"""Reuse accepted H-chain leaves without creating another scientific owner."""
import base64
from contextlib import closing
import os
import json
from pathlib import Path
import sqlite3

from run_rpe_hchain_preparation import leaves, planned_inputs, preparation_request
from trottertracks.rpe_preparation_artifacts import ArtifactCache
from trottertracks.rpe_q3_coordinator import digest
from trottertracks.rpe_completion_acceptance import validate_preparation_result

PLAN_FIELDS=("owner","binding","target","compiler","scope","price_cohort",
             "verified_saved_preparations","provider_contract")


def _read(path):
    return json.loads(Path(path).read_bytes())


def _positive_end(run):
    """Do not use an uncertain or still-writing predecessor."""
    try:
        launch,ended=(_read(run/name) for name in ('LAUNCH.json','EXIT.json'))
        return bool(launch.get('invocation_id') and launch['invocation_id']==ended.get('invocation_id'))
    except (OSError,ValueError,KeyError,TypeError):
        return False


def _matching_packet(packet,plan):
    return {k:packet[k] for k in PLAN_FIELDS if k in packet}=={k:plan[k] for k in PLAN_FIELDS if k in plan}


def _completion(prior,row):
    # The immutable staged transfer bytes are the original accepted digest.
    job,attempt=row['job_id'],row['attempt_id']
    if any(not isinstance(v,str) or Path(v).name!=v or v in ('','.','..') for v in (job,attempt)):
        raise ValueError('Invalid predecessor identity')
    directory=prior/'distributed/transfers'/job/attempt
    if directory.resolve()!=directory or not directory.is_dir():
        raise ValueError('Missing accepted transfer')
    files={}
    for path in directory.iterdir():
        if path.name=='REQUEST.json':
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError('Unexpected preparation transfer entry')
        files[path.name]=base64.b64encode(path.read_bytes()).decode()
    if digest(files)!=row['completion_sha256']:
        raise ValueError('Accepted preparation transfer differs')
    return tuple(_read(directory/name) for name in
                 ('PROCESS_RETURN.json','OUTCOME.json','PREPARATION_RECEIPT.json'))


def reuse_prior_leaves(executor,plan,settings,budgets,run):
    """Only exact fixed-plan, explicitly pinned-runtime leaves are reused.

    The new graph has not been frozen yet. No job/attempt or budget record
    is synthesized: the original accepted owner remains the provenance.
    """
    run=Path(run).resolve()
    core=settings.get('openblas_coretype')
    if core not in ('HASWELL','NEHALEM'):
        return 0
    targets={}
    for stage in ('signal','compile','audit'):
        if not budgets.get(stage):
            continue
        for cell in planned_inputs(plan,stage).values():
            paths=plan['scope']['price_paths'] if stage!='signal' and cell['method']=='PR' else 1
            for index in range(paths):
                targets[(stage,cell['id'],index)]=cell
    if not targets:
        return 0
    reused=0
    for prior in map(lambda p:Path(p).resolve(),executor.prior_run_dirs):
        if prior==run or not _positive_end(prior):
            continue
        db=prior/'distributed/QUEUE.sqlite3'; root=prior/'distributed/artifacts'
        if not db.is_file() or not root.is_dir():
            continue
        wal=Path(str(db)+'-wal')
        if wal.is_file() and wal.stat().st_size:
            continue  # immutable SQLite must not ignore committed WAL rows
        old=ArtifactCache(root)
        with closing(sqlite3.connect(db.as_uri()+'?mode=ro&immutable=1',uri=True)) as connection:
            connection.row_factory=sqlite3.Row
            rows=connection.execute("""SELECT job_id,attempt_id,host_id,request,contract,request_sha256,
                completion_sha256,eligibility FROM jobs WHERE state='ACCEPTED' AND adopted_epoch IS NOT NULL
                AND method LIKE 'PREPARATION:%' AND json_extract(request,'$.preparation.provider')='HCHAIN_SCOPED'""")
            for record in rows:
                if getattr(executor,'_user_stop_requested',lambda:False)():
                    return reused
                try:
                    row=dict(record)
                    for field in ('request','contract','eligibility'):row[field]=json.loads(row[field])
                    request=row['request'];prep=request['preparation'];stage=prep['stage']
                    if (digest(request)!=row['request_sha256'] or
                            row['contract']['evaluation'].get('fixed_plan_sha256')!=settings['plan']['sha256'] or
                            prep['source']!=settings['provider_source'] or
                            prep['required_paths']!={'hchain_assets':settings['assets']} or
                            prep['runtime_requirements'].get('openblas_coretype')!=core):
                        continue
                    packet_sha=prep['inputs']['PLAN.json']
                    if old.missing([packet_sha]):continue
                    packet=_read(old.path(packet_sha))
                    if not _matching_packet(packet,plan):continue
                    command=prep['command']
                    if ('--spectral-signals' in command)!=(stage=='signal' and bool(settings.get('spectral_signals'))):
                        continue
                    batch=json.loads(command[command.index('--targets')+1])
                    matching=[]
                    for cell,index,name in leaves(packet,stage,batch):
                        key=(stage,cell['id'],index)
                        if targets.get(key)==cell and not (run/name).exists():
                            matching.append((key,cell,name))
                    if not matching:continue
                    expected=preparation_request(packet,packet_sha,stage,batch,source=settings['provider_source'],
                        assets=settings['assets'],spectral_signals=bool(settings.get('spectral_signals')),
                        openblas_coretype=core)['preparation']
                    if prep['runtime_requirements']!=expected['runtime_requirements'] or prep['command']!=expected['command']:
                        continue
                    process,outcome,receipt=_completion(prior,row)
                    if outcome.get('status')!='PREPARATION_COMPLETE':continue
                    validate_preparation_result(row,process,outcome,old)
                    runtime=receipt['runtime'];environment=receipt['numerical_environment']
                    if (receipt['returncode']!=0 or receipt.get('signal') is not None or
                            receipt['source']!=prep['source'] or runtime.get('openblas_coretype')!=core or
                            runtime['packages']!=prep['runtime_requirements']['packages'] or
                            environment.get('OPENBLAS_CORETYPE')!=core or
                            any(environment.get(k)!='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'))):
                        continue
                    if set(outcome['artifacts'])!=set(prep['outputs']):continue
                    for key,cell,name in matching:
                        manifest={name:outcome['artifacts'][name]}
                        value=_read(old.path(manifest[name]))
                        if value.get('owner')!=plan['owner']:continue
                        if stage!='signal' and cell.get('R_outer'):
                            side=str(Path(name).with_suffix('.events.json.xz'))
                            manifest[side]=outcome['artifacts'][side]
                            if value.get('events',{}).get('sha256')!=manifest[side]:continue
                        cache=executor.artifact_cache
                        for sha in cache.missing(manifest.values()):
                            if not cache.share_verified_file(old.path(sha),sha):
                                cache.put_file(old.path(sha),sha)
                        # Write the original owner link before the JSON marker
                        # becomes visible to batches()/the candidate assembler.
                        provenance=run/'distributed/HCHAIN_PREPARATION_REUSE.jsonl'
                        provenance.parent.mkdir(parents=True,exist_ok=True)
                        with provenance.open('a') as stream:
                            stream.write(json.dumps(dict(stage=key[0],cell=key[1],index=key[2],
                                prior_run=str(prior),job_id=row['job_id'],attempt_id=row['attempt_id'],
                                request_sha256=row['request_sha256'],completion_sha256=row['completion_sha256'],
                                artifacts=manifest),separators=(',',':'))+'\n')
                            stream.flush();os.fsync(stream.fileno())
                        ordered=sorted(manifest,key=lambda n:(not n.endswith('.events.json.xz'),n))
                        cache.materialize({n:manifest[n] for n in ordered},run,immutable=True)
                        targets.pop(key,None);reused+=1
                except (OSError,ValueError,KeyError,TypeError,IndexError,sqlite3.Error):
                    continue
    return reused
