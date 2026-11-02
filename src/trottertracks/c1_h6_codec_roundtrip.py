"""Dormant saved-H6 corrected-codec evaluation. Separate science permission.

The entry reuses saved before-codec J/U and the existing sector evaluator.
No initializer/decomposition, optimizer, gradient, PF or circuit is called.
"""
from pathlib import Path
import importlib.metadata
import subprocess
import time

from . import c1_h6_initialization as old
from . import c1_real_orthogonal_codec as codec
from .c1_stage3_molecular import component_check

io=old.io
SCOPE='C1_H6_CORRECTED_CODEC_ROUNDTRIP_V1'
FAMILY='artifacts/research_c_c1_h6_codec_roundtrip'
CODEC_LIMITS=dict(saved_codec_load=1,saved_initial_identity=2,corrected_codec=2,tensor_assessment=4)
SECTOR_LIMITS=dict(snapshots_load=1,sector_context=1,physical_sector=1,sector_assessment=4,
                    saved_sector_replay=2,fidelity_and_budget_gate=2)
ALLOWANCES=dict(component_maxabs=old.ATOL,tensor_frobenius='1e-10*max(1,||g_before||F)',
                saved_sector_replay_absolute_Ha=old.ATOL)


def preflight(root,path,*,worker=False):
    root=Path(root);c=io.read(path)
    if (c['scope']!=SCOPE or c['status']!='UNAPPROVED_SCIENCE_CONTRACT' or c['ranks']!=[14,18]
            or c['codec_calls']!=CODEC_LIMITS or c['sector_calls']!=SECTOR_LIMITS
            or c['maximum_logical_units']!=20 or c['new_fit'] or c['retry'] or c['resume']
            or c['DF_reserve_Ha']!=.00159/8 or c['guard_policy']!=old.molecule.GUARD_POLICY
            or c['allowances']!=ALLOWANCES):
        raise ValueError('CORRECTED_CODEC_SCOPE')
    manifest=root/c['source_manifest']
    if io.sha(manifest)!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for f,w in io.read(manifest)['files'].items():
        if io.sha(root/f)!=w:raise ValueError('SOURCE: '+f)
    for f,w in c['saved_inputs'].items():
        if io.sha(root/f)!=w:raise ValueError('SAVED_INPUT: '+f)
    for k,w in c['fit_packages' if worker else 'evaluation_packages'].items():
        if importlib.metadata.version(k)!=w:raise ValueError('ENVIRONMENT: '+k)
    if worker:
        import ffsim
        if io.sha(Path(ffsim.__file__).parent/'linalg/util.py')!=c['ffsim_util_sha256']:
            raise ValueError('FFSIM_UTIL_SOURCE')
    return c


def worker(root,output,c):
    import numpy as np
    from unittest.mock import patch
    import scipy.optimize
    root,output=Path(root),Path(output);output.mkdir(exist_ok=False)
    calls=old.Calls(output,CODEC_LIMITS,sum(CODEC_LIMITS.values()));start=time.monotonic()
    try:
        def load():
            with np.load(root/c['original_input'],allow_pickle=False) as a:g=a['g']
            with np.load(root/c['XDF_full'],allow_pickle=False) as a:full=(a['J'],a['U'])
            saved={}
            for r in (14,18):
                with np.load(root/c['initial_arrays'][str(r)],allow_pickle=False) as a:saved[r]=(a['J'],a['U'])
            return g,full,saved
        g,full,saved=calls('saved_codec_load','saved originals only',load);results={}
        with patch.object(scipy.optimize,'minimize',side_effect=AssertionError('OPTIMIZER_FORBIDDEN')):
            for r in (14,18):
                J,U=saved[r]
                identity=calls('saved_initial_identity',str(r),lambda:old.initialization_identity(
                    J,U,full[0][:r],full[1][:r]))
                io.save(output/f'R{r}_identity.json',identity)
                J2,U2,x,report,raw=calls('corrected_codec',str(r),lambda:codec.roundtrip(J,U))
                io.save(output/f'R{r}_codec.json',report)
                old._save_arrays(output/f'R{r}_parameters.npz',x=x,**raw)
                metrics={}
                for state,A,B in [('before',J,U),('after',J2,U2)]:
                    old._save_arrays(output/f'R{r}_{state}.npz',J=A,U=B)
                    metrics[state]=calls('tensor_assessment',f'R{r}_{state}',lambda A=A,B=B:old.tensor_metrics(g,A,B))
                    io.save(output/f'R{r}_{state}_tensor.json',metrics[state])
                results[str(r)]=dict(identity=identity,codec=report,tensor=metrics)
        io.save(output/'result.json',dict(ranks=results,new_fit=0,optimizer_iterations=0))
    except BaseException as e:
        calls.stop.set();io.save(output/'failure.json',dict(error=str(e),first_failure=calls.first_failure));raise
    finally:old._finish(output,calls,start)


def run(root,contract_path,output,fit_python):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from threadpoolctl import threadpool_limits
    root,output=Path(root),Path(output);c=preflight(root,contract_path)
    output.mkdir(parents=True,exist_ok=False);calls=old.Calls(output,SECTOR_LIMITS,sum(SECTOR_LIMITS.values()));start=time.monotonic()
    io.save(output/'contract.json',c)
    try:
        subprocess.run([str(Path(fit_python).absolute()),str(root/c['entry']),'--contract',
            str(Path(contract_path).absolute()),'--worker','--output',str(output/'codec')],check=True)
        def load():
            with np.load(root/c['original_input'],allow_pickle=False) as a:packet={k:a[k] for k in a.files}
            saved={}
            for r in (14,18):
                for state in ('before','after'):
                    with np.load(output/'codec'/f'R{r}_{state}.npz',allow_pickle=False) as a:saved[r,state]=(a['J'],a['U'])
            return packet,saved
        packet,snapshots=calls('snapshots_load','original h/g/c and codec states',load)
        ctx=calls('sector_context','sector400',lambda:old.molecule.SectorCAR(len(packet['h'])))
        physical=calls('physical_sector','original h/g/c',lambda:ctx.physical(packet['h'],packet['g'],packet['constant']))
        plan=old.resource_plan(task_count=4,full_dimension=ctx.d);io.save(output/'resource_plan.json',plan)
        def assess(key):
            r,state=key;J,U=snapshots[key]
            row,G=calls('sector_assessment',f'R{r}_{state}',lambda:old.molecule.rank_assessment(ctx,packet,J,U,physical,r))
            row.update(state=state,budget_Ha=c['DF_reserve_Ha'],budget_status='PASS' if
                row['sector_error']+row['numerical_guard']<=c['DF_reserve_Ha'] else 'NOT_MET')
            io.save(output/f'R{r}_{state}_sector.json',row);return key,(row,sum(G))
        with threadpool_limits(limits=1 if plan['workers']>1 else None):
            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:rows=dict(pool.map(assess,snapshots))
        results={}
        for r in (14,18):
            before,after=rows[r,'before'][0],rows[r,'after'][0]
            def replay(r=r,before=before):
                saved=io.read(root/c['saved_sector_rows'][str(r)])
                delta=abs(before['sector_error']-saved['sector_error'])
                if delta>old.ATOL:raise ValueError('SAVED_BEFORE_SECTOR_MISMATCH')
                return dict(status='PASS',difference_Ha=delta)
            comparison=calls('saved_sector_replay',str(r),replay)
            io.save(output/f'R{r}_replay.json',comparison)
            def gate(r=r,before=before,after=after):
                a,b=rows[r,'before'][1],rows[r,'after'][1]
                fidelity=component_check(a,b,old.ATOL,label='corrected codec sector equivalence')
                difference=float(np.linalg.norm(a-b,2))
                if before['budget_status']!='PASS' or after['budget_status']!='PASS':
                    raise ValueError('DF_BUDGET_NOT_PRESERVED')
                return dict(status='PASS',operator=fidelity,sector_operator_difference_norm=difference,
                    before=before,after=after,rigorous_enclosure=None)
            results[str(r)]=calls('fidelity_and_budget_gate',str(r),gate)
            io.save(output/f'R{r}_result.json',results[str(r)])
        io.save(output/'result.json',dict(status='CORRECTED_CODEC_ROUNDTRIP_COMPLETE',ranks=results,
            new_fit=0,optimizer_iterations=0,retry=0,resume=0,fit_success=None,
            total_resource_superiority=None,mandatory_stop=True))
    except BaseException as e:
        calls.stop.set();io.save(output/'failure.json',dict(error=str(e),first_failure=calls.first_failure));raise
    finally:old._finish(output,calls,start)
