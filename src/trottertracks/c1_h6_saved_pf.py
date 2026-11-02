"""Saved rank14 H6 sector-only pilot, dormant until separate science permission.

No fit/initializer/integral engine or circuit dispatch. Reuse saved DF errors,
SectorCAR/general-K, and the existing ordered-S2 PF_rounds unchanged.
"""
from pathlib import Path
import importlib.metadata
import math
import resource
import time

from . import c1_molecule_extension as molecule

io=molecule.io
SCOPE='C1_H6_SAVED_RANK14_SECTOR_PF_V1'
FAMILY='artifacts/research_c_c1_h6_saved_pf'
RANK=14
ARMS=('CDF','XDF')
CANDIDATES=tuple((arm,s) for arm in ARMS for s in (1,4))
CONDITIONS=dict(atoms=6,distance_angstrom=1.,basis='sto-3g',Nalpha=3,Nbeta=3,
    sector_dimension=400,rank=14,CA_Ha=.00159,DF_reserve_Ha=.00159/8,
    base_time=.1,rounds=list(range(13)),substeps=[1,4],
    product_formula='ordered symmetric S2; all original frames; no PR split',
    target='own_DF_sector_ground_oracle',state_preparation='EXCLUDED')
ALLOWANCES=dict(component_maxabs=1e-10,ground_residual=1e-10,
    alias_rule='abs(E0*0.1)+pi/6 < pi',PF_phase_room='sin(pi/12)',
    rigor='numerical pilot; no rigorous reference enclosure')
CALL_LIMITS=dict(saved_input_load=1,saved_DF_gate=2,sector_context=1,
    sector_generators=2,representation_gate=2,physical_sector=1,ground_alias=3,
    sector_kernel_reference=4,PF_round_assessment=52,candidate_summary=4)
MAX_CALLS=sum(CALL_LIMITS.values())


def preflight(root,path):
    """Bytes/JSON/version checks only; no molecular array/result decoding."""
    root=Path(root);c=io.read(path)
    if (c['scope']!=SCOPE or c['status']!='UNAPPROVED_SCIENCE_CONTRACT'
            or c['conditions']!=CONDITIONS or c['allowances']!=ALLOWANCES
            or c['call_limits']!=CALL_LIMITS or c['maximum_logical_units']!=MAX_CALLS
            or c['candidate_order']!=[f'{a}:s{s}' for a,s in CANDIDATES]
            or c['new_fit'] or c['optimizer'] or c['retry'] or c['resume']
            or c['guard_policy']!=molecule.GUARD_POLICY):raise ValueError('SAVED_PF_SCOPE')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for f,w in io.read(root/c['source_manifest'])['files'].items():
        if io.sha(root/f)!=w:raise ValueError('SOURCE: '+f)
    for f,w in c['saved_inputs'].items():
        if io.sha(root/f)!=w:raise ValueError('SAVED_INPUT: '+f)
    required=[c['original_input'],c['saved_optimizer'],*c['factor_arrays'].values(),*c['saved_sector_rows'].values()]
    if any(p not in c['saved_inputs'] for p in required):raise ValueError('UNPINNED_SCIENTIFIC_INPUT')
    for name,w in c['evaluation_packages'].items():
        if importlib.metadata.version(name)!=w:raise ValueError('ENVIRONMENT: '+name)
    return c


def load_saved(root,c):
    import numpy as np
    def arrays(f):
        with np.load(root/f,allow_pickle=False) as a:return {k:a[k] for k in a.files}
    packet=arrays(c['original_input'])
    n=c['conditions']['atoms']
    if (packet['h'].shape!=(n,n) or packet['g'].shape!=(n,)*4
            or packet['h'].dtype!=np.float64 or packet['g'].dtype!=np.float64
            or packet['constant'].shape!=()):raise ValueError('ORIGINAL_INPUT_SCHEMA')
    molecule.stage1.finite_arrays(packet['h'],packet['g'],packet['constant'])
    factors={arm:arrays(c['factor_arrays'][arm]) for arm in ARMS}
    rows={arm:io.read(root/c['saved_sector_rows'][arm]) for arm in ARMS}
    return packet,factors,rows,io.read(root/c['saved_optimizer'])


def saved_DF_gate(data,row,optimizer,*,rank,n,budget):
    import numpy as np
    J,U=data['J'],data['U']
    if (J.shape!=(rank,n,n) or U.shape!=J.shape or J.dtype!=np.float64 or U.dtype!=np.float64
            or not np.array_equal(J,J.transpose(0,2,1))):raise ValueError('SAVED_DF_SCHEMA')
    molecule.stage1.finite_arrays(J,U)
    orthogonal=float(np.max(abs(U.transpose(0,2,1)@U-np.eye(n))))
    if orthogonal>1e-10:raise ValueError('SAVED_DF_ROTATION')
    if row['actual_rank']!=rank or row['guard_policy']!=molecule.GUARD_POLICY:
        raise ValueError('SAVED_DF_EVIDENCE_IDENTITY')
    for k in ('sector_error','numerical_guard'):
        if not math.isfinite(row[k]) or row[k]<0:raise ValueError('INVALID_SAVED_DF_ERROR')
    if optimizer is not None and not optimizer['success']:raise ValueError('SAVED_OPTIMIZER_NOT_SUCCESS')
    total=row['sector_error']+row['numerical_guard']
    if total>budget:raise ValueError('SAVED_DF_BUDGET_NOT_MET')
    return dict(status='PASS',actual_rank=rank,orthogonality_maxabs=orthogonal,
        saved_sector_error_Ha=row['sector_error'],saved_guard_Ha=row['numerical_guard'],
        norm_plus_guard_Ha=total,budget_Ha=budget,new_sector_error_evaluation=False)


def ground_alias(H,output,label):
    """Existing H6 eigh/residual/base-time alias rule, with reusable spectra."""
    import numpy as np
    molecule.stage1.finite_arrays(H)
    if np.max(abs(H-H.conj().T))>1e-10:raise ValueError('NONHERMITIAN_SECTOR_HAMILTONIAN')
    vals,vecs=np.linalg.eigh(H)
    residual=float(np.linalg.norm(H@vecs[:,0]-vals[0]*vecs[:,0]))
    molecule.stage1.finite_arrays(vals,vecs,residual)
    alias_room=math.pi-(abs(float(vals[0])*.1)+math.pi/6)
    if residual>1e-10 or alias_room<=0:raise ValueError('GROUND_OR_ALIAS_FAIL')
    with (output/('ground_'+label+'.npz')).open('xb') as f:
        np.savez_compressed(f,energies=vals,eigenvectors=vecs,energy=vals[0],state=vecs[:,0])
    return (vals,vecs),dict(energy=float(vals[0]),residual=residual,alias_margin=alias_room,
        status='PASS',rule=ALLOWANCES['alias_rule'])


def summarize(rows):
    if len(rows)!=13 or [r['m'] for r in rows]!=list(range(13)):raise ValueError('PF_REGISTERED_ROUNDS')
    bad=[r['m'] for r in rows if r['known_part_status']=='INFEASIBLE_REGISTERED_PHASE_ROOM']
    return dict(status='INFEASIBLE_REGISTERED_PHASE_ROOM' if bad else 'POTENTIALLY_FEASIBLE',
        failed_sufficient_rounds=bad,maximum_pf_operator_error=max(r['pf_operator_error'] for r in rows),
        reason='registered PF sufficient condition, not physical impossibility' if bad else
            'saved DF reserve and PF phase-room only; synthesis/CA not qualified',
        CA_achieved=None,synthesis_error=None,shots=None,total_cost=None)


def run(root,contract_path,output):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from threadpoolctl import threadpool_limits
    root,output=Path(root),Path(output);c=preflight(root,contract_path)
    output.mkdir(parents=True,exist_ok=False);(output/'cells').mkdir();(output/'kernels').mkdir()
    calls=molecule.integrated.Calls(output,CALL_LIMITS,MAX_CALLS);start=time.monotonic();cpu=time.process_time()
    plan=molecule.integrated.resource_plan(task_count=4,full_dimension=c['conditions']['sector_dimension'])
    io.save(output/'contract.json',c);io.save(output/'resource_plan.json',plan)
    try:
        packet,factors,saved_rows,optimizer=calls('saved_input_load','original H6 and saved CDF14/XDF14',lambda:load_saved(root,c))
        df={}
        for arm in ARMS:
            df[arm]=calls('saved_DF_gate',arm,lambda arm=arm:saved_DF_gate(factors[arm],saved_rows[arm],
                optimizer if arm=='CDF' else None,rank=c['conditions']['rank'],n=len(packet['h']),budget=c['conditions']['DF_reserve_Ha']))
            io.save(output/(arm+'_saved_DF_gate.json'),df[arm])
        ctx=calls('sector_context','sector400',lambda:molecule.SectorCAR(len(packet['h'])))
        generators={};representations={}
        with threadpool_limits(limits=1 if plan['workers']>1 else None):
            for arm in ARMS:
                data=factors[arm]
                generators[arm]=calls('sector_generators',arm,lambda data=data:ctx.generators(
                    packet['h'],packet['g'],packet['constant'],data['J'],data['U']))
                if any(G.shape!=(ctx.d,ctx.d) for G in generators[arm]):raise ValueError('SECTOR_DIMENSION')
                representations[arm]=calls('representation_gate',arm,lambda arm=arm,data=data:
                    molecule.representation_gate(ctx,packet,data['J'],data['U'],generators[arm]))
                io.save(output/(arm+'_representation.json'),representations[arm])
            physical=calls('physical_sector','original Hamiltonian for ground/alias only',lambda:ctx.physical(packet['h'],packet['g'],packet['constant']))
            ground={};eigenpairs={}
            for arm,G in [('physical',[physical]),*generators.items()]:
                eigenpairs[arm],ground[arm]=calls('ground_alias',arm,lambda arm=arm,G=G:ground_alias(sum(G),output,arm))
                io.save(output/('ground_'+arm+'.json'),ground[arm])
            def job(key):
                arm,s=key;label=f'{arm}:s{s}'
                def cell_call(kind,name,function):
                    def evaluate():
                        value=function()
                        if kind=='sector_kernel_reference':
                            # Immutable per-arm kernel; no cross-arm cache reuse.
                            with (output/'kernels'/f'{arm}_s{s}.npz').open('xb') as f:np.savez_compressed(f,pf=value[0]['pf'])
                        elif kind=='PF_round_assessment':
                            error=value['pf_operator_error']
                            if not math.isfinite(error) or error<0:raise ValueError('NONFINITE_PF_ERROR')
                            if value['attenuation']!=1. or value['rte_corrected_operator_error']!=0.:
                                raise ValueError('DETERMINISTIC_MODEL_CHANGED')
                            io.save(output/'cells'/f'{arm}_s{s}_m{value["m"]}.json',value)
                        return value
                    return calls(kind,name,evaluate)
                rows=molecule.PF_rounds(generators[arm],s,cell_call,label,eigenpair=eigenpairs[arm])
                summary=calls('candidate_summary',label,lambda:summarize(rows))
                io.save(output/f'{arm}_s{s}_summary.json',summary)
                return label,dict(arm=arm,s=s,rounds=rows,summary=summary)
            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:results=dict(pool.map(job,CANDIDATES))
        io.save(output/'result.json',dict(status='H6_SAVED_RANK14_SECTOR_PF_COMPLETE',
            saved_DF=df,representation=representations,ground=ground,candidates=results,
            CA_achieved=None,rigorous_CA_qualification='UNRESOLVED_REFERENCE_ENCLOSURES',
            native_build_compile=0,new_fit=0,optimizer=0,retry=0,resume=0,shots=None,total_cost=None,mandatory_stop=True))
    except BaseException as e:
        calls.stop.set();io.save(output/'failure.json',dict(error=str(e),first_failure=calls.first_failure));raise
    finally:
        io.save(output/'STOP.json',dict(mandatory_stop=True,failed=calls.stop.is_set(),retry=0,resume=0,
            scientific_units=sum(calls.counts.values()),call_counts=dict(calls.counts)))
        io.save(output/'resources.json',dict(wall_seconds=time.monotonic()-start,process_CPU_seconds=time.process_time()-cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
