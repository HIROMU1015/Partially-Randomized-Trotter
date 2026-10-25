"""H2 full pilot / H6 sector DF-PF pilot. Dormant until explicit execution.

No molecular values are loaded at import or planning. Fixed H4 runners and
their consumed execution controls are not reused as execution entry points.
"""
from dataclasses import replace
from fractions import Fraction
import importlib.metadata
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time

from . import c1_stage1 as stage1
from . import c1_stage1_execution as io
from . import c1_stage2_native as native
from . import c1_stage3_accuracy as accuracy
from . import c1_stage3_connection as connection
from . import c1_stage3_deterministic_accuracy as det
from . import c1_stage3_reference as reference
from . import c1_stage3_integrated as integrated
from .c1_stage3_molecular import component_check

FAMILY='artifacts/research_c_c1_h2_h6_science'
SCOPE='C1_H2_FULL_H6_SECTOR_PREPARATION_V1'
GUARD_POLICY='sector_operation_scale_guard8_proposal_v1'
FIT_PACKAGES=dict(ffsim='0.0.83',numpy='1.26.4',scipy='1.14.1',jax='0.4.38',jaxlib='0.4.38',
                  qiskit='2.4.2',pyscf='2.14.0')


def specification(n):
    if n not in (2,6):raise ValueError('Only registered H2/H6')
    d=math.comb(n,n//2)**2;rank=n*(n+1)//2
    return dict(atoms=n,norb=n,modes=2*n,Nalpha=n//2,Nbeta=n//2,
        distance_angstrom=1.,basis='sto-3g',charge=0,spin=0,
        sector_dimension=d,full_dimension=1<<(2*n),controlled_dimension=1<<(2*n+1),
        sector_matrix_bytes=16*d*d,full_matrix_bytes=16*(1<<(4*n)),
        controlled_matrix_bytes=16*(1<<(4*n+2)),rank_ladder=list(range(1,rank+1)),
        scope='H2_FULL_NUMERICAL_CA_COST_PILOT' if n==2 else 'H6_SECTOR_DF_PF_ONLY',
        native_allowed=n==2)


def call_limits(n):
    p=specification(n);r=len(p['rank_ladder'])
    calls=dict(scf=1,spatial_integrals=1,physical_sector=1,sector_context=1,
        explicit_XDF=1,CDF_fit=r,rank_assessment=2*r,representation_gate=2,ground_alias=3,
        sector_kernel_reference=4,PF_round_assessment=52)
    if n==2:calls.update(full_CAR_context=1,full_generators=2,full_kernel_reference=4,
        native_model=2,gaussian_build=2*(r+1),circuit_build=7,compile=7,
        operator_decode=14,segment_assessment=7,metrics=7,
        round_accuracy=52,energy_assessment=4,cost_aggregation=4,cost_comparison=2)
    return calls


def rank_choice(records,cap):
    """First converged rank passing a physical-sector numerical DF budget.

    No assumed monotonicity, tensor residual proxy, H4 rank inheritance, or
    best-of-restarts. A missing/failed optimizer result stops the ladder.
    """
    for row in records:
        if row['status']!='CONVERGED':raise ValueError('FIT_OR_DECOMPOSITION_NOT_CONVERGED')
        value=row['sector_error'];guard=row['numerical_guard']
        accuracy._finite(value,'DF norm');accuracy._finite(guard,'DF guard')
        if value+guard<=cap:return row['rank']
    return None


def integrals(n,call,*,factory=None):
    """Reuse PySCF RHF/ao2mo workflow from historical h4_molecular_inputs.

    Only chemistry interfaces/geometry generalized; no new integral engine.
    Output is chemist ERI, not OpenFermion's (ps|qr) array convention.
    """
    p=specification(n)
    if factory is not None:return factory(p,call)
    import numpy as np
    from pyscf import ao2mo,gto,scf
    def run_scf():
        mol=gto.M(atom=[('H',(0.,0.,float(i))) for i in range(n)],unit='Angstrom',
                  basis=p['basis'],charge=0,spin=0,verbose=0)
        mf=scf.RHF(mol);mf.conv_tol=1e-12;mf.kernel()
        if not mf.converged:raise ValueError('SCF_NOT_CONVERGED')
        return mol,mf
    mol,mf=call('scf','canonical RHF',run_scf)
    def export():
        mo=np.asarray(mf.mo_coeff,dtype=np.float64)
        h=mo.T@mf.get_hcore()@mo
        g=ao2mo.restore(1,ao2mo.kernel(mol,mo),n)
        if np.max(abs(mo.T@mf.get_ovlp()@mo-np.eye(n)))>1e-10:
            raise ValueError('MO_ORTHONORMALITY')
        return dict(h=np.asarray(h,dtype=np.float64),g=np.asarray(g,dtype=np.float64),
                    constant=np.asarray(mol.energy_nuc()),mo_coeff=mo)
    return call('spatial_integrals','chemist original h/g/c',export)


def factor(g,rank,*,optimize,fitter=None,metadata_callback=None):
    """One existing fixed ffsim call; rank is a new proposed ladder condition."""
    if fitter is None:
        from .c1_ffsim_compat import double_factorized
        import ffsim  # fixed package enables x64 before the JAX check
        import jax
        if not jax.config.jax_enable_x64:raise ValueError('JAX_X64_REQUIRED')
        fitter=double_factorized
    kw=dict(tol=1e-8,max_vecs=rank,optimize=optimize,cholesky=True)
    if optimize:kw.update(method='L-BFGS-B',return_optimize_result=True)
    value=fitter(g,**kw)
    if optimize:
        J,U,result=value
        if metadata_callback:metadata_callback(dict(success=bool(result.success),status=int(result.status),
            message=str(result.message),fun=float(result.fun),nit=int(result.nit),nfev=int(result.nfev)))
        if not result.success:raise ValueError('CDF_OPTIMIZER_STOP: '+str(result.message))
    else:J,U=value
    stage1.finite_arrays(J,U)
    import numpy as np
    if (J.ndim!=3 or J.shape!=U.shape or J.shape[1:]!=g.shape[:2]
            or not 0<len(J)<=rank or not np.array_equal(J,J.transpose(0,2,1))
            or np.max(abs(U.transpose(0,2,1)@U-np.eye(len(g))))>1e-10):
        raise ValueError('DF_EXPORT_SCHEMA')
    return J,U


def factor_worker(input_path,output,rank,optimize):
    """Separate fit environment; no grant or mutation of an existing venv."""
    import numpy as np
    if {k:importlib.metadata.version(k) for k in FIT_PACKAGES}!=FIT_PACKAGES:
        raise ValueError('FIT_ENVIRONMENT')
    output=Path(output)
    with np.load(input_path,allow_pickle=False) as a:g=a['g']
    metadata={}
    def capture(value):
        metadata.update(value);io.save(output.with_suffix('.optimizer.json'),metadata)
    output.parent.mkdir(parents=True,exist_ok=True)
    J,U=factor(g,rank,optimize=optimize,metadata_callback=capture)
    with output.open('xb') as f:np.savez(f,J=J,U=U)
    metadata.update(requested_rank=rank,actual_rank=len(J),optimize=optimize,
        input_sha256=io.sha(input_path),export_sha256=io.sha(output),number_representation=True,
        original_g_kappa_required=True,ffsim_version=importlib.metadata.version('ffsim'))
    io.save(output.with_suffix('.json'),metadata)


class SectorCAR:
    """Sparse sector bilinears from the existing excitation table.

    Only spin-preserving bilinears are used. Complete one-body and density
    factors conserve Nalpha/Nbeta, so restriction commutes with products.
    Individual Gaussian gates need not preserve this sector; no intermediate
    circuit projection is used or authorized for H6.
    """
    def __init__(self,n):
        import numpy as np
        from scipy.sparse import csr_matrix
        df=stage1.legacy_df_module()
        self.n=n;self.sector=df.PhysicalSector.spin_sector(n_qubits=2*n,nelec_alpha=n//2,nelec_beta=n//2)
        table=df._build_excitation_table(self.sector);self.d=self.sector.dimension
        self.bilinears={}
        for p in range(2*n):
            for q in range(2*n):
                if p%2!=q%2:continue
                take=(table.p_indices==p)&(table.q_indices==q)
                self.bilinears[p,q]=csr_matrix((table.signs[take],
                    (table.target_positions[take],table.source_positions[take])),shape=(self.d,self.d))
        self.E=[[self.bilinears[2*p,2*q]+self.bilinears[2*p+1,2*q+1] for q in range(n)] for p in range(n)]

    def one(self,h):
        import numpy as np
        h=np.asarray(h)
        return sum((h[p,q]*self.E[p][q] for p in range(self.n) for q in range(self.n))).toarray()

    def raw_frame(self,J,U):
        import numpy as np
        from scipy.sparse import csr_matrix
        densities=[sum((U[p,i]*U[q,i].conjugate()*self.E[p][q]
                    for p in range(self.n) for q in range(self.n))) for i in range(self.n)]
        out=csr_matrix((self.d,self.d),dtype=complex)
        for i in range(self.n):
            weighted=sum((J[i,j]*densities[j] for j in range(self.n)))
            out+=.5*(densities[i]@weighted)
        return out.toarray()

    def generators(self,h,g,constant,J,U):
        import numpy as np
        base=self.one(stage1.corrected_kappa(h,g))+float(constant)*np.eye(self.d)
        return [base]+[self.raw_frame(a,b) for a,b in zip(J,U,strict=True)]

    def candidate(self,model):
        """Independent general-K I/Z/ZZ expansion on complete sector factors."""
        import numpy as np
        from scipy.sparse import eye,csr_matrix
        from .c1_general_k_adapter import general_k_coefficients
        out=self.one(np.asarray(model.spatial_one_body_tensor))+model.constant*np.eye(self.d)
        identity=eye(self.d,format='csr',dtype=complex)
        for frame in model.frames:
            V=np.asarray(frame.orbital_rotation);zs=[]
            for a in range(2*self.n):
                density=csr_matrix((self.d,self.d),dtype=complex)
                for (p,q),E in self.bilinears.items():density+=V[p,a]*V[q,a].conjugate()*E
                zs.append(identity-2*density)
            block=csr_matrix((self.d,self.d),dtype=complex)
            for support,coefficient in general_k_coefficients(frame.K):
                product=identity
                for a in support:product=product@zs[a]
                block+=coefficient*product
            out+=block.toarray()
        return out

    def physical(self,h,g,constant):
        permutation=stage1.bit_reverse_indices(2*self.n)
        # Existing OF sparse physical Hamiltonian; only sector is densified.
        return stage1.physical_sector(h,g,constant,permutation[self.sector.basis_indices])[0]


def rank_assessment(ctx,packet,J,U,physical,rank,*,prebuilt=None):
    import numpy as np
    gens=ctx.generators(packet['h'],packet['g'],packet['constant'],J,U) if prebuilt is None else prebuilt
    total=sum(gens)
    if np.max(abs(total-total.conj().T))>1e-10:raise ValueError('NONHERMITIAN_DF')
    norm=float(np.linalg.norm(total-physical,2))
    # Proposed H2/H6 engineering policy, never a rigorous rounding enclosure.
    scale=float(np.linalg.norm(physical,'fro')+sum(np.linalg.norm(x,'fro') for x in gens))
    ops=(ctx.n**4+len(J)*ctx.n**3)*(ctx.d+1)
    guard=8*np.finfo(float).eps*ops*max(1.,scale)
    row=dict(rank=rank,actual_rank=len(J),status='CONVERGED',sector_error=norm,numerical_guard=guard,
             evidence_kind='numerical_pilot',guard_policy=GUARD_POLICY,rigorous_enclosure=None)
    return row,gens


def representation_gate(ctx,packet,J,U,G):
    import numpy as np
    from .c1_general_k_adapter import from_ffsim_number_arrays
    model=from_ffsim_number_arrays(one_body_tensor=stage1.corrected_kappa(packet['h'],packet['g']),
        diag_coulomb_mats=J,orbital_rotations=U,constant=float(packet['constant']),z_representation=False,
        frame_ids=[f'F{i}' for i in range(len(J))],group_ids=[f'F{i}' for i in range(len(J))],
        source_identity='fresh-DF',input_identity='fresh-input',one_body_role='ffsim_corrected_kappa',enable_general_k=True)
    raw=sum(G);candidate=ctx.candidate(model)
    return component_check(candidate,raw,1e-10,label='sector raw-CAR vs general-K')


def PF_rounds(generators,substeps,call,label,*,eigenpair=None):
    """400x400 maximum; reuse expm/S2 and repeated squaring, no 4096 operator."""
    import numpy as np
    from scipy.linalg import eigh
    h=Fraction(1,10*substeps)
    def prepare():
        kernel=reference.reference_kernels(generators,h,ld=len(generators)-1)
        vals,vecs=eigh(sum(generators)) if eigenpair is None else eigenpair
        return kernel,vals,vecs
    kernels,vals,vecs=call('sector_kernel_reference',label,prepare)
    power=np.linalg.matrix_power(kernels['pf'],substeps);rows=[]
    for m in range(13):
        def assess(m=m,power=power):
            exact=(vecs*np.exp(-1j*.1*(1<<m)*vals))@vecs.conj().T
            error=float(np.linalg.norm(power-exact,2))
            return dict(m=m,pf_operator_error=error,
                rte_corrected_operator_error=0.,attenuation=1.,evidence_kind='numerical_sector_operator_norm',
                known_part_status='POTENTIALLY_FEASIBLE' if error<=math.sin(math.pi/12)
                    else 'INFEASIBLE_REGISTERED_PHASE_ROOM',
                CA_achieved=None,synthesis_error=None)
        rows.append(call('PF_round_assessment',f'{label}:m{m}',assess))
        if m<12:power=power@power
    return rows


def native_H2(packet,factors,pf,df_errors,call,output):
    """Reuse NativeBuilder/P0/full CAR and old CA/cost functions on H2 only."""
    import numpy as np
    from qiskit import qpy
    from qiskit.quantum_info import Operator
    from .c1_general_k_adapter import from_ffsim_number_arrays
    from .pr_dictionary_validation.fixture import CompilerProfile
    from .pr_dictionary_validation.gates import compiled_wrapper
    from .pr_dictionary_validation.f2_execution import circuit_metrics
    if len(packet['h'])!=2:raise ValueError('H6_NATIVE_OUT_OF_SCOPE')
    parent=io.read(Path(__file__).resolve().parents[2]/'docs/research/research_c_c1_stage3_cost_compat_contract_20261025.json')
    proposal=parent['proposal'];target=accuracy.AccuracyTarget(**proposal['accuracy_target'])
    tol=proposal['correctness_tolerance'];profile=CompilerProfile(**{**proposal['compiler'],
        'basis_gates':tuple(proposal['compiler']['basis_gates'])})
    builder=native.NativeBuilder(call=lambda kind,label:call(kind,label,lambda:None))
    circuits=Path(output)/'circuits';circuits.mkdir()
    def segment(label,build,ideal):
        qc=call('circuit_build',label,build)
        compiled=call('compile',label,lambda:compiled_wrapper(qc,profile))
        for name,q in [('raw',qc),('compiled',compiled)]:
            with (circuits/(label+'_'+name+'.qpy')).open('xb') as f:qpy.dump(q,f)
        def matrix(q):return Operator(q.remove_final_measurements(inplace=False) if q.num_clbits else q).data
        a=call('operator_decode',label+':raw',lambda:matrix(qc));b=call('operator_decode',label+':compiled',lambda:matrix(compiled))
        report=call('segment_assessment',label,lambda:det.numerical_segment(a,b,ideal,tol,label))
        metrics=call('metrics',label,lambda:circuit_metrics(compiled))
        return report,metrics
    H=np.array([[1,1],[1,-1]],complex)/math.sqrt(2);eye=np.eye(16)
    shells={};sm={}
    for name in ('head','X','Y'):
        def build(name=name):
            a,b=connection.readout_segments(4,'Y' if name=='Y' else 'X')
            return a if name=='head' else b
        shells[name],sm[name]=segment('shell_'+name,build,
            np.kron(H@np.diag([1,-1j]) if name=='Y' else H,eye))
    car=call('full_CAR_context','H2 only',lambda:stage1.car_bilinears(4))
    results=[]
    for arm,(J,U) in factors.items():
        model=from_ffsim_number_arrays(one_body_tensor=stage1.corrected_kappa(packet['h'],packet['g']),
            diag_coulomb_mats=J,orbital_rotations=U,constant=float(packet['constant']),z_representation=False,
            frame_ids=[f'F{i}' for i in range(len(J))],group_ids=[f'F{i}' for i in range(len(J))],
            source_identity='fresh-'+arm,input_identity='fresh-molecule',one_body_role='ffsim_corrected_kappa',enable_general_k=True)
        m=call('native_model',arm,lambda:replace(native.prepare_cdf(model),label=arm))
        def full_gens():
            return [stage1.raw_base(stage1.corrected_kappa(packet['h'],packet['g']),packet['constant'],car)]+[
                stage1.raw_frame(a,b,car)[0] for a,b in zip(J,U,strict=True)]
        full=call('full_generators',arm,full_gens)
        for s in (1,4):
            label=f'{arm}_D_s{s}';h=Fraction(1,10*s)
            ideal=call('full_kernel_reference',label,lambda:
                reference.reference_kernels(full,h,ld=len(full)-1)['pf'])
            controlled=np.block([[eye,np.zeros_like(eye)],[np.zeros_like(eye),ideal]])
            report,metrics=segment(label,lambda:builder.ordered(m,h,optimized=True),controlled)
            costs=[];plans=[]
            for row in pf[arm,s]:
                index=row['m'];synth=det.round_synthesis(report,shells,s*(1<<index))
                plan=call('round_accuracy',f'{label}:m{index}',lambda row=row,synth=synth:
                    accuracy.assess_round(target,accuracy.RoundEvidence(m=row['m'],attenuation=1.,
                        pf_operator_error=row['pf_operator_error'],rte_corrected_operator_error=0.,
                        synthesis_signal_error=synth['complex_signal_error_pilot'],finite_law_signal_error=0.,
                        target_id=label,state_model='own_DF_ground_state_oracle',alias_branch_verified=True,
                        iid_per_shot=True,law_model='deterministic')))
                plans.append(plan)
                kernel=dict(target_id=label,policy='native_kernel_repeat_with_fences',mean=metrics,
                    lower=metrics,upper=metrics,simultaneous_interval_qualified=True)
                for axis in ('X','Y'):
                    sh={k:sm['head'][k]+sm[axis][k] for k in accuracy.METRICS}
                    costs.append(connection.compose_serial_cost(kernel,dict(axis=axis,policy='separate_native_readout',
                        mean=sh,lower=sh,upper=sh,simultaneous_interval_qualified=True),m=index,substeps=s,target_id=label))
            energy=call('energy_assessment',label,lambda:accuracy.assess_energy(target,df_errors[arm],plans,
                state_model='own_DF_ground_state_oracle',alias_branch_verified=True))
            total=call('cost_aggregation',label,lambda:accuracy.aggregate_totals(energy,plans,costs,cost_scope=proposal['cost_scope']))
            results.append(dict(id=label,arm=arm,s=s,rounds=plans,energy=energy,cost=total,kernel=report,
                                CA_achieved=None,rigorous_CA_qualification='UNRESOLVED_REFERENCE_ENCLOSURES'))
    comparisons=[]
    for s in (1,4):
        a,b=[next(x['cost'] for x in results if x['arm']==arm and x['s']==s) for arm in ('CDF','XDF')]
        comparisons.append(call('cost_comparison',str(s),lambda:accuracy.compare_totals(a,b)))
    return dict(candidates=results,comparisons=comparisons,shells=shells,
        guarantee='numerical_pilot_only',guaranteed_total_resource_superiority=False)


def run(n,output,fit_python,*,input_factory=None,factor_provider=None):
    """Future authorized science; each run fresh, registered rank ladder only."""
    import numpy as np
    from threadpoolctl import threadpool_limits
    p=specification(n);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    calls=integrated.Calls(output,call_limits(n),sum(call_limits(n).values()))
    start=time.monotonic();cpu=time.process_time()
    plan=integrated.resource_plan(task_count=4,full_dimension=p['sector_dimension'])
    try:
        packet=integrals(n,calls,factory=input_factory)
        if (packet['h'].shape!=(n,n) or packet['g'].shape!=(n,n,n,n)
                or packet['h'].dtype!=np.float64 or packet['g'].dtype!=np.float64):
            raise ValueError('ORIGINAL_SPATIAL_INPUT_SCHEMA')
        stage1.finite_arrays(*packet.values())
        with (output/'input.npz').open('xb') as f:np.savez(f,**packet)
        identity=io.sha(output/'input.npz');io.save(output/'input_identity.json',dict(sha256=identity,conditions=p))
        def obtain(rank,optimize):
            if factor_provider is not None:return factor_provider(packet['g'],rank,optimize)
            name=('CDF_'+str(rank) if optimize else 'XDF_full')+'.npz'
            command=[str(Path(fit_python).absolute()),str(Path(__file__).resolve().parents[2]/'scripts/run_research_c_c1_h2_h6.py'),
                '--fit-worker','--input',str((output/'input.npz').absolute()),'--output',str((output/name).absolute()),
                '--rank',str(rank)]
            if optimize:command.append('--optimize')
            subprocess.run(command,check=True)
            with np.load(output/name,allow_pickle=False) as a:return a['J'],a['U']
        ctx=calls('sector_context',str(n),lambda:SectorCAR(n))
        physical=calls('physical_sector','original h/g',lambda:ctx.physical(packet['h'],packet['g'],packet['constant']))
        rmax=len(p['rank_ladder']);Jx,Ux=calls('explicit_XDF','one Cholesky decomposition',lambda:obtain(rmax,False))
        cap=.00159/8;selected={};gens={};errors={};records={'CDF':[],'XDF':[]}
        xdf_prefix=[]
        for arm in ('XDF','CDF'):
            for rank in p['rank_ladder']:
                if arm=='XDF' and rank>len(Jx):break
                if arm=='XDF':J,U=Jx[:rank],Ux[:rank]
                else:J,U=calls('CDF_fit',str(rank),lambda rank=rank:obtain(rank,True))
                def assess():
                    if arm=='XDF':
                        if not xdf_prefix:
                            xdf_prefix.append(ctx.one(stage1.corrected_kappa(packet['h'],packet['g']))+
                                              float(packet['constant'])*np.eye(ctx.d))
                        xdf_prefix.append(ctx.raw_frame(J[-1],U[-1]))
                    return rank_assessment(ctx,packet,J,U,physical,rank,
                        prebuilt=list(xdf_prefix) if arm=='XDF' else None)
                row,G=calls('rank_assessment',f'{arm}:{rank}',assess)
                records[arm].append(row);io.save(output/f'{arm}_rank_{rank}.json',row)
                if rank_choice([row],cap) is not None:
                    selected[arm]=(J,U);gens[arm]=G;errors[arm]=row['sector_error']+row['numerical_guard']
                    with (output/(arm+'_selected.npz')).open('xb') as f:np.savez(f,J=J,U=U)
                    break
            if arm not in selected:
                io.save(output/'result.json',dict(status='DF_BUDGET_NOT_MET',arm=arm,rank_records=records,mandatory_stop=True))
                return
        representation={arm:calls('representation_gate',arm,lambda arm=arm:
            representation_gate(ctx,packet,*selected[arm],gens[arm])) for arm in selected}
        ground={};eigenpairs={}
        for arm,G in [('physical',[physical]),*gens.items()]:
            def solve(G=G):
                vals,vecs=np.linalg.eigh(sum(G));res=float(np.linalg.norm(sum(G)@vecs[:,0]-vals[0]*vecs[:,0]))
                if res>1e-10 or abs(vals[0]*.1)+math.pi/6>=math.pi:raise ValueError('GROUND_OR_ALIAS_FAIL')
                eigenpairs[arm]=(vals,vecs)
                with (output/('ground_'+arm+'.npz')).open('xb') as f:np.savez(f,energy=vals[0],state=vecs[:,0])
                return dict(energy=float(vals[0]),residual=res)
            ground[arm]=calls('ground_alias',arm,solve)
        pf={}
        with threadpool_limits(limits=1 if plan['workers']>1 else None):
            from concurrent.futures import ThreadPoolExecutor
            def job(key):
                arm,s=key;return key,PF_rounds(gens[arm],s,calls,f'{arm}:s{s}',eigenpair=eigenpairs[arm])
            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
                pf=dict(pool.map(job,[(arm,s) for arm in gens for s in (1,4)]))
        io.save(output/'PF.json',{f'{arm}:s{s}':rows for (arm,s),rows in pf.items()})
        result=dict(status='H6_DF_PF_ONLY_COMPLETE' if n==6 else 'H2_NUMERICAL_PILOT_COMPLETE',
            rank_records=records,ground=ground,DF_errors=errors,representation=representation,
            rank_policy='first passing physical-sector norm+engineering guard',
            CA_achieved=None,rigorous_CA_qualification='UNRESOLVED_REFERENCE_ENCLOSURES',mandatory_stop=True)
        if n==2:result['native']=native_H2(packet,selected,pf,errors,calls,output)
        else:result.update(native='NOT_EXECUTED_OUT_OF_SCOPE',shots=None,total_cost=None,
            feasibility_interpretation='DF reserve passed; saved PF operator errors are a known-part diagnostic only. No synthesis/CA qualification.')
        io.save(output/'result.json',result)
    except BaseException as error:
        calls.stop.set();io.save(output/'failure.json',dict(error=str(error),first_failure=calls.first_failure));raise
    finally:
        io.save(output/'STOP.json',dict(mandatory_stop=True,failed=calls.stop.is_set(),retry=0,resume=0,
            scientific_units=sum(calls.counts.values()),call_counts=dict(calls.counts)))
        io.save(output/'resources.json',dict(plan=plan,wall_seconds=time.monotonic()-start,
            process_CPU_seconds=time.process_time()-cpu,
            parent_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            child_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))


def preflight(root,contract_path,n):
    root=Path(root).resolve();c=io.read(contract_path)
    if c['scope']!=SCOPE or c['status']!='UNAPPROVED_PROPOSAL':raise ValueError('EXTENSION_CONTRACT')
    if c['molecules'][str(n)]!=specification(n) or c['call_limits'][str(n)]!=call_limits(n):
        raise ValueError('CHANGED_EXTENSION_CONDITIONS')
    for path,want in c['SOURCE'].items():
        if io.sha(root/path)!=want:raise ValueError('SOURCE: '+path)
    for name,want in c['evaluation_packages'].items():
        if importlib.metadata.version(name)!=want:raise ValueError('ENVIRONMENT: '+name)
    if c['fit_packages']!=FIT_PACKAGES:raise ValueError('FIT_ENVIRONMENT_CONTRACT')
    return c
