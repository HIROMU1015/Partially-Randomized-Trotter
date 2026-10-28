"""Structural descriptors, not compiled prices. No data access on import.

Only artificial inputs may be evaluated until the separate saved-data contract
is permitted. Reuse the existing atom normalization, run scan and event codec.
"""
from collections import Counter
from fractions import Fraction
import math
from types import SimpleNamespace

VERSION='basis_run_atom_descriptors_v1'
MODELS={'A_switch':('basis_switches',),'B_overlap':('overlap_path',),
        'C_composite':('basis_switches','ZZ_count','same_basis_support_pairs')}


def atoms_from_records(rows):
    from .pr_dictionary_validation.prototype import MixedAtom
    return tuple(MixedAtom(x['atom_id'],x['kind'],Fraction(x['coefficient']),
        tuple(x['origins']),tuple(x['word']),x['basis_key'],x['frame_id']) for x in rows)


def law_features(atoms,r=128):
    from .pr_dictionary_validation.prototype import make_tail
    if type(r)is not int or r<1:raise ValueError('POSITIVE_EVENT_COUNT')
    if any(x.kind!='FRAME_Z' for x in atoms):raise ValueError('IDENTITY_EXTRACTED_FRAME_TAIL_REQUIRED')
    tail=make_tail(atoms,'DESCRIPTOR_ONLY');q={};qz=0.;frames={}
    if tail.deterministic_only_certified:
        return dict(status='NO_RANDOM_TAIL',lambda_R=0.,q_basis={},q_frame={},
            P_same=None,expected_switches=0.,expected_runs=0.,expected_runs_by_basis={},
            expected_Z=0.,expected_ZZ=0.,raw_central_RZ=0.,raw_central_CX=0.)
    for a,p in zip(atoms,tail.components,strict=True):
        q[a.basis_key]=q.get(a.basis_key,0.)+p.probability
        frames[a.frame_id]=frames.get(a.frame_id,0.)+p.probability
        if len(a.support)==1:qz+=p.probability
    same=math.fsum(v*v for v in q.values());runs={k:v+(r-1)*v*(1-v) for k,v in q.items()}
    return dict(status='IDEAL_IID_LAW_DESCRIPTOR',lambda_R=tail.lambda_r,q_basis=q,q_frame=frames,
        P_same=same,expected_switches=(r-1)*(1-same),expected_runs=math.fsum(runs.values()),
        expected_runs_by_basis=runs,expected_Z=r*qz,expected_ZZ=r*(1-qz),
        raw_central_RZ=2*r,raw_central_CX=2*r+2*r*(1-qz),
        compiled_price=None,head_price=None,basis_pair_prices=None)


def overlap(U,V):
    import numpy as np
    U,V=np.asarray(U),np.asarray(V)
    if (U.shape!=V.shape or U.ndim!=2 or not len(U) or U.shape[0]!=U.shape[1]
            or np.iscomplexobj(U) or np.iscomplexobj(V) or not np.isfinite(U).all() or not np.isfinite(V).all()):
        raise ValueError('REAL_ORTHOGONAL_SHAPE')
    n=len(U)
    if max(float(np.max(abs(a.T@a-np.eye(n)))) for a in (U,V))>1e-10:raise ValueError('NONORTHOGONAL_BASIS')
    d=float(n-np.sum(abs(U.T@V)**4))
    if not -1e-8<=d<=n-1+1e-8:raise ValueError('OVERLAP_RANGE')
    return d # retain binary64 roundoff; no threshold/gauge change of U


def geometry(atoms,U):
    """Saved spatial U; verify exact declared spin-lift basis keys, no circuit."""
    import numpy as np,hashlib
    from .c1_stage2_native import _rotation
    U=np.asarray(U);pairs={};frame_basis={};matrices={}
    if U.shape!=(8,4,4) or U.dtype!=np.float64:raise ValueError('FIXED_CDF8_U_SCHEMA')
    for a in atoms:
        if a.frame_id not in frame_basis:
            leaf=a.frame_id.rsplit(':',1)[-1]
            if leaf not in ('F2','F3','F4','F5','F6','F7'):raise ValueError('ORIGINAL_TAIL_FRAME')
            mat=U[int(leaf[1:])];overlap(mat,mat)
            key=hashlib.sha256(repr(_rotation(np.kron(mat,np.eye(2)))).encode()).hexdigest()
            if key!=a.basis_key:raise ValueError('EXACT_SAVED_BASIS_KEY')
            frame_basis[a.frame_id]=key;matrices[a.frame_id]=mat
        if frame_basis[a.frame_id]!=a.basis_key:raise ValueError('FRAME_BASIS_CONFLICT')
    for l,u in matrices.items():
        for k,v in matrices.items():pairs[l,k]=overlap(u,v)
    return pairs


def trajectory_features(payload,atoms,distances,*,event_count=128):
    from .c_pr_aware_df_fixed_car_execution import restore_events
    from trotterlib.rte import _basis_reuse_intervals
    if len(payload['events'])!=event_count:raise ValueError('FIXED128_EVENTS')
    events=restore_events(payload,SimpleNamespace(plan=SimpleNamespace(atoms=atoms)))
    if any(e.taylor_order!=0 or len(e.application_sequence)!=1 or e.product_component_ids
           or e.phase!=1 or e.application_sequence[0].is_identity for e in events):raise ValueError('SIGNED_K0_ROTATIONS_ONLY')
    catalog={x.atom_id:x for x in atoms};chosen=[catalog[e.rotation_component_id] for e in events]
    components=[SimpleNamespace(basis_id=x.basis_key,basis_hash=x.basis_key) for x in chosen]
    intervals=_basis_reuse_intervals(components) # same exact-key grouping as emitter
    starts={x.start:x.stop for x in intervals};runs=[];i=0
    while i<len(chosen):
        j=starts.get(i,i+1);runs.append((chosen[i].basis_key,j-i));i=j
    z=sum(len(x.support)==1 for x in chosen);zz=len(chosen)-z
    same_support=0;negative=0;path=[];frame_switch=0
    for left,right in zip(chosen,chosen[1:]):
        frame_switch+=left.frame_id!=right.frame_id
        same_support+=left.basis_key==right.basis_key and left.support==right.support
        try:path.append(distances[left.frame_id,right.frame_id])
        except KeyError as error:raise ValueError('MISSING_GEOMETRY') from error
    for x in chosen:negative+=x.coefficient<0
    angles=[e.unsigned_rotation_angle for e in events]
    return dict(basis_switches=len(runs)-1,frame_switches=frame_switch,run_count=len(runs),
        run_lengths=[length for _,length in runs],runs_by_basis=dict(Counter(b for b,_ in runs)),
        Z_count=z,ZZ_count=zz,negative_count=negative,same_basis_support_pairs=same_support,
        overlap_path=math.fsum(path),signed_angle_sum=math.fsum(angles),
        first_basis=chosen[0].basis_key,last_basis=chosen[-1].basis_key,
        raw_central_RZ=2*event_count,raw_central_CX=2*event_count+2*zz,
        event_order_preserved=True,head_price=None,compiled_price_prediction=None)


def describe_models(rows,metric):
    """Four fixed index folds WITHIN ONE cohort. No static-DF pseudo-samples.

    Reuse NumPy least squares/SciPy descriptive correlations, not a price engine.
    Missing/rank-deficient columns remain unresolved; no adaptive feature drop.
    """
    import numpy as np
    from scipy.stats import pearsonr,spearmanr
    if len(rows)!=32 or [x['sample'] for x in rows]!=list(range(32)):raise ValueError('FIXED32_ORDER')
    y=np.array([x['metrics'][metric] for x in rows],float)
    if not np.isfinite(y).all() or (y<0).any():raise ValueError('METRIC_VALUES')
    results={}
    for name,features in {'mean_only':(),**MODELS}.items():
        X=np.array([[1.]+[x['features'][k] for k in features] for x in rows],float)
        if not np.isfinite(X).all():raise ValueError('NONFINITE_FEATURE')
        folds=[];pred=np.empty(32);qualified=True
        for fold in range(4):
            test=np.arange(32)%4==fold;train=~test
            beta,_,rank,_=np.linalg.lstsq(X[train],y[train],rcond=1e-12)
            if rank!=X.shape[1]:qualified=False
            pred[test]=X[test]@beta
            folds.append(dict(fold=fold,train=24,test=8,rank=int(rank),coefficients=beta.tolist()))
        value=dict(status='DESCRIPTIVE_HELDOUT' if qualified else 'UNRESOLVED_RANK_DEFICIENT',features=list(features),folds=folds,
            MAE=float(np.mean(abs(pred-y))) if qualified else None,RMSE=float(np.sqrt(np.mean((pred-y)**2))) if qualified else None,
            negative_predictions=int(np.count_nonzero(pred<0)) if qualified else None,
            correlations={},causal_effect=None,new_DF_prediction_validated=False)
        for k in features:
            x=np.array([a['features'][k] for a in rows],float)
            value['correlations'][k]=(dict(status='UNIDENTIFIABLE_CONSTANT',Pearson=None,Spearman=None)
                if np.ptp(x)==0 or np.ptp(y)==0 else dict(status='DESCRIPTIVE_ONLY',Pearson=float(pearsonr(x,y).statistic),Spearman=float(spearmanr(x,y).statistic)))
        results[name]=value
    return results


def qualification(cohorts):
    checks={}
    for name,metrics in cohorts.items():
        models=metrics['rz_count'];score=models['C_composite']['MAE']
        checks[name]=score is not None and models['C_composite']['negative_predictions']==0 and all(v['MAE'] is not None and score<v['MAE'] for k,v in models.items() if k!='C_composite')
    return dict(status='RETAIN_FOR_INDEPENDENT_DF_VALIDATION' if len(checks)==2 and all(checks.values()) else 'SURROGATE_NOT_QUALIFIED',
        primary_checks=checks,new_DF_fit_GO=False,optimizer_objective_adopted=False,
        reason='two DF clusters cannot validate static-U prediction or causal improvement')
