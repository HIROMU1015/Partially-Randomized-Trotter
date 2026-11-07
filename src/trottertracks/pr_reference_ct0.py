"""CT-0: fixed saved keys, phase domains, RPE headroom; NO synthesis/compile.

Only read fixed Git blobs. Reuse C2-B read-validation rules and the original
192-bit interval endpoint helpers, without importing acquisition/optimizer
modules. Every price field below is an existing record field, not a new quote.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from fractions import Fraction as F
from pathlib import Path
import hashlib, io, json, lzma, math, os, platform, subprocess, tarfile
import mpmath as mp
from .track_c_compiler_a_18round import I, upper, lower
from .track_c_h4 import save_json

ROOT=Path(__file__).resolve().parents[2]
BASE='b3ff9d1b4d356ae67e8f55bf1e2cfd0715814cbb'
OUT=ROOT/'artifacts/pr_reference_ct0/2026-11-07'
V3='artifacts/pr_reference_rotation_v3/2026-11-07/'
BP='artifacts/pr_reference_bprime/2026-11-07/'
TR='artifacts/pr_reference_transfer_qualification/2026-11-06/'
CT='artifacts/track_c2_b_pilot/2026-11-06/'
VR='artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/'
INPUTS={}

def data(path):
    raw=subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)
    INPUTS[path]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    return raw

def read(path):
    raw=data(path)
    return json.loads(lzma.decompress(raw) if path.endswith('.xz') else raw)

def archive(path):
    with tarfile.open(fileobj=io.BytesIO(data(path)),mode='r:xz') as t:
        return [json.load(t.extractfile(m)) for m in t if m.isfile()]

def positive(key):
    value=float.fromhex(key)
    assert math.isfinite(value)
    return abs(value).hex()

def validate_record(v,filename=None):
    angle=v['theta_hex'];exponent=v['precision_exponent']
    assert positive(angle)==angle and isinstance(exponent,int)
    spec=f'{angle}:10^{exponent}:pygridsynth2.0.0:strict:seed17'
    assert v['spec']==spec and v['key']==hashlib.sha256(spec.encode()).hexdigest()
    if filename is not None:assert filename=='synthesis/'+v['key']+'.json'
    assert v['status']=='PASS'
    word=v['sequence'];assert set(word)<=set('HTtSXW')
    assert hashlib.sha256(word.encode()).hexdigest()==v['sequence_sha256']
    assert 0<=F(v['strict_error_upper'])<=F(10)**exponent
    # Four legacy records lack the method metadata; retain as unqualified
    # provenance, never fill it in or rerun a guard to repair historical data.
    assert v.get('guard_method') in (None,'exact cyclotomic word matrix + outward interval strict Frobenius')
    assert v['route'] in ('pygridsynth_2.0.0','strict_clifford_fast_path')
    assert v['T']==word.count('T')+word.count('t')
    assert v['Tdagger']==word.count('t') and v['global_W']==word.count('W')
    return angle,exponent

def load_records():
    bundles=read(CT+'BUNDLES.json');raw=data(CT+'raw_synthesis.tar.xz')
    assert hashlib.sha256(raw).hexdigest()==next(x['sha256'] for x in bundles if x['path']=='raw_synthesis.tar.xz')
    records={}
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:xz') as t:
        for m in t:
            assert m.isfile()
            v=json.load(t.extractfile(m));key=validate_record(v,m.name)
            assert key not in records;records[key]=v
    audit=read(V3+'SYNTHESIS_REUSE_AUDIT.json')
    assert len(records)==audit['records_read']==7603
    assert len({a for a,e in records})==audit['distinct_positive_binary_angle_keys']==963
    return records

def key_map(keys,records):
    byangle=defaultdict(list)
    for (a,e),v in records.items():
        byangle[a].append(dict(exponent=e,key=v['key'],strict_error_upper=v['strict_error_upper'],
            sequence_sha256=v['sequence_sha256'],existing_record_T=v['T'],route=v['route'],
            actual_adjoint_available=True,guard_recomputed=False,
            qualification='STRICT_REUSE_CANDIDATE' if v.get('guard_method') else 'GUARD_METHOD_PROVENANCE_UNQUALIFIED'))
    return {k:dict(positive_key=positive(k),status='STRICT_REUSE_CANDIDATE_PRECISION_UNSELECTED' if positive(k) in byangle else 'NUMERIC_KEY_MISSING',
        records=sorted(byangle[positive(k)],key=lambda v:v['exponent'])) for k in sorted(keys)}

def stage_a(records):
    ledger=read(V3+'ANGLE_KEY_MULTIPLICITIES.json.xz');tokens=ledger['tokens'];rows=ledger['rows']
    required={a:set() for a in ('Bprime','reference','registered')};role_rows=[]
    for row in rows:
        total=defaultdict(F)
        for role,counts in row['parts'].items():
            numeric=set();symbolic=set()
            for token,count in counts.items():
                value=F(count);key=tokens[token]['angle_key'];total[key]+=value
                if value:
                    (symbolic if key.startswith('symbolic:') else numeric).add(key)
            role_rows.append(dict(arm=row['arm'],axis=row['axis'],m=row['m'],role=role,
                numeric_keys=sorted(numeric),symbolic_keys=sorted(symbolic),
                signed_correction=role=='D_boundary_fusion_difference'))
        # Combine signed fusion differences before identifying physical key use.
        assert all(v>=0 for v in total.values()),(row['m'],row['arm'])
        required[row['arm']].update(k for k,v in total.items() if v>0)
    numeric={a:{k for k in keys if not k.startswith('symbolic:')} for a,keys in required.items()}
    symbolic={a:sorted(k for k in keys if k.startswith('symbolic:')) for a,keys in required.items()}
    maps={a:key_map(keys,records) for a,keys in numeric.items()}
    summaries={}
    for a,keys in numeric.items():
        pos={positive(k) for k in keys};hit=pos&{k for k,e in records}
        summaries[a]=dict(signed_numeric_keys=len(keys),positive_numeric_keys=len(pos),
            positive_keys_with_record=len(hit),positive_keys_missing=len(pos-hit),
            signed_keys_with_record=sum(positive(k) in hit for k in keys),symbolic_keys=len(symbolic[a]),
            missing_positive_keys=sorted(pos-hit),precision_selected=False,
            adopted_precision_qualified_keys=None)
    precision=[]
    for e in sorted({e for k,e in records}):
        rr={k for k,j in records if j==e}
        qualified={k for (k,j),v in records.items() if j==e and v.get('guard_method')}
        precision.append(dict(exponent=e,stored_records=len(rr),strict_method_documented_records=len(qualified),arms={a:dict(
            positive_keys_with_record=len({positive(k) for k in keys}&rr),
            positive_keys_strict_reuse_candidate=len({positive(k) for k in keys}&qualified),
            positive_keys_missing_at_this_exponent=len({positive(k) for k in keys}-rr)) for a,keys in numeric.items()}))
    # V5 actual inverse SOURCE lowering is a supplementary, distinct saved domain.
    inverse=archive(VR+'stage_A_raw.tar.xz');invkeys=set();inverse_ops=[]
    for batch in inverse:
        for rec in batch['records']:
            keys={float(p[0]).hex() for op,qs,p in rec['native']['lowering'] if op=='rz'}
            invkeys.update(keys);inverse_ops.append(dict(context=batch['context'],name=rec['name'],RZ=len([1 for op,qs,p in rec['native']['lowering'] if op=='rz'])))
    invmap=key_map(invkeys,records);refpos={positive(k) for k in numeric['reference']}
    supplement=dict(saved_inverse_records=len(inverse_ops),signed_numeric_keys=len(invkeys),
        positive_numeric_keys=len({positive(k) for k in invkeys}),
        positive_keys_not_in_V3_reference=sorted({positive(k) for k in invkeys}-refpos),
        mapping=invmap,local_RZ_counts=inverse_ops,
        qualification='saved inverse D only, not all-support vacuum/event phase coverage')
    shared={positive(k) for k in numeric['Bprime']}&{positive(k) for k in numeric['reference']}
    result=dict(status='SAVED_NUMERIC_DOMAIN_ONLY_PRECISION_UNSELECTED',summaries=summaries,
        V3_tokens=len(tokens),V3_rows=len(rows),symbolic_keys=symbolic,
        shared_Bprime_reference_positive_keys=len(shared),
        shared_positive_keys_with_record=len(shared&{k for k,e in records}),
        precision_coverage=precision,mapping=maps,role_rows=role_rows,
        saved_records=len(records),distinct_record_angles=963,core_record_checks_passed=True,
        guard_method_provenance_unqualified=[dict(key=v['key'],angle=v['theta_hex'],exponent=v['precision_exponent']) for v in records.values() if not v.get('guard_method')],
        strict_guards_recomputed=0,synthesis_acquisition_calls=0,T_price_aggregations=0,
        actual_adjoint='unitary adjoint preserves strict norm error; same positive word must be inverted, no fresh synthesis',
        full_support_coverage=False,inverse_D_supplement=supplement)
    save_json(OUT/'KEY_COVERAGE.json',result)
    return result


def rzcount(rec):return sum(op=='rz' for op,_,_ in rec['lowering'])

def count_assets():
    old=archive(TR+'local_catalogue_boundary.tar.xz');new=archive(BP+'Bprime_local_catalogue.tar.xz')
    inv=archive(VR+'stage_A_raw.tar.xz')
    groups={'Bprime':new,'reference':[b for b in old if b['request'][0]=='reference']}
    counts={}
    for arm,batches in groups.items():
        byname=defaultdict(list)
        for batch in batches:
            for rec in batch['records']:byname[rec['name']].append(rzcount(rec))
        counts[arm]={k:max(v) for k,v in byname.items()}
    inverse=defaultdict(list)
    for batch in inv:
        for rec in batch['records']:inverse[rec['name']].append(rzcount(rec['native']))
    return counts,{k:max(v) for k,v in inverse.items()}

def worst_counts(q,r,axis,c,inv):
    def family(arm,f):return c[arm][f]
    b=c['Bprime'];ref=c['reference']
    bd=b['left_all']+b['right_all']+(q-1)*(b['left_no0']+b['right_no0']+b['boundary_control'])
    def lane(n,a):
        return 2*a['block:0:half']+(n-1)*a['block:0:full']+2*n*sum(a[f'block:{j}:half'] for j in (1,2,3))
    if q==1:rd=sum(ref[f'block:{j}:half']+inv[f'block:{j}:half'] for j in range(4))
    else:rd=lane(q//2,ref)+lane(q//2,inv)
    def frames(a):
        bids=[k[6:] for k in a if k.startswith('basis:')]
        return max(a['basis:'+bid]+a['basis_inverse:'+bid] for bid in bids)
    rotb=max(v for k,v in b.items() if k.startswith(('crz:','bprime:')))
    brot=rotb*q*r;rrot=q*r
    bframes=3*q*r*frames(b);rframes=3*q*r*frames(ref)
    # K2: <=2 products/event, each <=2 Z; every possible run retained.
    bprod=4*q*r*family('Bprime','cz');rprod=4*q*r*family('reference','z')
    common=lambda a:2*a['h']+(axis=='Y')*a['sdg']+1
    return dict(Bprime=dict(D=bd,frames=bframes,rotation=brot,products=bprod,
        event_phase=q,scalar_and_wrapper=common(b),CSWAP=0),
        reference=dict(D=rd,frames=rframes,rotation=rrot,products=rprod,event_phase=0,
        scalar_and_wrapper=common(ref),CSWAP=16*ref['cswap']))

def geometry(rho0,dold,B,x):
    d=I(dold)+I(x);ratio=d/I(rho0)
    beta=I(B)-mp.iv.atan2(ratio,mp.iv.sqrt(1-ratio**2))
    return I(rho0)-d,beta

def fixed_shot_budget(rho0,dold,B,N,upper_limit):
    # Bound the continuous extra-disk threshold, retaining the SAVED integer N.
    # No new shot count or precision is selected.
    lo=0.;hi=upper_limit
    for _ in range(90):
        mid=(lo+hi)/2
        if mid==lo or mid==hi:break
        rho,beta=geometry(rho0,dold,B,mid)
        req=4*mp.iv.log(I(1440))/(rho**2*mp.iv.sin(beta)**2)
        if upper(req)<=N:lo=mid
        elif lower(req)>N:hi=mid
        else:break
    assert lo>0
    return dict(lower_inclusive=lo,upper_exclusive_or_equal=hi)

def stage_c():
    mp.iv.prec=192
    bp=read(BP+'ERROR_RPE.json')['rows'];tr=read(TR+'STAGE3_ALL_SUPPORT.json')['rows']
    shots=read(TR+'STAGE4_RPE.json')['rows']
    original=read('artifacts/track_c_guarantee_cn_price/2026-11-06/A_guarantee/verified/ROUNDS.json')
    original=next(x['rows'] for x in original if x['name']=='CN_S2_h0.02_ld3')
    c,inv=count_assets();rows=[]
    for b,t,s,o in zip(bp,tr,shots,original):
        assert b['q']==t['q']==s['q']==o['q'] and b['r']==t['r']==o['r']
        rho0=t['nominal_raw_radius_lower'];B=lower(I('.4')-I(o['PF_phase_upper']))
        counts={axis:worst_counts(b['q'],b['r'],axis,c,inv) for axis in 'XY'}
        arms={}
        for arm,old,N in [('Bprime',b,b['shots_per_axis']),('reference',t['arms']['reference'],s['arms']['reference']['shots_per_axis'])]:
            d=old['total_raw_disk_upper'];rho,beta=geometry(rho0,d,B,0)
            assert lower(rho)>0 and lower(beta)>0
            # Input endpoints are already outward-rounded; serialization differences
            # explain sub-ulp tightening/loss, no old disk is reconstructed or added.
            room=lower(I(rho0)*mp.iv.sin(I(B))-I(d));assert room>0
            cap=fixed_shot_budget(rho0,d,B,N,room)
            epsraw=lower(I(cap['lower_inclusive'])/(2*mp.iv.sqrt(I(2))))
            per_axis={}
            for axis in 'XY':
                parts=counts[axis][arm];L=sum(parts.values());assert L>0
                per_axis[axis]=dict(worst_case_RZ_upper=L,parts=parts,
                    uniform_gate_error_strict_upper=lower(I(epsraw)/L))
            arms[arm]=dict(d_old=d,rho0=rho0,PF_phase_upper=o['PF_phase_upper'],B_lower=B,
                saved_shots_per_axis=N,old_rho_lower=old['raw_radius_lower'],old_beta_lower=old['statistical_phase_lower'],
                recomputed_conservative_rho_lower=lower(rho),recomputed_conservative_beta_lower=lower(beta),
                additional_raw_disk_positive_statistical_room_strict_upper=room,
                additional_raw_disk_retaining_saved_N=cap,
                joint_synthesis_error_retaining_saved_N_inclusive_upper=epsraw,axis=per_axis)
        rows.append(dict(m=b['m'],q=b['q'],r=b['r'],bits=b['bits'],arms=arms))
    eps=min(a['axis'][axis]['uniform_gate_error_strict_upper'] for row in rows for a in row['arms'].values() for axis in 'XY')
    assert eps>0
    result=dict(status='CONDITIONALLY_FEASIBLE',rows=rows,
        common_uniform_gate_error_sufficient_inclusive_upper=eps,
        precision_exponent_or_policy_adopted=False,word_existence_at_required_error_not_verified=True,
        error_budget_existence='FEASIBLE: strictly positive additional disk and fixed-N headroom in every saved round/arm',
        all_support_words_and_price='UNQUALIFIED: unknown vacuum keys, missing numeric keys, no price acquisition',
        operator_to_signal='2sqrt(2) times full joint spectral norm bound, same reset/readout, strict phases; additive raw disk ONCE',
        finite_law_unchanged=True,alpha_per_axis='1/720',total_failure='36/720=0.05',
        new_shot_schedules=0,precision_optimizations=0,T_totals=0,
        count_rule='worst-case finite K2 applications/runs and saved context-max lowerings including actual inverse D; no expectations')
    save_json(OUT/'PRECISION_FEASIBILITY.json',result)
    return result

def stage_b(records):
    tr=read(TR+'STAGE3_ALL_SUPPORT.json')['rows'];constants=read(BP+'V2_INPUTS_CONSTANTS.json')
    assert constants['all_angles_equal'] and constants['tail_basis_RZ_max']==8
    # Counter overapproximation: all r event sign additions, then <=2r negative
    # product additions, all are +float(pi) or +0. No circuit is built.
    event=[]
    for r in sorted({row['r'] for row in tr}):
        phases=[0.]
        for _ in range(3*r):phases.append(phases[-1]+math.pi)
        keys={v.hex() for v in phases[1:]}
        event.append(dict(r=r,max_pi_additions=3*r,positive_nonzero_keys=len(keys),keys=sorted(keys),
            mapping=key_map(keys,records),superset_not_reachability_claim=True,
            differs_from_single_multiplication=[k for k,v in enumerate(phases) if v!=k*math.pi]))
    vacuum=[]
    for row in tr:
        p=row['phase_arithmetic']
        # Saved exact-prefix and forward-error bounds enclose both lane phase
        # accumulations and their final binary addition; keep the source algorithm.
        L=upper(2*I(p['exact_prefix_absolute_upper'])+I(p['naive_accumulation_upper']))
        vacuum.append(dict(m=row['m'],q=row['q'],r=row['r'],absolute_actual_binary_angle_upper=L,
            phase_generation_error_raw_disk_already_in_old=row['arms']['reference']['phase_generation_extra_raw_disk_upper'],
            per_lane_RZ_subtractions_upper=p['RZ_subtractions_per_half_upper'],
            representation='binary64 state + finite SOURCE event/basis/run alphabet, bounded steps, fixed rem_euclid(tau) and sequential subtract; final fl(Fphase+Gphase)',
            exact_numeric_key_list=None,word_price_upper=None))
    result=dict(phase_error_domain_status='FULL_SUPPORT_REPRESENTABLE',
        full_support_strict_word_price_status='UNQUALIFIED',
        overall_stage_B_status='UNQUALIFIED',
        reason='Finite recurrence/angle envelope and uniform synthesis-error condition available; no certified finite word-price envelope over vacuum domain',
        event_phase=event,vacuum_phase=vacuum,
        new_phase_generation_algorithm=False,new_word_generation=0,new_phase_circuits=0,
        finite_cohort_used_as_full_support=False,IEEE_model_inherited=True,
        safety='No phase is replaced by zero/average/parity/modulo key; bounded domain alone is not a T price bound')
    save_json(OUT/'PHASE_SUPPORT.json',result)
    return result

def synthetic_tests():
    from copy import deepcopy
    checks=[]
    assert positive('-0x0.0p+0')==positive('0x0.0p+0');checks.append('signed zero lookup')
    assert positive((-0.125).hex())==(0.125).hex();checks.append('negative angle lookup with sign preserved outside key')
    assert positive((.125+2*math.pi).hex())!=positive((.125).hex());checks.append('no 2pi normalization')
    spec=(.125).hex()+':10^-6:pygridsynth2.0.0:strict:seed17'
    v=dict(theta_hex=(.125).hex(),precision_exponent=-6,spec=spec,key=hashlib.sha256(spec.encode()).hexdigest(),
        status='PASS',sequence='',sequence_sha256=hashlib.sha256(b'').hexdigest(),strict_error_upper='0',
        T=0,Tdagger=0,global_W=0,route='strict_clifford_fast_path',
        guard_method='exact cyclotomic word matrix + outward interval strict Frobenius')
    validate_record(v)
    for name,value in [('spec','projective'),('sequence_sha256','bad'),('status','UNQUALIFIED'),('strict_error_upper','1'),('T',1)]:
        bad=deepcopy(v);bad[name]=value
        try:validate_record(bad)
        except AssertionError:checks.append('reject tampered '+name)
        else:raise AssertionError('validator accepted '+name)
    mp.iv.prec=192
    for x in (0.,.001,.01):
        rho,beta=geometry(.9,.001,.3,x)
        if x:
            assert upper(rho)<last_rho and upper(beta)<last_beta
        last_rho=lower(rho);last_beta=lower(beta)
    checks.append('raw disk monotonic geometry')
    return dict(passed=True,checks=checks,synthetic_schema_not_physical_word=True,
        compiler_calls=0,simulations=0,word_generation=0)

def run():
    if (OUT/'STOP.json').exists():raise RuntimeError('CT-0 mandatory STOP already recorded')
    OUT.mkdir(parents=True,exist_ok=True)
    tests=synthetic_tests();save_json(OUT/'TECHNICAL_TESTS.json',tests)
    records=load_records();a=stage_a(records)
    # Independent budget work does not choose a precision or depend on phase prices.
    c=stage_c();b=stage_b(records)
    save_json(OUT/'SOURCE_INPUTS.json',dict(base_commit=BASE,inputs=INPUTS,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source='src/trottertracks/pr_reference_ct0.py',interval_bits=192,
        reuse=['C2-B load_inputs spec/hash/PASS/guard/T validation pattern','compiler A I/upper/lower',
            'Bprime/transfer saved corrected raw disks','V3 signed finite-law ledger','V5 actual inverse D saved lowerings'],
        new_scientific_conditions=False,old_SOURCE_results_STOPs_changed=False))
    save_json(OUT/'ENVIRONMENT.json',dict(host=platform.node(),python=platform.python_version(),
        executable=os.sys.executable,affinity=len(os.sched_getaffinity(0)),
        memory=Path('/proc/meminfo').read_text(),cpu_pressure=Path('/proc/pressure/cpu').read_text(),
        memory_pressure=Path('/proc/pressure/memory').read_text(),io_pressure=Path('/proc/pressure/io').read_text(),
        shared_settings_changed=False))
    save_json(OUT/'STOP.json',dict(status='MANDATORY_STOP',
        reason='CT-0 finite approved scope complete; full-support vacuum word-price coverage remains UNQUALIFIED',
        stage_A=a['status'],stage_B=b['overall_stage_B_status'],stage_C=c['status'],
        new_synthesis_compile_word_generation_T_price_precision_optimization_total_T=0,
        old_results_STOPs_preserved=True,next_owner='GPT/user: phase-price qualification route and independent finite synthesis scope'))
    print(json.dumps(dict(stage_A=a['summaries'],stage_B=b['overall_stage_B_status'],
        stage_C=c['status'],uniform_gate_error_sufficient_upper=c['common_uniform_gate_error_sufficient_inclusive_upper']),indent=2))
