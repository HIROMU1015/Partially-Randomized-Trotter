"""Update064 metadata adapter for the fixed-commit A price-statistics engine.

No numerical work or PRNG draws in passive preflight. The new prospective
cohort, one-sample approximate t inference and execution need separate permission.
"""
from pathlib import Path
import importlib.metadata
import math
import resource
import subprocess
import time

from . import c_r32_price_statistics as shared
from . import c_update064_r32_downstream as downstream

io, ret = shared.io, shared.ret
BASE = 'cca6b487c8e1ed3362f8825e95c937ebbcc87755'
A_SOURCE = '62fa4c0144b9e36f9d96b7dff67628051946632d'
PEER = downstream.PEER
CANDIDATE = downstream.CANDIDATE
INPUT = downstream.INPUT
SCOPE = 'C_UPDATE064_R32_PROSPECTIVE_PRICE_V1'
VERSION = 'shared_A_engine_saved064_native256_approximate_t_v1'
ENTRY = 'scripts/run_research_c_update064_r32_price_statistics.py'
FAMILY = 'artifacts/research_c_update064_r32_price_statistics/2026-11-03'
FRESH = FAMILY + '/run01'
N = 256
BATCHES = N // 32
STATISTICS = dict(shared.STATISTICS, samples=N, precision_halfwidth_fraction_of_D=.01,
    skewness_abs_threshold=2., maximum_delete_one_shift_fraction_of_halfwidth=.25)
POLICY = downstream.POLICY
A_LIMITS = dict(input_load=1, selected_load=1, reference_load=1, candidate_binding=1,
    sector_context=1, native_model=1, general_factor_gate=9, ground_load=1,
    ground_gate=1, law_prepare=1, law_gate=1, normalization_gate=1)
LIMITS = {**A_LIMITS, 'saved_r32_gate': 1, 'seed_disjoint_gate': 1,
    **{'native.'+k: v for k, v in ret.COMMON_NATIVE.items()},
    **{f'batch{i:02d}.'+k: v for i in range(BATCHES) for k, v in ret.SAMPLE_NATIVE.items()},
    'accounting_binding': 1, 'native_rows_validation': 1, 'primary_statistics': 1,
    'secondary_summary': 1, 'result_summary': 1}
MAX_CALLS = sum(LIMITS.values())
ACCOUNTING = dict(W_ideal=13381920, shell_RZ_ideal=52528, D_RZ_ideal=105038112319,
    ideal_shots=15008, D_ideal_shots=13706)
RECORDS = {'result', 'STOP', 'SOURCE', 'contract', 'ground_report', 'law', 'shells'}
OLD_SEEDS = dict(retuning='65c8aadbbbc2dd0179ef2465ce02c9fa546a0e942f3caa4aaf3920dd44eaeef7',
    integrated='7eda4aa319b9b861cfa2b5e8c25e3381b4670c275bdad2bdb073335df550aff5',
    update064='ab9eb265211c83a3836b2322b00b58249e1492b4bd28dd2748e5fe09b4e56434',
    A_prospective='f6f0be47a453349209237694826637cd289b57a00f7184d2cee2d6241172314b')


def sampling(batch):
    if type(batch) is not int or not 0 <= batch < BATCHES: raise ValueError('REGISTERED_BATCH_ONLY')
    return dict(master=2026110303, cohort=f'update064_r32_prospective_v1_batch{batch:02d}',
        axis='UPDATE064_R32_PRICE_HOLDOUT_ONLY', samples=32, events_per_sample=32,
        PRNG='numpy.random.PCG64', paired=False, adaptive=False)


def seed_manifest(digest, old):
    """Only deterministic hash keys; no trajectory, PRNG draw or old raw access."""
    def seeds(policy, source, candidate):
        master = (policy['master'], policy['cohort'], source)
        return [ret.perf.b.connection.trajectory_seed(master, 'proposed', candidate,
            -1, policy['axis'], i, 0) for i in range(32)]
    old_seeds = []
    for r in ret.NEW_R: old_seeds.extend(seeds(ret.sampling(r), old['retuning'], ret.CANDIDATE))
    old_seeds.extend(seeds(ret.perf.SAMPLING, old['integrated'], ret.CANDIDATE))
    old_seeds.extend(seeds(downstream.SAMPLING, old['update064'], CANDIDATE))
    old_cohorts = {ret.sampling(r)['cohort'] for r in ret.NEW_R}
    old_cohorts.update([ret.perf.SAMPLING['cohort'], downstream.SAMPLING['cohort']])
    for b in range(shared.BATCHES):
        policy = shared.sampling(b)
        old_seeds.extend(seeds(policy, old['A_prospective'], ret.CANDIDATE))
        old_cohorts.add(policy['cohort'])
    policies = [sampling(b) for b in range(BATCHES)]
    if (len({p['cohort'] for p in policies}) != BATCHES
            or {p['cohort'] for p in policies} & old_cohorts):
        raise ValueError('COHORT_ID_OVERLAP')
    rows = [dict(id=f'batch{b:02d}/sample{i:02d}', global_id=32*b+i, seed=seed)
        for b, policy in enumerate(policies) for i, seed in enumerate(seeds(policy, digest, CANDIDATE))]
    new = [x['seed'] for x in rows]
    if len(set(new)) != N or set(new) & set(old_seeds): raise ValueError('SEED_COLLISION_OR_OLD_COHORT_OVERLAP')
    return dict(status='PASS', rows=rows, old_keys_checked=len(old_seeds),
        old_key_groups=dict(update005_exploratory=128,update064_exploratory=32,A_fixed_prospective=shared.N),
        seed_bits=64, old_trajectories_decoded=0, PRNG_draws=0,
        interpretation='disjoint fixed hash-seed keys;not a mathematical independence/ideal-law certificate')


def saved_gate(root, c):
    """Only saved JSON verdicts; never old native32 sample prices or new inference."""
    root = Path(root); s = {k: io.read(root/v) for k, v in c['records'].items()}
    result, stop, contract = (s[k] for k in ('result', 'STOP', 'contract'))
    pr, d = result['PR'], result['D']
    if (result['status'] != 'UPDATE064_PR_AND_SAME_JU_D_COMPLETE'
            or result['candidate'] != CANDIDATE or pr['candidate'] != CANDIDATE
            or d['candidate'] != CANDIDATE or pr['r'] != 32 or not pr['native_admitted']
            or stop['failed'] or stop['scientific_units'] != 617 or stop['retry'] or stop['resume']
            or not stop['mandatory_stop'] or stop['call_counts'] != downstream.LIMITS):
        raise ValueError('SAVED_UPDATE064_IDENTITY_STOP')
    if (contract['policy'] != POLICY or contract['target'] != ret.m4.TARGET
            or contract['compiler'] != ret.COMPILER or contract['sampling'] != downstream.SAMPLING
            or contract['source_manifest_sha256'] != io.sha(root/c['records']['SOURCE'])
            or contract['saved_inputs'][c['input']] != INPUT
            or contract['peer_inputs'][contract['peer_paths']['candidate']] != CANDIDATE):
        raise ValueError('SAVED_SOURCE_POLICY')
    df = d['saved_DF']
    if (df['candidate_sha256'] != CANDIDATE or df['actual_rank'] != 8 or not df['budget_PASS']
            or df['budget_Ha'] != .00019875 or df['kappa'] != 'original_g_corrected'
            or df['scalar'] != 'nuclear_once'
            or not 0 <= df['sector_error']+df['numerical_guard'] <= .00019875):
        raise ValueError('SAVED_DF_QUALIFICATION')
    if (s['ground_report']['status'] != 'PASS' or s['ground_report']['alias_margin'] <= 0
            or s['ground_report']['candidate_sha256'] != CANDIDATE
            or s['law']['status'] != 'PASS' or s['law']['atoms'] != 216
            or pr['calibration']['samples'] != 32
            or pr['lanes']['evidence']['conditional_totals']['totals'] is not None):
        raise ValueError('SAVED_GROUND_LAW_LANE')
    for lane in ('ideal_exact_native', 'assumed_raw_signal_budget'):
        v = pr['lanes'][lane]
        if (not v['energy']['same_accuracy'] or [x['m'] for x in v['rounds']] != list(range(13))
                or any(x['target_id'] != 'update064:r32:'+CANDIDATE for x in v['rounds'])
                or not d['ideal_exact_native']['energy']['same_accuracy']):
            raise ValueError('SAVED_RPE_TARGET')
    v = pr['lanes']['ideal_exact_native']
    if (v['total_shots'] != ACCOUNTING['ideal_shots']
            or v['weighted_SE']['kernel_reuse_weight'] != ACCOUNTING['W_ideal']
            or d['ideal_exact_native']['cost']['totals']['rz_count'] != ACCOUNTING['D_RZ_ideal']
            or d['ideal_exact_native']['cost']['shots'] != ACCOUNTING['D_ideal_shots']):
        raise ValueError('SAVED_ACCOUNTING_DENOMINATOR')
    return pr, dict(D=d, shell_reports=s['shells'], law=s['law'], physical=df, ground_report=s['ground_report'])


def preflight(root, path, peer):
    root, peer = Path(root).resolve(), Path(peer).resolve(); c = io.read(path)
    fixed = dict(base_commit=BASE, A_technical_commit=A_SOURCE, candidate_commit=PEER,
        scope=SCOPE, implementation=VERSION, entry=ENTRY, policy=POLICY,
        statistics=STATISTICS, fixed_accounting=ACCOUNTING, sampling=[sampling(b) for b in range(BATCHES)],
        target=ret.m4.TARGET, compiler=ret.COMPILER, allowances=ret.m3.ALLOWANCES,
        call_limits=LIMITS, maximum_logical_units=MAX_CALLS, fresh_output=FRESH,
        retry=0, resume=0, new_fit=0, optimizer=0, new_PF_RTE=0, new_shots=0,
        new_ground_eigensolve=0, new_DF_error=0, mandatory_stop=True)
    for key, want in fixed.items():
        if c[key] != want: raise ValueError('FIXED_UPDATE064_PRICE:'+key)
    if set(c['records']) != RECORDS: raise ValueError('RECORD_SCHEMA')
    if set(c['peer_paths']) != {'candidate','reference','basis'}: raise ValueError('PEER_ROLE_SCHEMA')
    if c['old_seed_policy'] != OLD_SEEDS: raise ValueError('OLD_SEED_PROVENANCE')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=peer,text=True).strip() != PEER:
        raise ValueError('FIXED_PEER_COMMIT')
    needed = {c['input'], c['ground'], c['dependency_lock'], *c['records'].values()}
    needed.update(p for row in c['shells'].values() for p in row.values())
    if (not needed <= c['saved_inputs'].keys() or c['saved_inputs'][c['input']] != INPUT
            or not set(c['peer_paths'].values()) <= c['peer_inputs'].keys()
            or c['peer_inputs'][c['peer_paths']['candidate']] != CANDIDATE):
        raise ValueError('PINNED_PRICE_INPUT')
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']: raise ValueError('SOURCE_MANIFEST')
    files = io.read(root/c['source_manifest'])['files']
    if ENTRY not in files: raise ValueError('UNPINNED_ENTRY')
    for folder, group in ((root,files),(root,c['saved_inputs']),(peer,c['peer_inputs'])):
        for name, digest in group.items():
            p = (folder/name).resolve()
            if not p.is_relative_to(folder) or io.sha(p) != digest: raise ValueError('SOURCE_INPUT:'+name)
    for name, version in c['packages'].items():
        if importlib.metadata.version(name) != version: raise ValueError('ENVIRONMENT:'+name)
    from .c1_h4_native_fair import GAUSSIAN, PHASE
    if c['gaussian'] != GAUSSIAN or c['compiler_phase'] != PHASE: raise ValueError('NATIVE_PRECISION')
    saved_gate(root,c)
    seed_manifest(c['source_manifest_sha256'], c['old_seed_policy'])
    if (root/FRESH).exists(): raise ValueError('FRESH_NO_RETRY_RESUME')
    return c


def prepare_native(root, peer, c, out, call):
    import numpy as np
    from . import c1_molecule_extension as molecule
    from .c_rc2_pr_stage_b import distribution_gate
    b, loader = ret.perf.b, ret.perf.loader
    p = call('input_load','original h/g/scalar',lambda:loader.load(root/c['input'],('h','g','constant')))
    d = call('selected_load','fixed update064',lambda:loader.load(peer/c['peer_paths']['candidate'],('J','U')))
    def refs():
        v=loader.load(peer/c['peer_paths']['reference'],('J','U','generators','H'))
        basis=loader.load(peer/c['peer_paths']['basis'],('sector_basis_indices',))['sector_basis_indices']
        v['full_input_indices']=b.stage1.bit_reverse_indices(8)[basis]
        return v
    sector=call('reference_load','same064 CAR factors/basis',refs)
    def binding():
        downstream.strong.m2.validate_arrays(p,d,'update064')
        if not all(np.array_equal(d[k],sector[k]) for k in ('J','U')): raise ValueError('MIXED_JU')
        from .c1_stage1 import sector_indices
        if (sector['generators'].shape!=(9,36,36) or sector['H'].shape!=(36,36)
                or not all(np.isfinite(sector[k]).all() for k in ('generators','H'))
                or not np.array_equal(np.sort(sector['full_input_indices']),sector_indices(4,2,2))):
            raise ValueError('SECTOR_SCHEMA')
        return b.component_check(sum(sector['generators']),sector['H'],1e-10,label='same064 sector sum')
    io.save(out/'identity.json',call('candidate_binding','saved newcandidate factors only',binding))
    ctx=call('sector_context','SectorCAR4',lambda:molecule.SectorCAR(4))
    arm=call('native_model','fixed JU/LD2/generalK',lambda:b.prepare_arm(**p,**d,role='proposed',source_identity=CANDIDATE,input_identity=INPUT,ld=2))
    for i in range(9):call('general_factor_gate',str(i),lambda i=i:b.component_check(b.candidate_factor(ctx,arm,i),sector['generators'][i],1e-10,label='same064 generalK/CAR'))
    ground=call('ground_load','saved own064 ground;no solve',lambda:loader.load(root/c['ground'],('energy','sector_state')))
    report=io.read(root/c['records']['ground_report'])
    def gate():
        psi=ground['sector_state'];e=float(ground['energy'])
        if psi.shape!=(36,) or not np.isfinite(psi).all() or not math.isfinite(e): raise ValueError('GROUND_SCHEMA')
        residual=float(np.linalg.norm(sector['H']@psi-e*psi));norm=float(abs(np.vdot(psi,psi)-1))
        if residual>1e-10 or norm>1e-12 or abs(e-report['energy_Ha'])>1e-12 or not math.isfinite(residual+norm): raise ValueError('GROUND_CORRECTNESS')
        return dict(status='PASS',candidate=CANDIDATE,residual=residual,normalization=norm,alias_margin=report['alias_margin'],new_eigensolve=0)
    io.save(out/'ground.json',call('ground_gate','candidate-bound saved ground/alias',gate))
    tail,config,dist=call('law_prepare','same064 r32/K0',lambda:b.configure(arm,ret.Fraction(1,10),32,0))
    saved_law=io.read(root/c['records']['law'])
    def law():
        v=b.tail_gate(ctx,arm,sector['generators'],tail)
        distribution_gate(tail,config,dist,expected_steps=32)
        if any(v[k]!=saved_law[k] for k in ('lambda_R','identity','atoms')): raise ValueError('LAW_CANDIDATE_IDENTITY')
        return v
    io.save(out/'law.json',call('law_gate','same signed/identity law',law))
    io.save(out/'normalization.json',call('normalization_gate','same K0',lambda:ret.normalization(tail,config,32)))
    return dict(arm=arm,p=p,tail=tail,config=config,dist=dist,sector=sector,ground=ground['sector_state'])


def execute(root, peer, c, out, call, *, native_function=None):
    study=dict(samples=N,statistics=STATISTICS,saved_gate=saved_gate,
        seed_manifest=seed_manifest,sampling=sampling,candidate=CANDIDATE,affine_bindings=affine_bindings)
    return shared.execute(root,c,out,call,study=study,
        prepare_function=lambda root,c,out,call:prepare_native(root,peer,c,out,call),native_function=native_function)


def affine_bindings(v, saved):
    bindings = shared.affine_bindings(v,saved)
    a = bindings['ideal_exact_native']
    if (a['W'] != ACCOUNTING['W_ideal'] or a['shell']['rz_count'] != ACCOUNTING['shell_RZ_ideal']
            or a['D']['rz_count'] != ACCOUNTING['D_RZ_ideal'] or a['shots'] != ACCOUNTING['ideal_shots']):
        raise ValueError('UPDATE064_AFFINE_BINDING')
    return bindings


def run(root, path, peer, *, execution_opt_in=False):
    # Explicit opt-in prevents accidental execution. It is not a proof of user
    # permission: the agent must first have the separate scientific chat approval.
    if not execution_opt_in: raise ValueError('EXPLICIT_EXECUTION_OPT_IN_REQUIRED')
    root,peer=Path(root).resolve(),Path(peer).resolve();c=preflight(root,path,peer);out=root/FRESH
    out.mkdir(parents=True);call=ret.m4.integrated.Calls(out,LIMITS,MAX_CALLS)
    tick,cpu=time.monotonic(),time.process_time()
    try:
        io.save(out/'contract.json',c);io.save(out/'SOURCE.json',io.read(root/c['source_manifest']))
        value=execute(root,peer,c,out,call);io.save(out/'result.json',value);return value
    except BaseException as error:
        call.stop.set();io.save(out/'failure.json',dict(error=repr(error),first_failure=call.first_failure));raise
    finally:
        io.save(out/'STOP.json',dict(failed=call.stop.is_set(),scientific_units=sum(call.counts.values()),
            maximum_logical_units=MAX_CALLS,call_counts=dict(call.counts),first_failure=call.first_failure,
            retry=0,resume=0,mandatory_stop=True))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
