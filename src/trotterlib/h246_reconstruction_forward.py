"""Separate, one-shot reference restoration and missing H2 signal checks."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('migration_util', Path(__file__).with_name('h246_migration.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def paths(private, system):
    private = Path(private)
    if system == 'H6':
        p = private/'recovered/H6'
        return p/'h6_input_snapshot.npz', p/'snapshot_receipt.json'
    p = private/'generation'/system/'run01'
    return p/'tol_only_snapshot.npz', p/'tol_only_receipt.json'


def prepare(private):
    private = Path(private).resolve()
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    recovery = m.read(private/'recovery_manifest.json')
    provider = next(x for x in recovery if 'provider_SOURCE' in x)
    for system in ('H2', 'H4', 'H6'):
        inp, receipt = paths(private, system)
        files = [inp, receipt, private/'recovery_manifest.json']
        if system == 'H6':
            files.append(private/'recovered/H6/df_receipt.json')
        else:
            comp = private/'generation'/system/'run01/comparison.json'
            result = m.read(comp)
            m.require(all(result['array_hash_comparisons'].values()) and
                result['same_hamiltonian_fingerprint'] and result['same_state_fingerprint'],
                'ALL_OLD_SCIENTIFIC_ARRAY_HASHES_REQUIRED')
            files.append(comp)
        plan = dict(SOURCE=source, system=system, private=str(private), provider=provider,
            input_files={str(p):m.sha(p) for p in files},
            source_files={p:m.sha(ROOT/p) for p in
                ('src/trotterlib/h246_migration.py', 'src/trotterlib/h246_reconstruction_forward.py',
                 'scripts/run_h246_reconstruction_forward.py')},
            call_budget=dict(snapshot_restore=1, occupation_oracle_context=1,
                full_preparation=1, prefix_preparation=1, tail_spectrum=1,
                full_H_reference=1, missing_candidate_signal=2 if system=='H2' else 0,
                shots_recomputed=0, random_sampling=0, compile=0),
            input_policy='OLD_ALL_SCIENTIFIC_ARRAY_HASHES_IDENTICAL_OR_ORIGINAL_H6_SNAPSHOT',
            bias_agreement_tolerance=1e-9, scalar_tolerance=1e-9, retry=False)
        grant = dict(valid=True, approved_by_user=True, retry=False, SOURCE=source,
            grant_id=system+'_REFERENCE_RESTORATION_'+m.digest(plan), plan_digest=m.digest(plan),
            call_budget=plan['call_budget'])
        dest = private/'forward'/system
        m.save(dest/'plan.json', plan)
        m.save(dest/'grant.json', grant)
        m.save(dest/'approval_receipt.json', dict(execution_authorized=True, role='user',
            SOURCE=source, plan_digest=m.digest(plan), scope='Restore missing reference/preparation and necessary H2 signals using fixed original methods; frozen shots retained only after comparison.'))
    return dict(SOURCE=source, reference_jobs=3, missing_H2_signals=2, shots_recomputed=0)


def verify(plan, grant):
    m.require(grant.get('approved_by_user') is True and grant.get('valid') is True and
        grant.get('retry') is False and grant['plan_digest']==m.digest(plan) and
        grant['call_budget']==plan['call_budget'], 'FORWARD_EXPLICIT_FIXED_GRANT')
    for p, h in plan['input_files'].items():
        m.require(m.sha(p)==h, 'RESTORED_INPUT_CHANGED:'+p)
    for p, h in plan['source_files'].items():
        b=subprocess.check_output(['git','show',plan['SOURCE']+':'+p],cwd=ROOT)
        import hashlib
        m.require(m.sha(ROOT/p)==h==hashlib.sha256(b).hexdigest(),'FORWARD_SOURCE_CHANGED:'+p)


def execute(private, system):
    import sys
    private=Path(private).resolve();base=private/'forward'/system
    plan,grant=m.read(base/'plan.json'),m.read(base/'grant.json')
    verify(plan,grant);m.claim(base/'CONSUMED.json',grant)
    out=base/'run01';out.mkdir(exist_ok=False)
    for kind,n in plan['call_budget'].items():
        if n:m.save(out/'ledger'/f'{kind}.json',dict(state='RESERVED_NONREFUNDABLE',count=n,SOURCE=plan['SOURCE'],grant_id=grant['grant_id']))
    sys.path.insert(0,str(Path(plan['provider']['provider_root'])/'src'))
    sys.path.insert(0,str(ROOT/'scripts'))
    try:
        import numpy as np
        from trotterlib.df_hamiltonian import DFHamiltonian, PhysicalSector
        from trotterlib.df_partial_s2 import df_hamiltonian_hash
        from trotterlib.pr2_s0_s1_validation import _state_hash
        from h46_new2_native import OccupationOracle,prepare as prepare_pf
        from h46_new2_core import matrix_only_tail_coefficients,spectrum
        inp,receipt=paths(private,system);r=m.read(receipt)
        if system=='H6':
            from trottertracks.resource_applicability.ax2b_h6_saved_completion_loader_v1 import load_h6_snapshot
            ham,sector,full,state,md,_=load_h6_snapshot(inp,r,m.read(private/'recovered/H6/df_receipt.json'))
        else:
            with np.load(inp,allow_pickle=False) as z:a={k:z[k] for k in z.files}
            ham=DFHamiltonian(float(a['constant']),a['one_body'],a['lambdas'],tuple(a['g_matrices']),r['hamiltonian_metadata'])
            n=int(system[1:]);sector=PhysicalSector.spin_sector(n_qubits=2*n,nelec_alpha=n//2,nelec_beta=n//2)
            full,state=a['state_vector'],a['sector_state_vector']
        old=m.read(ROOT/'artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/input_provenance.json')
        binding=next(x['binding'] for x in old['input_metadata'] if x['system']==system)
        m.require(ham.n_blocks==binding['rank'] and df_hamiltonian_hash(ham)==binding['hamiltonian']
            and _state_hash(full,state)==binding['state'],'EXACT_OLD_HAMILTONIAN_AND_STATE')
        prefix={'H2':1,'H4':3,'H6':5}[system]
        oracle=OccupationOracle(ham,full,prefix=ham.n_blocks)
        fullprep,cut=prepare_pf(ham,ham.n_blocks),prepare_pf(ham,prefix)
        qstate=np.zeros(1<<ham.n_qubits,complex);qstate[oracle.qindices]=oracle.initial
        matrices=tuple((s[1]*s[0])@s[2] for s in oracle.specs)
        scalar,lam=matrix_only_tail_coefficients(ham,prefix)
        m.require(abs(cut.extracted_identity_coefficient-scalar)<1e-9 and abs(cut.exact_rte_lambda_r-lam)<1e-9,'TAIL_SCALAR_LAMBDA')
        tail=sum(matrices[prefix+1:],np.zeros_like(matrices[0]))-scalar*np.eye(len(oracle.initial))
        tail_spec=spectrum(tail)
        if system=='H2':
            from scipy.linalg import expm
            reference=expm(-.8j*(sum(matrices)+ham.constant*np.eye(len(oracle.initial))))@oracle.initial
        else:
            spec_h=spectrum(sum(matrices))
            reference=spec_h[1]@(np.exp(-.8j*spec_h[0])*(spec_h[2]@oracle.initial))
            reference*=np.exp(-.8j*ham.constant)
        with (out/'reference.npz').open('xb') as f:np.savez(f,reference=reference,qstate=qstate)
        m.save(out/'receipt.json',dict(status='PASS',system=system,SOURCE=plan['SOURCE'],
            reference_sha256=m.sha(out/'reference.npz'),snapshot=m.sha(inp),
            hamiltonian_hash=binding['hamiltonian'],state_hash=binding['state'],
            full_preparation=fullprep.preparation_hash,cut_preparation=cut.preparation_hash,
            scalar=ham.constant,tail_identity=scalar,lambda_R=lam,
            reference_policy='ORIGINAL_FULL_H_DF_OCCUPATION_REFERENCE_CONSTANT_ONCE',
            reference_norm=float(np.linalg.norm(reference)),missing_reference_regenerated=True))
        fixed=m.read(ROOT/'references/h246_confirmation_design_v1.json')
        comp=next(x for x in fixed['comparisons'] if x['id']==system+'_PRIMARY')
        ctx=SimpleNamespace(np=np,ham=ham,oracle=oracle,qstate=qstate,fullprep=fullprep,cut=cut,
            reference=reference,scalar=ham.constant,tail_identity=scalar,lambda_R=lam,tail_spec=tail_spec)
        rows=[]
        saved46=m.read(ROOT/'artifacts/resource_applicability/track_a_h46_m5_signal_run02_results/2026-10-20/signal_records.json')
        for group in comp['groups']:
            candidate=group['candidate'];shot=group['cost_row']['shot']
            if system=='H2':
                from h46_m5_native import signal
                sig=signal(ctx,dict(candidate,policy=candidate['r_policy']))
                row=dict(candidate=candidate,signal=sig,preparation_hash=cut.preparation_hash,origin='REGENERATED_MISSING_H2_SIGNAL')
            else:
                row=next(x for x in saved46['records'] if x['candidate']['id']==candidate['id'])
                sig=row['signal']
                m.require(row['preparation_hash']==cut.preparation_hash,'ORIGINAL_PREPARATION_HASH')
                oldref=complex(sig['reference']['real'],sig['reference']['imag'])
                m.require(abs(np.vdot(oracle.initial,reference)-oldref)<1e-9,'REGENERATED_REFERENCE_MATCHES_SAVED_SIGNAL')
            bias=sig['total_signed_bias']
            errors={axis:abs(bias[axis]-shot['saved_signed_bias'][axis]) for axis in ('real','imag')}
            errors['log_B']=abs(sig['log_B']-shot['log_B'])
            m.require(max(errors.values())<1e-9,'FIXED_SAVED_BIAS_NORMALIZATION_AGREEMENT')
            rows.append(dict(**row,migration_agreement_errors=errors,frozen_shots=shot,
                new_shot_calculation=0,exact_old_hamiltonian_and_state_arrays=True))
        m.save(out/'signals.json',dict(records=rows,system=system,SOURCE=plan['SOURCE'],
            status='FIXED_SIGNALS_AND_FROZEN_SHOT_APPLICABILITY_PASS',new_signal_count=2 if system=='H2' else 0,
            shot_recalculation=0,reference_sha256=m.sha(out/'reference.npz')))
        verify(plan,grant)
        m.save(out/'completion.json',dict(status='REFERENCE_AND_FROZEN_SHOT_APPLICABILITY_PASS',
            SOURCE=plan['SOURCE'],grant_id=grant['grant_id'],call_budget=plan['call_budget'],
            signals_sha256=m.sha(out/'signals.json'),reference_sha256=m.sha(out/'reference.npz'),retry=False))
        return dict(system=system,status='REFERENCE_AND_FROZEN_SHOT_APPLICABILITY_PASS',
            reference_sha256=m.sha(out/'reference.npz'),preparation_hash=cut.preparation_hash,
            maximum_old_bias_or_log_B_difference=max(max(x['migration_agreement_errors'].values()) for x in rows))
    except BaseException as exc:
        import traceback
        m.save(out/'STOP.json',dict(status='STOP',reason=str(exc),traceback=traceback.format_exc(),retry=False,grant_id=grant['grant_id']))
        raise
