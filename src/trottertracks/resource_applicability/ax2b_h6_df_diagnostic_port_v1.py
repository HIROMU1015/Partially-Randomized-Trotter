"""Authorized future diagnostic only; tests inject arrays/decomposer, never real inputs."""
from __future__ import annotations
import hashlib
import contextlib
import io
import os
from pathlib import Path
import numpy as np
from .ax2b_limits import CallBudget, output_size
from .ax2b_h6_df_diagnostic_contract_v1 import INPUT_SHA, plan

def array_record(a):
    a=np.asarray(a)
    return {'shape':list(a.shape),'dtype':a.dtype.str,
            'sha256':hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}

def write_npz(writer, name, arrays, cap):
    if Path(name).name!=name or not name.endswith('.npz'):raise ValueError('RAW_NAME')
    arrays={k:np.asarray(v) for k,v in arrays.items()}
    if any(a.dtype.str not in ('<f8','<c16') for a in arrays.values()):raise ValueError('RAW_DTYPE')
    size=sum(a.nbytes+65536 for a in arrays.values())
    if size>cap or output_size(writer.output)+size>writer.byte_cap-writer.reserve:raise RuntimeError('RAW_OUTPUT_CAP')
    pending=writer.output/('.pending_'+name)
    with pending.open('xb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    if pending.stat().st_size>size or output_size(writer.output)>writer.byte_cap-writer.reserve:raise RuntimeError('RAW_WRITTEN_CAP')
    os.link(pending,writer.output/name);pending.unlink()
    return {'file':name,'sha256':hashlib.sha256((writer.output/name).read_bytes()).hexdigest(),
            'bytes':(writer.output/name).stat().st_size,'arrays':{k:array_record(a) for k,a in arrays.items()},
            'array_hash_order':'logical C-order bytes, original dtype; NPZ stores values without coefficient adjustment'}

def antisymmetrize(t):
    return (t-t.swapaxes(0,1)-t.swapaxes(2,3)+t.swapaxes(0,1).swapaxes(2,3))/4

def normal_order(g):
    """(sum g_pq a†_p a_q)^2: g@g one-body and -g_pr*g_qs quartic."""
    return g@g, -np.einsum('pr,qs->pqrs',g,g)

def fnorm(a):return float(np.linalg.norm(np.asarray(a).ravel()))

def describe(one, two, weights, gs, correction, truncation, *, budget):
    """All fragments retained; projected matrices are hypothetical diagnostic values."""
    n=one.shape[0];rank=len(weights);tol=plan()['hermitization_tolerance']
    if n!=12 or one.shape!=(n,n) or two.shape!=(n,)*4 or gs.shape!=(rank,n,n) or correction.shape!=(n,n):raise ValueError('DIAGNOSTIC_LAYOUT')
    if not 1<=rank<=36:raise ValueError('DIAGNOSTIC_FRAGMENT_CAP')
    if not all(np.isfinite(a).all() for a in (one,two,weights,gs,correction,truncation)) or np.any(np.imag(weights)!=0):raise ValueError('DIAGNOSTIC_NONFINITE_OR_COMPLEX_WEIGHT')
    projected=(gs+gs.conj().transpose(0,2,1))/2
    corrected=one+correction;corrected_projected=(corrected+corrected.conj().T)/2
    chem=two.transpose(0,3,1,2)[::2,::2,1::2,1::2]
    interaction=chem.reshape(36,36)
    if np.asarray(truncation).shape!=() or float(truncation)<0:raise ValueError('DIAGNOSTIC_TRUNCATION')
    raw_c=np.array(correction,dtype=np.complex128);post_c=raw_c.copy()
    raw_t=np.zeros_like(two);post_t=np.zeros_like(two);rows=[]
    for i,(weight,g,h) in enumerate(zip(weights,gs,projected)):
        budget.take('fragment_diagnostics');lam=float(np.real(weight))
        c,t=normal_order(g);hc,ht=normal_order(h)
        ac,ah=antisymmetrize(t),antisymmetrize(ht)
        raw_c+=lam*c;post_c+=lam*hc;raw_t+=lam*t;post_t+=lam*ht
        norm=fnorm(g);dev=fnorm(h-g);v=g[::2,::2].reshape(36)
        rows.append({'index':i,'lambda':lam,'abs_lambda':abs(lam),'raw':array_record(g),'hypothetical_post':array_record(h),
            'raw_frobenius_norm':norm,'hermiticity_defect_frobenius':fnorm(g-g.conj().T),
            'hermitization_change_frobenius':dev,'relative_change':dev/norm if norm else None,
            'exceeds_original_hermitization_tolerance':dev>tol,
            'returned_weight_proxy_abs_lambda_l1_squared':abs(lam)*float(np.abs(g).sum())**2,
            'cross_spin_frobenius':fnorm(g[::2,1::2])+fnorm(g[1::2,::2]),
            'alpha_beta_block_difference_frobenius':fnorm(g[::2,::2]-g[1::2,1::2]),
            'source_eigenpair_residual_frobenius':fnorm(interaction@v-lam*v),
            'weighted_raw_one_body_coefficient_frobenius':abs(lam)*fnorm(c),
            'weighted_raw_antisym_two_body_coefficient_frobenius':abs(lam)*fnorm(ac),
            'weighted_projection_one_body_difference_frobenius':abs(lam)*fnorm(c-hc),
            'weighted_projection_antisym_two_body_difference_frobenius':abs(lam)*fnorm(ac-ah),
            'weighted_one_body_hermiticity_defect_frobenius':abs(lam)*fnorm(c-c.conj().T),
            'weighted_two_body_hermiticity_defect_frobenius':abs(lam)*fnorm(ac-ac.conj().transpose(3,2,1,0))})
    expected_corr=np.zeros_like(correction);spatial_corr=-np.einsum('pqqs->ps',chem)
    expected_corr[::2,::2]=spatial_corr;expected_corr[1::2,1::2]=spatial_corr
    target_a=antisymmetrize(two);raw_a=antisymmetrize(raw_t);post_a=antisymmetrize(post_t)
    spatial_g=gs[:,::2,::2]
    reconstructed_chem=np.einsum('l,lpq,lrs->pqrs',weights,spatial_g,spatial_g)
    diagnostics={'chemist_input':array_record(chem),'chemist_input_frobenius':fnorm(chem),
        'chemist_interaction_transpose_asymmetry_l1':float(np.abs(interaction-interaction.T).sum()),
        'chemist_interaction_imaginary_l1':float(np.abs(interaction.imag).sum()),
        'chemist_plain_square_reconstruction_frobenius':fnorm(chem-reconstructed_chem),
        'correction_vs_source_reordering_frobenius':fnorm(correction-expected_corr),
        'raw_normal_order_one_body_residual_frobenius':fnorm(raw_c),
        'raw_normal_order_antisym_two_body_residual_frobenius':fnorm(target_a-raw_a),
        'raw_two_body_residual_relative':fnorm(target_a-raw_a)/fnorm(target_a) if fnorm(target_a) else None,
        'hypothetical_projection_one_body_change_frobenius':fnorm(raw_c-post_c),
        'hypothetical_projection_antisym_two_body_change_frobenius':fnorm(raw_a-post_a),
        'scope':'coefficient/tensor Frobenius diagnostics, not Hamiltonian/operator norm, signal error, u bound or ground-state proof',
        'formula':'C_raw=correction+sum(lambda*g@g); T_raw[p,q,r,s]=-sum(lambda*g[p,r]*g[q,s]); compare A(T)=(T-T_pq-T_rs+T_pq_rs)/4; no conjugation in squared products',
        'input_state_or_sector_operator_built':False}
    failed=[r['index'] for r in rows if r['exceeds_original_hermitization_tolerance']]
    change=fnorm(corrected_projected-corrected)
    summary={'schema':'track_a_h6_df_diagnostic_summary_v1','actual_rank':rank,'truncation_value':float(np.asarray(truncation)),
        'kwargs':dict(plan()['kwargs']),'implicit_defaults_not_passed':dict(plan()['implicit_defaults_not_passed']),
        'correction_raw':array_record(correction),'corrected_one_body_raw':array_record(corrected),
        'corrected_one_body_hypothetical_post':array_record(corrected_projected),
        'corrected_one_body_hermitization_change_frobenius':change,'hermitization_tolerance_unchanged':tol,
        'fragments':rows,'failed_fragment_indices':failed,'fragment_15':rows[15] if rank>15 else None,
        'all_original_hermitization_checks_satisfied':not failed and change<=tol,
        'representation_diagnostics':diagnostics,'raw_retained_without_filtering':True,
        'projected_arrays_diagnostic_only':True,'DF_policy_changed':False,'H6_input_accepted':False,
        'historical_raw_bytes_identity_claim':False,'H6_status':'H6_NOT_AUTHORIZED',
        'contract_status':'DRAFT_NOT_AUTHORIZATION','mandatory_stop':True,'next_stage_authorized':False}
    values={'g_hypothetical_hermitian':projected,'corrected_one_body_raw':corrected,
            'corrected_one_body_hypothetical_hermitian':corrected_projected,
            'chemist_input':chem,'chemist_raw_reconstruction':reconstructed_chem,
            'normal_order_one_body_raw_residual':raw_c,'normal_order_two_body_raw':raw_t,
            'normal_order_two_body_target_antisym':target_a}
    return summary,values

class DiagnosticPort:
    def __init__(self, manifest, writer, progress, *, root, decomposer=None, payload=None):
        self.manifest,self.writer,self.progress,self.root=manifest,writer,progress,Path(root)
        self.decomposer,self.payload=decomposer,payload
        self.calls=CallBudget(decomposition=1,fragment_diagnostics=36)
        self.decomposition_returned=False;self.summary_saved=False

    def execute(self):
        from .ax2b_h6_df_diagnostic_contract_v1 import file_hash
        caps=self.manifest['plan']['caps'];self.progress.phase('saved_input')
        if self.payload is None:
            path=self.root/self.manifest['input_identity']['input_path']
            if file_hash(path)!=INPUT_SHA:raise ValueError('INPUT_SHA_BEFORE_DECODE')
            with np.load(path,allow_pickle=False) as z:one=z['one_body'].copy();two=z['two_body'].copy()
        else:one,two=(np.asarray(x).copy() for x in self.payload)
        one.setflags(write=False);two.setflags(write=False)
        self.writer.write('input_decode_receipt.json',{'one_body':array_record(one),'two_body':array_record(two),
            'input_sha256':self.manifest['input_identity']['input_sha256'],'input_mutated':False})
        self.progress.phase('decomposition');self.calls.take('decomposition')
        self.writer.write('decomposition_call.json',{'kwargs':dict(self.manifest['plan']['kwargs']),
            'implicit_defaults_not_passed':dict(self.manifest['plan']['implicit_defaults_not_passed']),
            'two_body_before_call':array_record(two),'call_count':1,'different_execution_from_old_failure':True})
        if self.decomposer is None:
            from openfermion import low_rank_two_body_decomposition
            self.decomposer=low_rank_two_body_decomposition
        config=io.StringIO()
        with contextlib.redirect_stdout(config):np.show_config()
        if len(config.getvalue().encode())>32768:raise ValueError('RUNTIME_CONFIG_CAP')
        self.writer.write('runtime_environment.json',{'sealed_environment':self.manifest.get('environment'),
            'numpy_config_text':config.getvalue(),'actual_cpu_affinity':sorted(os.sched_getaffinity(0)),
            'thread_environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS')},
            'called_module':getattr(self.decomposer,'__module__',None),'called_qualname':getattr(self.decomposer,'__qualname__',None),
            'explicit_kwargs':dict(self.manifest['plan']['kwargs'])})
        returned=self.decomposer(two.copy(),**self.manifest['plan']['kwargs'])
        self.decomposition_returned=True
        if not isinstance(returned,(tuple,list)) or len(returned)!=4:raise ValueError('RAW_RETURN_ARITY')
        weights,gs,correction,truncation=(np.asarray(x) for x in returned)
        arrays={'lambdas_raw':weights,'g_matrices_raw':gs,'one_body_correction_raw':correction,'truncation_value_raw':truncation}
        receipt=write_npz(self.writer,'raw_decomposition.npz',arrays,caps['raw_expanded_bytes'])
        self.writer.write('raw_decomposition_receipt.json',{**receipt,'raw_original_values':True,
            'hypothetical_projection_applied':False,'kwargs':dict(self.manifest['plan']['kwargs'])})
        self.progress.phase('diagnostics')
        summary,values=describe(one,two,weights,gs,correction,truncation,budget=self.calls)
        projected=write_npz(self.writer,'hypothetical_hermitization.npz',values,caps['raw_expanded_bytes'])
        self.writer.write('hypothetical_receipt.json',{**projected,'diagnostic_only':True,'H6_input_accepted':False})
        self.writer.write('diagnostic_summary.json',summary);self.summary_saved=True
        self.progress.update(point='diagnostic_summary_saved',calls_attempted=dict(self.calls.used))
        if self.payload is None and file_hash(path)!=INPUT_SHA:raise ValueError('INPUT_SHA_AFTER_DIAGNOSTIC')
        return summary
