"""One frozen saved-input H4 or H6 diagnostic; no native compile or sampling."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, data):
    with Path(path).open('x') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')


def run(config, system, output):
    class NoGPUImports:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in ('cupy','cupyx','cuquantum'):
                raise ImportError('GPU modules excluded from this CPU diagnostic')
            return None
    sys.meta_path.insert(0, NoGPUImports())
    provider = Path(config['provider_root'])
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    sys.path.insert(1,str(provider/'src'))
    import numpy as np
    from scipy.linalg import expm
    from h46_new2_core import spectrum, occupation_one_body, compare_prefix, complex_record
    from trotterlib.df_hamiltonian import df_linear_operator
    from trotterlib.pr2_matched_accuracy_m1_execution import _prepare
    from trotterlib.rte import finite_taylor_operator

    def progress(phase, **details):
        print(json.dumps(dict(system=system,phase=phase,elapsed=time.monotonic()-started,**details)),flush=True)

    started = time.monotonic()
    inp = config['inputs']
    progress('load_verified_saved_input')
    if system=='H4':
        from trotterlib.pr2_new_series_validation import _load_snapshot_once
        ham,sector,full_state,state,metadata,_ = _load_snapshot_once(Path(inp['H4_snapshot.npz']['path']))
        expected_rank=12
    else:
        from trottertracks.resource_applicability.ax2b_h6_saved_completion_loader_v1 import load_h6_snapshot
        ham,sector,full_state,state,metadata,_ = load_h6_snapshot(
            Path(inp['H6_snapshot.npz']['path']),
            json.loads(Path(inp['H6_snapshot_receipt.json']['path']).read_text()),
            json.loads(Path(inp['H6_df_receipt.json']['path']).read_text()))
        expected_rank=19
    if ham.n_blocks!=expected_rank or abs(np.linalg.norm(state)-1)>1e-12:
        raise ValueError('SAVED_INPUT_RANK_OR_STATE')
    progress('occupation_matrices',dimension=sector.dimension,rank=ham.n_blocks)
    one=occupation_one_body(ham.one_body,sector)
    fragments=[]
    for i,g in enumerate(ham.g_matrices):
        b=occupation_one_body(g,sector)
        fragments.append(ham.lambdas[i]*(b@b))
    identity=np.eye(sector.dimension)
    full=ham.constant*identity+one+sum(fragments)
    # Independent existing matrix-free path: saved state and three fixed columns.
    operator,_=df_linear_operator(ham,sector,backend='python')
    probes=[state]+[identity[:,i] for i in (0,sector.dimension//2,sector.dimension-1)]
    independent_error=max(float(np.linalg.norm(full@v-operator@v)) for v in probes)
    if independent_error>1e-9:
        raise ValueError('INDEPENDENT_DF_RECONSTRUCTION')
    T=config['plan']['T']
    reference=expm(-1j*T*full)@state
    full_spec=spectrum(full)
    spectral_reference=full_spec[1]@(np.exp(-1j*T*full_spec[0])*(full_spec[2]@state))
    reference_error=float(np.linalg.norm(reference-spectral_reference))
    if reference_error>1e-9:
        raise ValueError('REFERENCE_EXPM_EIGH')
    reference_signal=complex(np.vdot(state,reference))
    progress('cached_fragment_spectra')
    spectra=[spectrum(one)]+[spectrum(a) for a in fragments]
    results=[]
    for prefix in config['plan']['prefixes'][system]:
        preparation=_prepare(ham,prefix)
        if preparation.deterministic_fragment_indices!=tuple(range(prefix)) or preparation.coefficient_atol!=0.:
            raise ValueError('GENERATION_PREFIX_OR_CUTOFF')
        scalar=ham.constant+preparation.extracted_identity_coefficient
        tail=sum(fragments[prefix:])-preparation.extracted_identity_coefficient*identity
        closure=float(np.linalg.norm(one+sum(fragments[:prefix])+tail+scalar*identity-full,'fro'))
        if closure>1e-9:
            raise ValueError('SCALAR_TAIL_FULL_H_CLOSURE')
        tail_spec=spectrum(tail)
        # Source-defined dense Taylor polynomial against our spectral implementation.
        probe_time=-.003
        source_taylor=finite_taylor_operator(tail/preparation.exact_rte_lambda_r,
            preparation.exact_rte_lambda_r*probe_time,4)@state
        from h46_new2_core import polynomial
        local_taylor=tail_spec[1]@(polynomial(tail_spec[0],probe_time,4)*(tail_spec[2]@state))
        taylor_error=float(np.linalg.norm(source_taylor-local_taylor))
        if taylor_error>1e-9:
            raise ValueError('PAIRED_TAYLOR_SOURCE_EQUIVALENCE')
        for label in config['plan']['formulas']:
            for q in config['plan']['q']:
                value=compare_prefix(state,spectra[:prefix+1],tail_spec,scalar,
                    preparation.exact_rte_lambda_r,label,q,T=T,
                    total_steps=config['plan']['tail_steps_per_outer'],cutoffs=tuple(config['plan']['K']))
                ideal_signal=complex(np.vdot(state,value.pop('exact_tail_vector')))
                ideal_bias=ideal_signal-reference_signal
                finite_rows=value.pop('finite_rows')
                for row in finite_rows:
                    z=row['corrected_signal'];bias=z-reference_signal
                    row['corrected_signal']=complex_record(z)
                    row['raw_signal']=complex_record(row['raw_signal'])
                    row['total_signed_bias']=complex_record(bias)
                    row['PF_signed_bias']=complex_record(ideal_bias)
                    row['RTE_signed_bias']=complex_record(z-ideal_signal)
                    row['absolute_complex_signal_bias']=abs(bias)
                    row['empirical_allowance_applied']=False
                    row['total_shot_count']=None
                    row['compiled_total_cost']=None
                results.append(dict(system=system,prefix=prefix,label=label,q=q,
                    reference_signal=complex_record(reference_signal),
                    exact_tail_PF_signal=complex_record(ideal_signal),
                    exact_tail_PF_absolute_signal_bias=abs(ideal_bias),
                    full_Fock_identity_scalar=scalar,
                    lambda_R=preparation.exact_rte_lambda_r,
                    preparation_hash=preparation.preparation_hash,
                    scalar_tail_closure_error=closure,
                    source_taylor_negative_time_error=taylor_error,
                    finite_rows=finite_rows,**value))
                progress('case_complete',prefix=prefix,label=label,q=q)
    return dict(status='COMPLETE_DIAGNOSTIC',system=system, SOURCE=config['SOURCE'],
        provider_SOURCE=config['provider_SOURCE'],plan_sha256=config['plan_sha256'],
        conditions=dict(geometry_angstrom=1.,basis='STO-3G',rank=ham.n_blocks,
            n_qubits=ham.n_qubits,sector_dimension=sector.dimension,T=T,
            saved_state_unchanged=True,generation_order_prefix=True,coefficient_atol=0.),
        saved_metadata_identity={k:metadata[k] for k in ('hamiltonian_hash','state_hash','sector_state_vector_hash') if k in metadata},
        reference_expm_eigh_vector_error=reference_error,
        independent_matrix_free_probe_error=independent_error,
        records=results,wall_seconds=time.monotonic()-started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        BLAS_threads_requested=int(os.environ['OPENBLAS_NUM_THREADS']),
        scientific_scope='deterministic finite-distribution expectation; no sampled trajectory, gate-level validation or compiled cost',
        normalization_multiplier_is_only_B_squared_not_complete_shot_budget=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--system',choices=['H4','H6'],required=True)
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    out=Path(config['output'])/args.system;out.mkdir(exist_ok=False)
    try:
        if config.get('authorized') is not True or args.system not in config['authorized_systems']:
            raise ValueError('USER_AUTHORIZATION_BINDING')
        if file_hash(config['plan_path'])!=config['plan_sha256']:
            raise ValueError('PLAN_CHANGED')
        if json.loads(Path(config['plan_path']).read_text())!=config['plan']:
            raise ValueError('PLAN_CONTENT_CHANGED')
        for path,h in config['source_hashes'].items():
            if file_hash(Path(config['source_root'])/path)!=h:raise ValueError('SOURCE_CHANGED:'+path)
        for path,h in config['provider_source_hashes'].items():
            if file_hash(Path(config['provider_root'])/path)!=h:raise ValueError('PROVIDER_CHANGED:'+path)
        for row in config['inputs'].values():
            if Path(row['path']).stat().st_size!=row['bytes'] or file_hash(row['path'])!=row['sha256']:
                raise ValueError('INPUT_CHANGED')
        result=run(config,args.system,out);write_new(out/'result.json',result)
        write_new(out/'terminal.json',dict(status='COMPLETE',SOURCE=config['SOURCE']))
    except BaseException as exc:
        traceback.print_exc()
        write_new(out/'terminal.json',dict(status='STOP',reason=type(exc).__name__+':'+str(exc),SOURCE=config.get('SOURCE')))
        raise


if __name__=='__main__':
    main()
