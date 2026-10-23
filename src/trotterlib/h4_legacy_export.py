"""Frozen-runtime preparation bridge. No execution or numerical imports on import.

Called only through the C controller's consumed, hash-bound worker capability.
Does not call chemistry, reference/law means, RNG, circuit builders or compilers.
"""
import hashlib
import json
import math
from pathlib import Path
import sys


def export(namespace,representation,control):
    c,seal=control.worker_authorize(namespace,representation,bridge=True)
    sys.path.insert(0,str(Path(c['legacy_root'])/'src'))
    import numpy as np
    from dataclasses import asdict
    from qiskit.quantum_info import Operator
    from trotterlib.h4_pilot_science import load_inputs
    from trotterlib.fixed_signal_evaluation import FixedSignalConfig,prepare_signal_adapter
    from trotterlib.df_trotter.ops import one_body_diagonal_primitives,df_squared_diagonal_primitives
    from trotterlib.pauli_signal_adapter import array_hash
    destination=Path(c['input_directory'])
    receipt=json.loads((destination/'receipt.json').read_text())
    pauli,df,state,loaded=load_inputs(destination,dict(receipt_sha256=control.sha(destination/'receipt.json'),snapshot_sha256=receipt['snapshot_sha256']))
    with np.load(destination/'snapshot.npz',allow_pickle=False) as snapshot:
        saved_matrix=snapshot['non_DF_matrix' if representation=='pauli_jw' else 'DF_matrix'].copy()
    targets=json.loads(Path(c['method_map_path']).read_text())['rows']
    old_results={method:json.loads(Path(c['saved_results'][representation+'_'+method]).read_text()) for method in ('PF','PR')}
    return build_export(pauli,df,state,saved_matrix,old_results,targets,representation,control,c['historical_scientific_SOURCE'])


def build_export(pauli,df,state,saved_matrix,old_results,targets,representation,control,historical_source):
    """Same production assembler for supplied artificial objects or gated inputs."""
    import numpy as np
    from dataclasses import asdict
    from qiskit.quantum_info import Operator
    from trotterlib.fixed_signal_evaluation import FixedSignalConfig,prepare_signal_adapter
    from trotterlib.df_trotter.ops import one_body_diagonal_primitives,df_squared_diagonal_primitives
    from trotterlib.pauli_signal_adapter import array_hash
    def pairs(a):
        a=np.asarray(a)
        if a.ndim==1:return [[float(z.real),float(z.imag)] for z in a]
        return [pairs(row) for row in a]
    def basis(spec):
        operations=[]
        for (gate,qubits),description in zip(spec.runtime_basis_operations,spec.basis_change_operations,strict=True):
            operations.append(dict(qubits=list(qubits),description=asdict(description),
                matrix=pairs(Operator(gate).data),inverse_matrix=pairs(Operator(gate.inverse()).data)))
        return dict(basis_hash=spec.basis_hash,operations=operations)
    cases={};h=pauli if representation=='pauli_jw' else df
    for method in ('PF','PR'):
        old=old_results[method]
        oldconfig=old['config'];adapter=prepare_signal_adapter(h,FixedSignalConfig(representation,method=method,T=.8,q=8,L_D=oldconfig['L_D'],r=4,K=4))
        control.require(adapter.metadata()==old['model'],'MISSING_FROZEN_PREPARATION')
        blocks=[];components=[]
        if representation=='pauli_jw':
            for index,(label,coef) in enumerate(adapter.retained):
                blocks.append(dict(operator_key=str(index),label=label,coefficient=coef))
            if method=='PR':
                components=[dict(label=label,sign=1 if coef>0 else -1,coefficient_abs=abs(coef)) for label,coef in adapter.tail]
            extracted=0.;scalar=[-pauli.constant*.1]
        else:
            prep=adapter.preparation
            for b in prep.deterministic_blocks:
                prim=one_body_diagonal_primitives(np.asarray(b.diagonal_eigenvalues),.1/2) if b.block_kind=='one_body' else df_squared_diagonal_primitives(np.asarray(b.diagonal_eta),b.lam,.1/2)
                # Preserve the source exact-zero pruning policy (atol=0).
                primitive=dict(global_phase=prim.global_phase if prim.global_phase!=0 else 0.,
                    rz=[(q,a) for q,a in prim.rz],rzz=[(p,q,a) for p,q,a in prim.rzz if a!=0])
                blocks.append(dict(operator_key='one_body' if b.block_kind=='one_body' else str(b.original_fragment_index),basis=basis(b),primitives=primitive))
            extracted=prep.extracted_identity_coefficient
            scalar=[-.1*prep.constant_coefficient,-.1*extracted]
            if method=='PR':
                extraction=prep.tail_extraction
                from types import SimpleNamespace
                for comp in prep.rte_preparation.symbolic_tail.components:
                    definition=extraction.basis_definition(comp.basis_id)
                    proxy=SimpleNamespace(basis_hash=definition.basis_hash,runtime_basis_operations=definition.runtime_operations,basis_change_operations=definition.metadata.operations)
                    components.append(dict(sign=comp.coefficient_sign,coefficient_abs=comp.coefficient_abs,
                        support=list(comp.diagonal_pauli_support),basis=basis(proxy)))
        axes=[]
        for row in targets:
            if row['case']!=representation+'_'+method:continue
            b=old['signal']['signed_bias']['real' if row['axis']=='cosine' else 'imag']
            axes.append(dict(epsilon_sig=row['epsilon_sig'],axis=row['axis'],signed_bias=b,target_exact=row['design_total_error_target_exact']))
        case=dict(metadata=adapter.metadata(),expected_metadata=old['model'],config=dict(T=.8,q=8,r=4,K=4,method=method),
            blocks=blocks,components=components,extracted_identity=extracted,native_scalar_phases=scalar,
            saved_axes=axes,saved_normalization=old['normalization'],
            semantic_error_bound_exact='0',circuit_correspondence_proved=True,
            exact_event_range_proved=(representation=='pauli_jw'))
        # Exact encoded-operator evidence and conservative basis cancellation
        # bounds are available. Literal compiler/wrapper correspondence remains
        # a separate acceptance gate, never fabricated from an empirical PASS.
        case['correspondence_scope']='encoded native local matrices and fixed law; literal compiled-wrapper transfer not certified'
        if method=='PR':
            distribution=adapter.distribution
            ps=[x.probability for x in (adapter.components if representation=='pauli_jw' else adapter.preparation.rte_preparation.symbolic_tail.components)]
            total=math.fsum(ps);ps=[x/total for x in ps]
            ops=[float(x) for x in distribution.order_probabilities]
            def cdf(p):
                result=np.asarray(p,dtype=np.float64).cumsum();result/=result[-1];return result.tolist()
            case['law']=dict(numpy_version=np.__version__,uniform_bits=53,
                component_probabilities=ps,component_cdf=cdf(ps),order_probabilities=ops,order_cdf=cdf(ops),
                angles=[math.atan(distribution.dimensionless_step_time/(k+1)) for k in (0,2,4)],
                finite_distribution=distribution.to_dict())
        cases[method]=case
    fullspec=dict(constant=h.constant)
    if representation=='pauli_jw':fullspec['terms']=list(pauli.terms)
    else:fullspec.update(one_body=pairs(df.one_body),lambdas=[float(x) for x in df.lambdas],g_matrices=[pairs(g) for g in df.g_matrices])
    model=dict(schema='h4_frozen_C_export_v1',representation=representation,n_qubits=int(round(math.log2(len(state)))),
        qubit_order='qiskit_little_endian',state=pairs(state),state_hash=array_hash(state),
        preparation_verified=True,hamiltonian=fullspec,saved_full_matrix=pairs(saved_matrix),
        operator_agreement_limit=1e-9,step_time=.1,cases=cases,
        old_scientific_SOURCE=historical_source,scope='NO_NEW_SCF_DF_STATE_OR_SAMPLING')
    model['binding_hash']=control.digest(model)
    return model
