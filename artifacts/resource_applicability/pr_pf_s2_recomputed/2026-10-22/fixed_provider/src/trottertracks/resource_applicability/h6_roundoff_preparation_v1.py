"""User-approved roundoff bridge; original seed, event choices and compiler stay.

Only randomized-tail bases with old saved definitions are aligned. Native
deterministic blocks are unchanged, including completed server Phase A 447.
No mathematical/scientific input is regenerated; comparisons are fail-closed.
"""
from contextlib import contextmanager
from dataclasses import replace
import json
import math
from pathlib import Path
from . import h6_frozen_preparation_v1 as codec

ATOL=1e-12
RTOL=1e-10
OPERATOR_CAP=1e-10
NATIVE_INPUT_CAP=1e-9  # Existing H6 absolute agreement gate, not a new allowance.
POLICY='H6_ROUNDOFF_SAVED_TAIL_BASIS_V1'


def policy():
    return dict(name=POLICY,absolute_tolerance=ATOL,relative_tolerance=RTOL,
        weighted_operator_difference_cap=OPERATOR_CAP,unitarity_cap=1e-12,
        existing_native_input_agreement_cap=NATIVE_INPUT_CAP,
        exact_fields='task/seed/order/selected-component/support/sign/basis/phase-role',
        numeric_fields='finite float metadata only',deterministic_basis='unchanged server native',
        compiler_changed=False,input_generated=False)


def compare(old,new,*,path='',differences=None):
    """Never round identities, strings, discrete event choices or signed values."""
    differences=[] if differences is None else differences
    if isinstance(old,float) and isinstance(new,(int,float)) and not isinstance(new,bool):
        if not math.isfinite(old) or not math.isfinite(new) or not math.isclose(old,new,rel_tol=RTOL,abs_tol=ATOL):
            raise ValueError('ROUNDING_NUMERIC_MISMATCH:'+path)
        if old!=new:differences.append(dict(path=path,absolute_difference=abs(old-new)))
    elif type(old) is dict and type(new) is dict and set(old)==set(new):
        for k in old:compare(old[k],new[k],path=path+'/'+k,differences=differences)
    elif type(old) is list and type(new) is list and len(old)==len(new):
        for i,(a,b) in enumerate(zip(old,new)):compare(a,b,path=path+'/'+str(i),differences=differences)
    elif type(old) is not type(new) or old!=new:
        raise ValueError('ROUNDING_DISCRETE_MISMATCH:'+path)
    return differences


def single_particle(operations,n):
    """Adjacent JW gates propagated in vacuum/one-particle space, no dense 2**n."""
    import numpy as np
    from qiskit.quantum_info import Operator
    result=np.eye(n,dtype=complex);vacuum=1+0j
    for gate,qubits in operations:
        matrix=Operator(gate).data
        step=np.eye(n,dtype=complex)*matrix[0,0]
        if gate.name=='rz' and len(qubits)==1:
            step[qubits[0],qubits[0]]=matrix[1,1]
        elif gate.name=='xx_plus_yy' and len(qubits)==2 and abs(qubits[0]-qubits[1])==1:
            step[np.ix_(qubits,qubits)]=matrix[np.ix_([1,2],[1,2])]
        else:raise ValueError('ROUNDING_UNSUPPORTED_JW_GATE')
        result=step@result;vacuum*=matrix[0,0]
    if abs(vacuum)<.99:raise ValueError('ROUNDING_VACUUM_PHASE')
    return result/vacuum


def saved_bases(root):
    from qiskit.circuit.library import RZGate,XXPlusYYGate
    from trotterlib.df_rte_tail import describe_basis_change_operations,canonical_basis_hash
    from dataclasses import asdict
    known={};references={}
    for path in sorted((Path(root)/codec.OLD_RUN).glob('cost_exploration_*/trajectory.json')):
        row=json.loads(path.read_bytes())
        for step in row.get('events') or []:
            for event in step:
                for app in event['application_sequence']:
                    index=int(app['df_fragment_id'].split('-')[-1]);ops=app['basis_change_operations']
                    if index in known and known[index]['metadata']!=ops:raise ValueError('ROUNDING_OLD_BASIS_CONFLICT')
                    if index not in known:
                        gates=[]
                        for op in ops:
                            if op['name'] not in ('rz','xx_plus_yy'):raise ValueError('ROUNDING_OLD_GATE')
                            factory={'rz':RZGate,'xx_plus_yy':XXPlusYYGate}[op['name']]
                            gates.append((factory(*[float(v) for v in op['parameters']]),tuple(op['qubits'])))
                        described=describe_basis_change_operations(gates)
                        new_ops=json.loads(codec.canonical([asdict(x) for x in described]))
                        # The exact gate names/positions/angle strings remain;
                        # libm-created matrix-byte hashes may differ by rounding.
                        if [{k:v for k,v in x.items() if k!='matrix_sha256'} for x in new_ops]!=[
                            {k:v for k,v in x.items() if k!='matrix_sha256'} for x in ops]:
                            raise ValueError('ROUNDING_OLD_LOCAL_GATE_METADATA')
                        known[index]=dict(metadata=ops,runtime=tuple(gates),new_metadata=new_ops,
                            new_basis_hash=canonical_basis_hash(12,described))
                    ref={k:app[k] for k in ('component_id','coefficient_abs','coefficient_sign','diagonal_pauli_support','basis_hash')}
                    if app['component_id'] in references and references[app['component_id']]!=ref:
                        raise ValueError('ROUNDING_OLD_COMPONENT_CONFLICT')
                    references[app['component_id']]=ref
    return known,references


def prepare_aligned(root,ham):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from trotterlib import df_trotter
    from trotterlib.df_trotter import ops
    from trotterlib.df_partial_randomized_pf import df_hamiltonian_to_model
    from trotterlib.pr2_matched_accuracy_m1_execution import _prepare,_prepare_discard
    from trotterlib.df_partial_s2 import df_hamiltonian_hash
    known,references=saved_bases(root);native=ops.build_df_blocks
    model=df_hamiltonian_to_model(ham);original=native(model);blocks={};checks=[]
    for index,(g,block) in enumerate(zip(model.G_list,original)):
        current=single_particle(block.U_ops,ham.n_qubits)
        current_g=(current*block.eta)@current.conj().T
        base_error=abs(float(block.lam))*ham.n_qubits**2*(np.linalg.norm(current_g)+np.linalg.norm(g))*np.linalg.norm(current_g-g)
        unitarity=float(np.linalg.norm(current.conj().T@current-np.eye(ham.n_qubits)))
        if unitarity>1e-12 or base_error>NATIVE_INPUT_CAP:raise ValueError('ROUNDING_NATIVE_INPUT_OPERATOR')
        aligned=block;difference=0.;permutation=list(range(ham.n_qubits))
        if index in known:
            basis=single_particle(known[index]['runtime'],ham.n_qubits)
            _,permutation=linear_sum_assignment(-np.abs(basis.conj().T@current)**2)
            eta=block.eta[permutation]
            aligned_g=(basis*eta)@basis.conj().T
            difference=abs(float(block.lam))*ham.n_qubits**2*(np.linalg.norm(aligned_g)+np.linalg.norm(current_g))*np.linalg.norm(aligned_g-current_g)
            if np.linalg.norm(basis.conj().T@basis-np.eye(ham.n_qubits))>1e-12 or difference>OPERATOR_CAP:
                raise ValueError('ROUNDING_ALIGNED_OPERATOR')
            aligned=replace(block,U_ops=known[index]['runtime'],eta=eta)
        identity=np.ascontiguousarray(g).tobytes()
        if identity in blocks:raise ValueError('ROUNDING_DUPLICATE_FRAGMENT')
        blocks[identity]=(float(block.lam),aligned)
        checks.append(dict(index=index,old_basis_used=index in known,
            permutation=[int(v) for v in permutation],native_input_weighted_bound=float(base_error),
            aligned_weighted_difference_bound=float(difference),unitarity=unitarity))
    if sum(x['aligned_weighted_difference_bound'] for x in checks)>OPERATOR_CAP:
        raise ValueError('ROUNDING_TOTAL_OPERATOR_DIFFERENCE')
    if sum(x['native_input_weighted_bound'] for x in checks)>NATIVE_INPUT_CAP:
        raise ValueError('ROUNDING_TOTAL_NATIVE_INPUT_DIFFERENCE')
    def tail_blocks(selected,*,sort='descending_abs'):
        if sort!='descending_abs':raise ValueError('ROUNDING_SORT_POLICY')
        result=[]
        for lam,g in zip(selected.lambdas,selected.G_list):
            expected,block=blocks[np.ascontiguousarray(g).tobytes()]
            if complex(lam).imag!=0. or float(complex(lam).real)!=expected:raise ValueError('ROUNDING_FRAGMENT_LAMBDA')
            result.append(block)
        return result
    # df_partial_s2's previously imported alias stays native; only the tail's
    # local import of ops.build_df_blocks receives the aligned definitions.
    ops.build_df_blocks=tail_blocks
    try:
        preparations={}
        for name in codec.expected_preparations(root):
            kind,prefix=name.split(':p')
            preparations[name]=(_prepare_discard if kind=='discard' else _prepare)(ham,int(prefix))
    finally:ops.build_df_blocks=native
    old=codec.expected_preparations(root)
    differences=[]
    for name,p in preparations.items():
        if name.startswith('faithful:') and p.hamiltonian_hash!=df_hamiltonian_hash(ham):
            raise ValueError('ROUNDING_PREPARATION_INPUT')
        compare({k:old[name][k] for k in ('lambda_r','extracted_identity','constant')},
            dict(lambda_r=p.exact_rte_lambda_r,extracted_identity=p.extracted_identity_coefficient,constant=p.constant_coefficient),
            path=name,differences=differences)
        components={c.component_id:c for c in p.tail_extraction.components}
        for cid,ref in references.items():
            if cid not in components:continue
            c=components[cid]
            mapped=dict(ref);index=int(c.df_fragment_id.split('-')[-1])
            if index in known:mapped['basis_hash']=known[index]['new_basis_hash']
            compare(mapped,dict(component_id=cid,coefficient_abs=c.coefficient_abs,
                coefficient_sign=c.coefficient_sign,diagonal_pauli_support=list(c.diagonal_pauli_support),basis_hash=c.basis_hash),
                path=name+'/'+cid,differences=differences)
    return preparations,dict(policy=policy(),fragment_checks=checks,old_basis_count=len(known),
        compared_saved_components=len(references),numeric_metadata_differences=differences,
        old_preparation_hash_exact=False,deterministic_preparation_unchanged=True,
        original_hamiltonian_hash=df_hamiltonian_hash(ham),compiler_changed=False)


def compare_partial(root,events,distribution):
    old=json.loads((Path(root)/codec.OLD_RUN/'cost_exploration_0450/trajectory.json').read_bytes())
    events=json.loads(codec.canonical(events))
    from copy import deepcopy
    known,_=saved_bases(root)
    mapped=deepcopy(old['events'])
    basis_map={}
    for step in mapped:
        for event in step:
            for app in event['application_sequence']:
                index=int(app['df_fragment_id'].split('-')[-1]);new_hash=known[index]['new_basis_hash']
                basis_map[app['basis_hash']]=new_hash
                app['basis_hash']=new_hash;app['basis_id']='df-basis-'+new_hash
                app['basis_change_operations']=known[index]['new_metadata']
    # Every replaced hash derives from the SAME ordered gate/angle definitions.
    def remap(value):
        if isinstance(value,dict):
            for k,v in list(value.items()):
                if k in ('basis_hash','basis_id') and isinstance(v,str):
                    if v in basis_map:value[k]=basis_map[v]
                    elif v.startswith('df-basis-') and v[9:] in basis_map:value[k]='df-basis-'+basis_map[v[9:]]
                elif k in ('basis_ids','basis_hashes') and isinstance(v,list):
                    value[k]=[('df-basis-'+basis_map[x[9:]]) if x.startswith('df-basis-') and x[9:] in basis_map else basis_map.get(x,x) for x in v]
                else:remap(v)
        elif isinstance(value,list):
            for row in value:remap(row)
    remap(mapped)
    differences=compare(mapped,events,path='events')
    compare(old['finite_distribution'],json.loads(codec.canonical(distribution.to_dict())),path='distribution',differences=differences)
    return dict(policy=policy(),original_event_digest=old['event_digest'],new_event_digest=codec.digest(events),
        exact_event_choices=True,only_numeric_differences=True,
        changed_numeric_values=len(differences),maximum_absolute_difference=max([0.,*[x['absolute_difference'] for x in differences]]))


@contextmanager
def use_preparations(preparations):
    from . import h6_roundoff_cost_port_v1 as port
    from trotterlib.df_partial_s2 import df_hamiltonian_hash
    before=port._prepare,port._prepare_discard
    def resolve(ham,prefix,discard):
        p=preparations[codec.key(discard,prefix)]
        target=ham.select_blocks(tuple(range(prefix))) if discard else ham
        if df_hamiltonian_hash(target)!=p.hamiltonian_hash:raise ValueError('ROUNDING_WORKER_HAMILTONIAN')
        return p
    port._prepare=lambda ham,prefix:resolve(ham,prefix,False)
    port._prepare_discard=lambda ham,prefix:resolve(ham,prefix,True)
    try:yield
    finally:port._prepare,port._prepare_discard=before


def read_preparations(path,sha):
    archive=codec.read_archive(path,sha256=sha)
    if archive['binding']['roundoff_policy']!=policy():raise ValueError('ROUNDING_ARCHIVE_POLICY')
    return codec.decode_preparations(archive['codec']),archive
