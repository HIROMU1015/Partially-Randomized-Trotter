"""Narrow pure-RTE q8 connection. No scientific work occurs on import.

Existing SOURCEs keep their r<=8 search domain. Only private copies of the
parameter-validation hooks are widened here; the finite law is unchanged.
"""
from dataclasses import asdict
from fractions import Fraction as Q
import math
from . import h4_pauli_pr_search as S
from . import h4_conditional_cng as A

H4_MEAN='artifacts/pauli_rte_event_extension_preparation/2026-10-26/inputs/h4_mean_d3.py'

def candidate_id(prefix,q,r,K):
    if any(type(x) is not int for x in (prefix,q,r,K)) or prefix!=0 or q!=8 or r not in (8,16,32,64,128) or K not in (2,4):
        raise ValueError('PURE_RTE_Q8_EXTENSION_DOMAIN')
    return f'p{prefix:03d}_q{q}_r{r}_K{K}'

def validate(molecule,r,K):
    candidate_id(0,8,r,K)
    if molecule not in ('H4','H6') or (molecule=='H4' and r not in (8,16,32)):
        raise ValueError('MOLECULE_EVENT_EXTENSION_DOMAIN')

def config(r,K):
    from .fixed_signal_evaluation import FixedSignalConfig
    candidate_id(0,8,r,K)
    return FixedSignalConfig('pauli_jw',method='PR',T=.8,L_D=0,q=8,r=r,K=K,cost_samples=8,seed=20261025)

_law=A.extract('src/trotterlib/h4_pauli_pr_search.py','law_descriptor',
    dict(__package__=__package__,asdict=asdict,math=math,candidate_id=candidate_id,probability_masses=S.probability_masses))
_h4_seed=A.extract('src/trotterlib/h4_pauli_pr_search.py','trajectory_seed',dict(candidate_id=candidate_id))
_h6_seed=A.extract('scripts/pauli_h2_h6_study.py','seeds',{})

def law_descriptor(hamiltonian,r,K):
    return _law(hamiltonian,config(r,K))

def trajectory_seed(molecule,r,K,cohort,index):
    validate(molecule,r,K)
    if cohort not in ('exploration','confirmation') or type(index) is not int or index<0:raise ValueError('COHORT_INDEX')
    if molecule=='H4':return _h4_seed(0,8,r,K,cohort,index)
    return _h6_seed(dict(molecule=6,config=dict(L_D=0,q=8,r=r,K=K)),cohort,index)

class Capture:
    def observe(self,adapter,sequences,step_seeds):
        self.events=[e for seq in sequences for e in seq]
        self.step_seeds=step_seeds

def trajectory(hamiltonian,molecule,r,K,cohort,index):
    from .rte_event_audit import AuditedPauliSignalAdapter
    from .h4_d2_pauli_control import central_control_body
    from .fixed_signal_evaluation import hadamard_wrapper
    validate(molecule,r,K);cfg=config(r,K);capture=Capture()
    seed=trajectory_seed(molecule,r,K,cohort,index)
    adapter=AuditedPauliSignalAdapter(hamiltonian,cfg.method,cfg.L_D,cfg.T/cfg.q,r,K,audit=capture)
    body=adapter.trajectory(q=cfg.q,seed=seed)
    assert len(capture.events)==8*r
    lift=central_control_body(body)
    return body,{a:hadamard_wrapper(lift.body,a,already_controlled=True) for a in A.AXES},dict(
        seed=seed,event_count=len(capture.events),events=capture.events,step_seeds=capture.step_seeds,control_sidecar=lift.sidecar)

def h4_mean(law,raw,reference,bits):
    # Same FLINT mean-matrix/vector path as the fixed H4 execution SOURCE.
    validate('H4',law['config']['r'],law['config']['K'])
    if law['config']['L_D']!=0 or law['config']['q']!=8 or law['n_system'] not in (2,3,8):
        raise ValueError('H4_PURE_RTE_INPUT_FAMILY')
    f=A.extract(H4_MEAN,'_mean_enclosure',dict(__package__=__package__,Q=Q))
    out=f(law,raw,reference,bits)
    out['B_saved_exact']=str(Q(law['B_saved_binary64']))
    return out

def h6_mean(model,reference,law):
    from .pauli_vector_tight_enclosure import evaluate
    validate('H6',law['config']['r'],law['config']['K'])
    if law['config']['L_D']!=0 or law['config']['q']!=8 or model['n_qubits']!=law['n_system'] or law['n_system'] not in (2,3,12):
        raise ValueError('H6_PURE_RTE_INPUT_FAMILY')
    return evaluate(model,reference,law['config'],law)

def account_target(target,prices,epsilon,law):
    from .h6_rte_fairness import account,scale
    return account(target,prices,epsilon,scale=scale(Q(law['B_saved_binary64'])))
