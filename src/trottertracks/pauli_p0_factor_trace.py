"""Local coefficient tracing for the fixed P0 square-factor/JW path.

No saved inputs are loaded here. The caller owns scientific scope and output.
Default additions call OpenFermion unchanged; shadow additions preserve every
nonzero partial coefficient. Local subclasses/function globals never mutate
OpenFermion classes, global EQ_TOLERANCE, or the production P0 module.
"""
import ast
from copy import deepcopy
import importlib
import inspect
from types import FunctionType

import numpy as np
from openfermion import FermionOperator, QubitOperator
from openfermion.config import EQ_TOLERANCE

from . import df_head_pauli_tail_p0 as P0


def number(value):
    z=complex(value)
    return dict(real=float(z.real),imag=float(z.imag),
                real_hex=float(z.real).hex(),imag_hex=float(z.imag).hex())


def terms(operator):
    """Retain insertion order and binary64 coefficients; never sort accumulation."""
    return [dict(term=[[int(q),a] for q,a in term],**number(c))
            for term,c in operator.terms.items()]


def differences(left,right):
    keys=list(left.terms)+[k for k in right.terms if k not in left.terms]
    return [dict(term=[[int(q),a] for q,a in k],
                 left=number(left.terms.get(k,0)),right=number(right.terms.get(k,0)),
                 difference=number(left.terms.get(k,0)-right.terms.get(k,0)))
            for k in keys if left.terms.get(k,0)!=right.terms.get(k,0)]


class Recorder:
    def __init__(self):
        self.context={};self.additions=0;self.deletions=[]

    def operator_class(self,base,*,retain_small):
        recorder=self
        class LocalOperator(base):
            def __iadd__(self,addend):
                if not isinstance(addend,type(self)):
                    raise TypeError('Trace requires same local operator type')
                before={k:self.terms.get(k,0.) for k in addend.terms}
                recorder.additions+=1
                if retain_small:
                    for k,c in addend.terms.items():
                        value=self.terms.get(k,0.)+c
                        if value!=0:self.terms[k]=value
                        else:self.terms.pop(k,None)
                else:
                    base.__iadd__(self,addend)
                # Include the source Fermion term in each outer JW addition.
                caller=inspect.currentframe().f_back
                source_term=caller.f_locals.get('term') if caller.f_code.co_name=='_jordan_wigner_fermion_operator' else None
                for k,c in addend.terms.items():
                    partial=before[k]+c
                    if partial!=0 and k not in self.terms:
                        recorder.deletions.append(dict(operation_index=recorder.additions,
                            context=deepcopy(recorder.context),operator_type=base.__name__,
                            caller=caller.f_code.co_name,
                            source_fermion_term=None if source_term is None else [[int(q),a]for q,a in source_term],
                            term=[[int(q),a]for q,a in k],previous=number(before[k]),
                            incoming=number(c),partial_before_removal=number(partial)))
                del caller
                return self
        return LocalOperator


def one_body(root,operator_class):
    path='src/trotterlib/o1_m1_matched_split.py'
    parent=next(n for n in ast.parse(P0.blob(root,P0.M1,path)).body
                if isinstance(n,ast.FunctionDef) and n.name=='prepare_matched_a')
    node=next(n for n in parent.body if isinstance(n,ast.FunctionDef) and n.name=='one_body')
    scope=dict(np=np,Any=object,FermionOperator=operator_class)
    exec(compile(ast.Module(body=[node],type_ignores=[]),P0.M1+':'+path,'exec'),scope)
    return scope['one_body']


def jw_function(operator_class):
    """Reuse installed JW bytecode and its multiplication/cache/insertion order."""
    module=importlib.import_module('openfermion.transforms.opconversions.jordan_wigner')
    original=module._jordan_wigner_fermion_operator
    local=dict(original.__globals__,QubitOperator=operator_class)
    return FunctionType(original.__code__,local,original.__name__,original.__defaults__,original.__closure__)


def to_base(operator,base):
    out=base();out.terms=dict(operator.terms)
    return out


def trace(root,ham,prefix):
    """One diagnostic split: evidence plus default/shadow/bridge Pauli operators.

    bridge_jw is no-filter JW of the default Fermion residual. Thus the two
    operator differences shadow-bridge and bridge-default telescope exactly to
    the complete shadow-default difference (up to matrix summation roundoff).
    No alternate physical candidate or production dictionary is constructed.
    """
    if type(prefix)is not int or not 0<=prefix<len(ham.lambdas):
        raise ValueError('Nonempty explicit factor prefix required')
    if len(ham.lambdas)!=len(ham.g_matrices):raise ValueError('Factor/weight count differs')
    default=Recorder();shadow=Recorder();bridge=Recorder()
    FD=default.operator_class(FermionOperator,retain_small=False)
    FS=shadow.operator_class(FermionOperator,retain_small=True)
    QD=default.operator_class(QubitOperator,retain_small=False)
    QS=shadow.operator_class(QubitOperator,retain_small=True)
    QB=bridge.operator_class(QubitOperator,retain_small=True)
    make_d=one_body(root,FD);make_s=one_body(root,FS)
    rd=FD();rs=FS();stages=[];factor_records=[]
    def stage(name,a,b,**context):
        diff=differences(a,b)
        row=dict(stage=name,context=context,default=terms(a),shadow=terms(b),
                 differences=diff,identical=not diff)
        stages.append(row)
        return row
    for index in range(prefix,len(ham.lambdas)):
        weight=float(ham.lambdas[index]);g=np.asarray(ham.g_matrices[index])
        if not np.isfinite(weight) or not np.isfinite(g).all():raise ValueError('Nonfinite factor input')
        default.context=shadow.context=dict(stage='one_body_addition',factor_index=index)
        a=make_d(g);b=make_s(g)
        first=stage('one_body',a,b,factor_index=index)
        factor=dict(factor_index=index,weight=number(weight),shape=list(g.shape),
                    nonzero_input=[dict(indices=[int(q),int(r)],**number(g[q,r]))
                                   for q,r in np.argwhere(g!=0)],one_body=first)
        default.context=shadow.context=dict(stage='square',factor_index=index)
        aa=a*a;bb=b*b
        factor['square']=stage('square',aa,bb,factor_index=index)
        default.context=shadow.context=dict(stage='weight_multiplication',factor_index=index)
        wa=weight*aa;wb=weight*bb
        factor['weighted']=stage('weighted',wa,wb,factor_index=index)
        default.context=shadow.context=dict(stage='fermion_residual_addition',factor_index=index)
        rd+=wa;rs+=wb
        factor['residual_after_factor']=stage('fermion_residual',rd,rs,factor_index=index)
        factor_records.append(factor)
    default.context=dict(stage='JW_addition',prefix=prefix)
    shadow.context=dict(stage='JW_addition',prefix=prefix)
    bridge.context=dict(stage='JW_of_default_fermion_no_filter',prefix=prefix)
    jd=jw_function(QD)(rd)
    js=jw_function(QS)(rs)
    jb=jw_function(QB)(rd)
    last=stage('JW',jd,js,prefix=prefix)
    first=next((dict(stage=s['stage'],context=s['context'],first_difference=s['differences'][0])
                for s in stages if not s['identical']),None)
    evidence=dict(schema='P0_LOCAL_FACTOR_COEFFICIENT_TRACE_V1',prefix=prefix,
        factor_indices=list(range(prefix,len(ham.lambdas))),EQ_TOLERANCE=float(EQ_TOLERANCE),
        first_stage_difference=first,first_removed_nonzero_term=next(iter(default.deletions),None),
        factors=factor_records,JW=last,bridge_JW=terms(jb),
        default_fermion_residual=terms(rd),shadow_fermion_residual=terms(rs),
        removed_nonzero_terms=default.deletions,
        addition_counts=dict(default=default.additions,shadow=shadow.additions,bridge=bridge.additions),
        global_tolerance_changed=False,production_dictionary_adopted=False)
    return dict(evidence=evidence,default_jw=to_base(jd,QubitOperator),
                shadow_jw=to_base(js,QubitOperator),bridge_jw=to_base(jb,QubitOperator))
