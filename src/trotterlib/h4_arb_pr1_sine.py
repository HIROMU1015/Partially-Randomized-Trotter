"""PR1 sine enclosure: preserve unitary arithmetic; select +Im at contraction."""
import ast
from pathlib import Path
from types import SimpleNamespace

from . import h4_arb_pr1_cosine as CO
from . import h4_r1_sine as S

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25'
PREP=BASE
BASE_COMMIT='5c549779536bc272eb500d126a94845ae2bb1d6b'
B,C,P0,R1,TID=CO.B,CO.C,CO.P0,CO.R1,CO.TID
QPY_SHA='38a1a74afe2452685c3e4dfdeb6182085b6d6a28d6c9d90ed1ab2f4026f7abdd'
QPY_FP='88287619a8c27fa8370115ae33fd7f945984884391c6824a56b18490f81d564d'
STATE_HASH,SCALE=CO.STATE_HASH,CO.SCALE
GUARD='102330934580088554157/30525376945269386661330944'
PRECISIONS,LIMITS,NONFINITE=CO.PRECISIONS,dict(CO.LIMITS),CO.NONFINITE
finite_schedule=CO.finite_schedule
FORBIDDEN=dict(new_compile=0,sampling=0,Hamiltonian_state_generation=0,input_QPY_state_actions=0,
    PF=0,PR0=0,additional_trajectory=0,cosine=0,DF=0,R2=0,N_G_reanalysis=0,formal_certificate_acceptance=0)


class SineScope(CO.ScopeStrings):
    """Scope strings plus the sole numerical change: dot.real -> dot.imag."""
    def visit_Constant(self,node):
        node=super().visit_Constant(node)
        if isinstance(node.value,str):
            node.value=(node.value.replace('SAVED_PR1_NO_EXECUTION','SAVED_PR1_SINE_NO_EXECUTION')
                .replace('COSINE','SINE').replace('cosine','sine'))
        return node

    def visit_Attribute(self,node):
        node=self.generic_visit(node)
        if isinstance(node.value,ast.Name) and node.value.id=='dot' and node.attr=='real':node.attr='imag'
        return node


# Preserve the PR1 metadata scope/guards and all scalar/Pauli/order arithmetic.
_tree=ast.parse((ROOT/'src/trotterlib/h4_arb_pr0_preparation.py').read_text())
_nodes=[n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name in ('build_plan','validate_plan','_selected_unitary_enclosure')]
C.require(len(_nodes)==3,'REUSED_SELECTED_FUNCTIONS')
_plan_env=dict(vars(P0),TID=TID,R1=R1,_pr1_orders=CO._pr1_orders)
_module=CO.PlanScope().visit(ast.Module(body=_nodes,type_ignores=[]))
exec(compile(ast.fix_missing_locations(SineScope().visit(_module)),'<same selected unitary, PR1 sine +Im>','exec'),_plan_env)
_plan_env['_base_validate']=_plan_env['validate_plan']
_plan_env['_plan_env']=_plan_env
_tree=ast.parse((ROOT/'src/trotterlib/h4_arb_pr1_cosine.py').read_text())
_nodes=[n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name in ('validate_plan','build_plan')]
# build_plan needs the base builder under its old key; retain it as a separate
# mapping so that the wrapper cannot call itself recursively.
_base_build=_plan_env['build_plan']
_plan_env['_plan_env']={'build_plan':_base_build}
exec(compile(ast.fix_missing_locations(SineScope().visit(ast.Module(body=_nodes,type_ignores=[]))),
    '<same PR1 fixed metadata guards, sine axis>','exec'),_plan_env)
build_plan,validate_plan=_plan_env['build_plan'],_plan_env['validate_plan']
_kernel=_plan_env['_selected_unitary_enclosure']


def _selected_unitary_enclosure(plan,raw,bits,*,progress=None):
    signal,record=_kernel(plan,raw,bits,progress=progress)
    record.update(axis='sine',convention='+Im of selected raw-bra inner product',cosine_evaluated=False)
    return signal,record


P=SimpleNamespace(**{**vars(P0),'TID':TID,'build_plan':build_plan,'validate_plan':validate_plan,
    '_selected_unitary_enclosure':_selected_unitary_enclosure})

# PR0 production math/interval union/B scaling, reservation, process waits and
# finite schedule are reused. The new namespace, QPY identity and sine guard bind.
_tree=ast.parse((ROOT/'src/trotterlib/h4_arb_pr0_cosine.py').read_text())
_names=('validate_contract','load_inputs','_action','action_pair','reduce_pair','run')
_nodes=[n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name in _names]
C.require(len(_nodes)==len(_names),'REUSED_PAIR_PRODUCTION_FUNCTIONS')
_run_env={**vars(CO.OLD),**globals()}
exec(compile(ast.fix_missing_locations(SineScope().visit(ast.Module(body=_nodes,type_ignores=[]))),
    '<same PR pair production, PR1 sine only>','exec'),_run_env)
for _name in _names:globals()[_name]=_run_env[_name]
