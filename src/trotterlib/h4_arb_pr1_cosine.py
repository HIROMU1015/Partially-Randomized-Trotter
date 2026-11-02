"""PR1 cosine scope adapter. Reuse PR0 arithmetic without editing old SOURCE."""
import ast
from fractions import Fraction as Q
from pathlib import Path
from types import SimpleNamespace

from . import h4_arb_pr0_cosine as OLD
from . import h4_arb_pr0_preparation as P0
from . import h4_r1_order2 as R1

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/non_df_h4_central_arb_pr1_cosine/2026-10-25'
PREP=BASE
BASE_COMMIT='e90d6cfed1721b59c33c22bb146aebcc1fea3973'
B,C=P0.B,P0.C
TID=R1.TID
QPY_SHA='e92515770ad16635153905e19c9b37e5c644f44884f46cdc5c2afe91b7b280df'
QPY_FP='e7411e76d88c7f02f9b7d3313f81adf58b412cd5fe5a284da6071dc65a704aea'
STATE_HASH=OLD.STATE_HASH
GUARD,SCALE=OLD.GUARD,OLD.SCALE
PRECISIONS,LIMITS,NONFINITE=OLD.PRECISIONS,dict(OLD.LIMITS),OLD.NONFINITE
finite_schedule=OLD.finite_schedule
FORBIDDEN=dict(new_compile=0,sampling=0,Hamiltonian_state_generation=0,input_QPY_state_actions=0,
    PF=0,PR0=0,additional_trajectory=0,sine=0,DF=0,R2=0,N_G_reanalysis=0,formal_certificate_acceptance=0)


def _pr1_orders(events):
    """Exact saved order profile; not an arbitrary PR trajectory adapter."""
    return ([e['taylor_order'] for e in events]==[0]*30+[2,0]
        and events[30]['product_component_ids']==['43','4']
        and events[30]['rotation_component_id']=='72')


class ScopeStrings(ast.NodeTransformer):
    """Change only scope/diagnostic strings. Numeric expressions are untouched."""
    def visit_Constant(self,node):
        if isinstance(node.value,str):node.value=node.value.replace('PR0','PR1')
        return node


class PlanScope(ScopeStrings):
    def visit_Compare(self,node):
        node=self.generic_visit(node)
        if (isinstance(node.left,ast.Subscript) and isinstance(node.left.slice,ast.Constant)
                and node.left.slice.value=='trajectory_index' and len(node.comparators)==1):
            C.require(ast.literal_eval(node.comparators[0])==0,'REUSED_PR0_INDEX_GUARD')
            node.comparators[0]=ast.Constant(1)
        return node

    def visit_Call(self,node):
        node=self.generic_visit(node)
        if (isinstance(node.func,ast.Name) and node.func.id=='all' and len(node.args)==1
                and isinstance(node.args[0],ast.GeneratorExp)):
            gen=node.args[0]
            if (isinstance(gen.elt,ast.Compare) and isinstance(gen.elt.left,ast.Subscript)
                    and isinstance(gen.elt.left.slice,ast.Constant) and gen.elt.left.slice.value=='taylor_order'):
                C.require(ast.literal_eval(gen.elt.comparators[0])==0,'REUSED_PR0_ORDER_GUARD')
                return ast.Call(func=ast.Name(id='_pr1_orders',ctx=ast.Load()),args=[gen.generators[0].iter],keywords=[])
        return node


def validate_plan(plan):
    _base_validate(plan)
    if plan['domain']=='SAVED_PR1_NO_EXECUTION':
        C.require(plan['trajectory_id']==TID and plan['event_record_hash']==R1.EHASH
            and plan['trajectory_record_hash']==R1.THASH,'FIXED_PR1_PLAN_IDENTITY')
        terms=[plan['components'][i] for i in ('43','4','72')]
        C.require([t['label'] for t in terms]==['IYXXYIII','IIIZIIII','YYIIIIXX']
            and [t['sign'] for t in terms]==[1,-1,-1],'FIXED_PR1_ORDER2_LABELS_SIGNS')
        C.require(Q(plan['events'][30]['rotation_angle_exact'])==Q(.029672025283184623)
            and Q(plan['events'][30]['order_phase_exact'])==-1,'FIXED_PR1_ORDER2_ANGLE_PHASE')
    return plan


def build_plan(model,join,*,domain='SAVED_PR1_NO_EXECUTION',fixture_id=None):
    if domain=='SAVED_PR1_NO_EXECUTION':R1.check_event_semantics(model,join)
    plan=_plan_env['build_plan'](model,join,domain=domain,fixture_id=fixture_id)
    return validate_plan(plan)


# Only index/order scope changes in the metadata builders; only a domain label
# changes in the shared numeric kernel. C exact operands/unitary arithmetic stay.
_tree=ast.parse((ROOT/'src/trotterlib/h4_arb_pr0_preparation.py').read_text())
_nodes=[n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name in ('build_plan','validate_plan','_selected_unitary_enclosure')]
C.require(len(_nodes)==3,'REUSED_SELECTED_FUNCTIONS')
_plan_env=dict(vars(P0),TID=TID,_pr1_orders=_pr1_orders)
exec(compile(ast.fix_missing_locations(PlanScope().visit(ast.Module(body=_nodes,type_ignores=[]))),
    '<PR0 selected kernel with explicit PR1 scope>','exec'),_plan_env)
# Builders and the kernel call the stronger validator; retain the scoped base
# validator separately to avoid recursion.
_base_validate=_plan_env['validate_plan']
_kernel=_plan_env['_selected_unitary_enclosure']
_build=_plan_env['build_plan']
_plan_env['validate_plan']=validate_plan


def _selected_unitary_enclosure(plan,raw,bits,*,progress=None):
    return _kernel(plan,raw,bits,progress=progress)


# Do not expose a real H4 call through an old artificial entry point.
P=SimpleNamespace(**{**vars(P0),'TID':TID,'build_plan':build_plan,'validate_plan':validate_plan,
    '_selected_unitary_enclosure':_selected_unitary_enclosure})

# Reuse production pair/schedule/reduction/authority checks. Only target strings
# change; numeric expressions, nonrefundable reservations and waits are identical.
_tree=ast.parse((ROOT/'src/trotterlib/h4_arb_pr0_cosine.py').read_text())
_names=('validate_contract','load_inputs','_action','action_pair','reduce_pair','run')
_nodes=[n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name in _names]
C.require(len(_nodes)==len(_names),'REUSED_PR0_PRODUCTION_FUNCTIONS')
_run_env={**vars(OLD),**globals()}
exec(compile(ast.fix_missing_locations(ScopeStrings().visit(ast.Module(body=_nodes,type_ignores=[]))),
    '<PR0 pair controller with PR1 scope>','exec'),_run_env)
for _name in _names:globals()[_name]=_run_env[_name]
