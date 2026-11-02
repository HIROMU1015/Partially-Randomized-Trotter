"""Saved-field oracle metrics. Registered execution requires a consumed receipt."""
from fractions import Fraction as F
import math
from .c1_saved_oracle_spec_v1 import FIXED, SCORES, ApprovalError
from .c1_saved_oracle_schema_v1 import validate

THRESHOLD = F(19,20)

def exact(x):
    return F.from_float(x) if type(x) is float else F(x)

def guard(x):
    lo,hi = math.nextafter(float(x), -math.inf), math.nextafter(float(x), math.inf)
    if not math.isfinite(hi) or lo <= 0:
        return None
    return exact(lo),exact(hi)

def minimum(c, methods):
    rows = [(i,r) for i,r in enumerate(c['rows']) if r['method'] in methods and r['eligible']]
    if not rows:
        return None
    i,r = min(rows,key=lambda t:exact(t[1]['conditional_quantum_cost']))
    v = exact(r['conditional_quantum_cost'])
    return {'row_index':i,'method':r['method'],'h_index':r['h_index'],
            'cost':r['conditional_quantum_cost'],'exact_cost':str(v),
            'tied_row_indices':[n for n,x in rows if exact(x['conditional_quantum_cost'])==v]}

def comparison(a,b):
    if a is None or b is None:
        return {'exact_ratio':None,'rounding_guard':None,'gate':'UNDETERMINED','reason':'NO_ELIGIBLE_POINT'}
    ratio=exact(a['cost'])/exact(b['cost']); ga,gb=guard(a['cost']),guard(b['cost'])
    if ga is None or gb is None:
        return {'exact_ratio':str(ratio),'rounding_guard':None,'gate':'UNDETERMINED','reason':'FLOAT_GUARD_UNAVAILABLE'}
    low,high=ga[0]/gb[1],ga[1]/gb[0]
    gate='ADVANTAGE' if high<THRESHOLD else 'NO_ADVANTAGE' if low>=THRESHOLD else 'UNDETERMINED'
    return {'exact_ratio':str(ratio),'rounding_guard':[str(low),str(high)],'gate':gate,
            'reason':'ONE_ULP_GATE_BOUNDARY' if gate=='UNDETERMINED' else None}

def context_metrics(c,seal,spec):
    cm=minimum(c,spec.macro);cb=minimum(c,spec.single+spec.s4+spec.c2);ck=minimum(c,spec.single+spec.macro)
    macro=comparison(cm,cb); selections={}; tags=[];unresolved=[]
    if macro['gate']=='NO_ADVANTAGE':tags.append('A_NO_ORACLE_5PCT_ADVANTAGE')
    elif macro['gate']=='UNDETERMINED':tags.append('UNDETERMINED')
    else:tags.append('KNOWN_COMPOSITION_EQUIVALENT_ORACLE_ADVANTAGE')
    st=seal[c['input_id']][str(c['price'])]['states'][c['state']]
    for score in SCORES:
        front=c['frontiers'][score+':C1']; point=None
        if front is not None:
            ri=next(i for i,r in enumerate(c['rows']) if (r['method'],r['h_index'])==(front['method'],front['h_index']))
            point={'row_index':ri,'method':front['method'],'h_index':front['h_index'],'cost':front['conditional_quantum_cost']}
        reg=None if point is None or ck is None else exact(point['cost'])/exact(ck['cost'])-1
        if reg is not None and reg<0:raise ValueError('NEGATIVE_METHOD_REGRET')
        against=comparison(point,cb)
        missed=macro['gate']=='ADVANTAGE' and (point is None or against['gate']=='NO_ADVANTAGE')
        selections[score]={'sealed_method':st['choices'][score]['C1'],'saved_oracle_h_point':point,
                           'method_regret':str(reg) if reg is not None else None,
                           'regret_null_reason':'NO_ELIGIBLE_SELECTED_OR_CATALOG_POINT' if reg is None else None,
                           'against_baseline':against,'misses_oracle_advantage':missed,
                           'precision_choice_metadata':c['coefficient_only_precision_choices']['C1'] if score=='joint' else None}
    if any(v['misses_oracle_advantage'] for v in selections.values()):tags.append('B_ORACLE_ADVANTAGE_SELECTOR_MISS')
    if macro['gate']=='UNDETERMINED':unresolved.append('macro_ratio:'+macro['reason'])
    for score,v in selections.items():
        if v['method_regret'] is None:unresolved.append('method_regret:'+score+':'+v['regret_null_reason'])
        if v['against_baseline']['gate']=='UNDETERMINED':unresolved.append('selector_ratio:'+score+':'+v['against_baseline']['reason'])
    if unresolved and 'UNDETERMINED' not in tags:tags.append('UNDETERMINED')
    return {'context':{k:c[k] for k in ('input_id','price','state','epsilon')},
            'C_M':cm,'C_B':cb,'C_K':ck,'macro_ratio':macro,'selectors':selections,
            'classifications':tags,'unresolved_parts':unresolved,'known_composition_equivalent':True,
            'eligible_source':'stored_boolean_unchanged','method_regret_scope':'saved_method_only_h_already_oracle'}

def diagnose(comparisons,seal,spec=FIXED,receipt=None):
    if not spec.fixture:
        if not receipt or receipt.get('attempt_consumed') is not True or receipt.get('scope')!='QC_C1_SAVED_ORACLE_POSTHOC_V1_20261014':
            raise ApprovalError('SEPARATE_CONSUMED_POSTHOC_APPROVAL_REQUIRED')
        if receipt.get('preflight',{}).get('status')!='IDENTITY_SCHEMA_PREFLIGHT_ONLY_PASS':
            raise ApprovalError('FROZEN_PREFLIGHT_RECEIPT_REQUIRED')
        from .c1_saved_oracle_execution_v1 import approval
        approval(receipt.get('authorization'),receipt['preflight']['identities'])
    validate(comparisons,seal,spec)
    return {'schema':'C1_SAVED_ORACLE_DIAGNOSTIC_V1','mode':'SYNTHETIC' if spec.fixture else 'REGISTERED_SAVED_POSTHOC',
            'contexts':[context_metrics(c,seal,spec) for c in comparisons],
            'arithmetic':'exact stored binary64 fractions;one-neighbor-ULP gate sensitivity only,not physical confidence bounds',
            'tie_rule':'first stored row;all exactly equal-cost row indices retained',
            'threshold':'19/20 strict','new_eigensolver_PF_compile_calls':0,
            'full_ground_energy_cost':'UNKNOWN','novelty_adopted':False,'mandatory_stop':True}
