"""Independent saved-field recomputation. Does not import diagnostic evaluator."""
from fractions import Fraction
import math
from .c1_saved_oracle_spec_v1 import FIXED, SCORES, ApprovalError, InputError
from .c1_saved_oracle_schema_v1 import validate


def verify_result(data,seal,result,spec=FIXED,receipt=None):
    if not spec.fixture and (not receipt or not receipt.get('attempt_consumed') or
        receipt.get('scope')!='QC_C1_SAVED_ORACLE_POSTHOC_V1_20261014' or
        receipt.get('authorization',{}).get('status')!='APPROVED'):
        raise ApprovalError('INDEPENDENT_AUDIT_REQUIRES_POSTHOC_LAUNCH_RECEIPT')
    if not spec.fixture:
        if receipt.get('preflight',{}).get('status')!='IDENTITY_SCHEMA_PREFLIGHT_ONLY_PASS':
            raise ApprovalError('FROZEN_PREFLIGHT_RECEIPT_REQUIRED')
        from .c1_saved_oracle_execution_v1 import approval
        approval(receipt.get('authorization'),receipt['preflight']['identities'])
    validate(data,seal,spec)
    def insist(x,code):
        if not x:raise InputError(code)
    def pool(c, allowed):
        ordered=sorted(((Fraction(r['conditional_quantum_cost']),n,r) for n,r in enumerate(c['rows'])
                        if r['eligible'] and r['method'] in set(allowed)),key=lambda x:(x[0],x[1]))
        if not ordered:return None
        val,n,row=ordered[0]
        return {'row_index':n,'method':row['method'],'h_index':row['h_index'],
                'cost':row['conditional_quantum_cost'],'exact_cost':str(val),
                'tied_row_indices':[i for v,i,r in ordered if v==val]}
    def gate(a,b):
        if a is None or b is None:
            return {'exact_ratio':None,'rounding_guard':None,'gate':'UNDETERMINED','reason':'NO_ELIGIBLE_POINT'}
        loA,hiA=[math.nextafter(float(a['cost']),d) for d in (-math.inf,math.inf)]
        loB,hiB=[math.nextafter(float(b['cost']),d) for d in (-math.inf,math.inf)]
        ratio=Fraction(a['cost'])/Fraction(b['cost'])
        if not all(math.isfinite(x) and x>0 for x in (loA,hiA,loB,hiB)):
            return {'exact_ratio':str(ratio),'rounding_guard':None,'gate':'UNDETERMINED','reason':'FLOAT_GUARD_UNAVAILABLE'}
        lower=Fraction(loA)/Fraction(hiB);upper=Fraction(hiA)/Fraction(loB)
        status='ADVANTAGE' if Fraction(hiA)*20<Fraction(loB)*19 else (
            'NO_ADVANTAGE' if Fraction(loA)*20>=Fraction(hiB)*19 else 'UNDETERMINED')
        return {'exact_ratio':str(ratio),'rounding_guard':[str(lower),str(upper)],'gate':status,
                'reason':'ONE_ULP_GATE_BOUNDARY' if status=='UNDETERMINED' else None}
    expected=[]
    for c in data:
        M=pool(c,spec.macro);B=pool(c,spec.single+spec.s4+spec.c2);K=pool(c,spec.single+spec.macro)
        mg=gate(M,B)
        tags={'ADVANTAGE':['KNOWN_COMPOSITION_EQUIVALENT_ORACLE_ADVANTAGE'],
              'NO_ADVANTAGE':['A_NO_ORACLE_5PCT_ADVANTAGE'],
              'UNDETERMINED':['UNDETERMINED']}[mg['gate']]
        sel={};st=seal[c['input_id']][str(c['price'])]['states'][c['state']]
        for score in SCORES:
            method=st['choices'][score]['C1'];chosen=pool(c,(method,))
            frontier=c['frontiers'][score+':C1']
            insist((chosen is None and frontier is None) or (chosen is not None and frontier is not None and
                    (chosen['method'],chosen['h_index'],chosen['cost'])==
                    (frontier['method'],frontier['h_index'],frontier['conditional_quantum_cost'])),
                   'SAVED_SELECTED_FRONTIER_NOT_FIRST_MINIMUM')
            point=None if chosen is None else {k:chosen[k] for k in ('row_index','method','h_index','cost')}
            regret=None if point is None or K is None else Fraction(point['cost'])/Fraction(K['cost'])-1
            insist(regret is None or regret>=0,'REGRET_SIGN')
            sg=gate(point,B);miss=mg['gate']=='ADVANTAGE' and (point is None or sg['gate']=='NO_ADVANTAGE')
            sel[score]={'sealed_method':method,'saved_oracle_h_point':point,
                        'method_regret':str(regret) if regret is not None else None,
                        'regret_null_reason':'NO_ELIGIBLE_SELECTED_OR_CATALOG_POINT' if regret is None else None,
                        'against_baseline':sg,'misses_oracle_advantage':miss,
                        'precision_choice_metadata':c['coefficient_only_precision_choices']['C1'] if score=='joint' else None}
        if any(x['misses_oracle_advantage'] for x in sel.values()):tags.append('B_ORACLE_ADVANTAGE_SELECTOR_MISS')
        unresolved=[]
        if mg['gate']=='UNDETERMINED':unresolved.append('macro_ratio:'+mg['reason'])
        for score,x in sel.items():
            if x['method_regret'] is None:unresolved.append('method_regret:'+score+':'+x['regret_null_reason'])
            if x['against_baseline']['gate']=='UNDETERMINED':unresolved.append('selector_ratio:'+score+':'+x['against_baseline']['reason'])
        if unresolved and 'UNDETERMINED' not in tags:tags.append('UNDETERMINED')
        expected.append({'context':{k:c[k] for k in ('input_id','price','state','epsilon')},
                         'C_M':M,'C_B':B,'C_K':K,'macro_ratio':mg,'selectors':sel,'classifications':tags,
                         'unresolved_parts':unresolved,'known_composition_equivalent':True,'eligible_source':'stored_boolean_unchanged',
                         'method_regret_scope':'saved_method_only_h_already_oracle'})
    insist(set(result)=={'schema','mode','contexts','arithmetic','tie_rule','threshold','new_eigensolver_PF_compile_calls',
                          'full_ground_energy_cost','novelty_adopted','mandatory_stop'},'OUTPUT_SCHEMA')
    insist(result['schema']=='C1_SAVED_ORACLE_DIAGNOSTIC_V1' and result['threshold']=='19/20 strict','OUTPUT_CONTRACT')
    insist(result['mode']==('SYNTHETIC' if spec.fixture else 'REGISTERED_SAVED_POSTHOC'),'OUTPUT_MODE')
    insist(result['contexts']==expected,'DIAGNOSTIC_RECOMPUTATION_MISMATCH')
    insist(result['arithmetic']=='exact stored binary64 fractions;one-neighbor-ULP gate sensitivity only,not physical confidence bounds' and
           result['tie_rule']=='first stored row;all exactly equal-cost row indices retained','ARITHMETIC_SEMANTICS')
    insist(result['new_eigensolver_PF_compile_calls']==0 and result['full_ground_energy_cost']=='UNKNOWN' and
           result['novelty_adopted'] is False and result['mandatory_stop'] is True,'CLAIM_BOUNDARY')
    return {'status':'INDEPENDENT_SAVED_FIELD_AUDIT_PASS','contexts_checked':len(expected),
            'mode':result['mode'],'new_eigensolver_PF_compile_calls':0,
            'external_scientific_reproduction':False,'mandatory_stop':True}
