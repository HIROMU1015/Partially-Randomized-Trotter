"""Schema/pointer checks only. No new minima, ratios, regret, or eligibility."""
import math
from .c1_saved_oracle_spec_v1 import InputError, FIXED, SCORES, LABELS, LIMITS

CTX_KEYS = {'input_id', 'price', 'state', 'epsilon', 'PF_budget', 'statistical_budget',
            'rows', 'frontiers', 'coefficient_only_precision_choices', 'classifications',
            'full_ground_energy_cost', 'novelty_adopted'}
ROW_KEYS = {'method', 'h_index', 'eligible', 'g_per_time', 'conditional_quantum_cost',
            'certified_PF_bound'}

def require(ok, code):
    if not ok:
        raise InputError(code)

def number(x, nullable=False):
    if x is None and nullable:
        return
    require(type(x) in (int, float) and math.isfinite(x) and x > 0, 'POSITIVE_FINITE_NUMBER')

def validate(comparisons, seal, spec=FIXED):
    if spec.fixture:
        require(all(s.startswith('F_') for s in spec.inputs), 'FIXTURE_IDS_ONLY')
    else:
        require(spec == FIXED, 'FIXED_SCOPE')
    require(isinstance(comparisons, list) and isinstance(seal, dict), 'ROOT_SCHEMA')
    expected = {(i,p,s,e) for i in spec.inputs for p in spec.prices
                for s in spec.states for e in spec.targets}
    require(len(comparisons) == len(expected) <= LIMITS['contexts'], 'CONTEXT_COVERAGE')
    require(set(seal) == set(spec.inputs), 'SEAL_INPUTS')
    seen = set(); nrows = 0
    expected_cells = {(m,h) for m in spec.methods for h in range(spec.h_levels)}
    frontier_keys = {f'{s}:{l}' for s in SCORES for l in LABELS} | {
        'oracle_diagnostic:all_single_subdivision', 'oracle_diagnostic:S4', 'oracle_diagnostic:C2'}
    for ci, c in enumerate(comparisons):
        require(isinstance(c,dict) and set(c) == CTX_KEYS, 'CONTEXT_FIELDS')
        require(type(c['price']) is int and type(c['epsilon']) in (int,float) and isinstance(c['input_id'],str) and isinstance(c['state'],str), 'CONTEXT_TYPES')
        key = (c['input_id'],c['price'],c['state'],c['epsilon'])
        require(key in expected and key not in seen, 'CONTEXT_IDENTITY_OR_DUPLICATE'); seen.add(key)
        require(c['PF_budget'] == c['epsilon']/2 and c['statistical_budget'] == c['epsilon']/2,
                'SAVED_BUDGET_IDENTITY')
        require(c['full_ground_energy_cost'] == 'UNKNOWN' and c['novelty_adopted'] is False,
                'CLAIM_SEMANTICS')
        require(isinstance(c['classifications'],list) and all(isinstance(x,str) for x in c['classifications']),
                'SAVED_CLASSIFICATIONS_SCHEMA')
        require(isinstance(c['rows'],list) and len(c['rows']) == len(expected_cells), 'ROW_COVERAGE')
        lookup = {}; nrows += len(c['rows'])
        for ri, row in enumerate(c['rows']):
            require(isinstance(row,dict) and set(row) == ROW_KEYS, 'ROW_FIELDS')
            require(type(row['h_index']) is int and type(row['eligible']) is bool, 'ROW_TYPES')
            cell = (row['method'],row['h_index'])
            require(cell in expected_cells and cell not in lookup, 'METHOD_H_OR_DUPLICATE')
            number(row['conditional_quantum_cost'], nullable=True); number(row['g_per_time'], nullable=True)
            require(not row['eligible'] or row['conditional_quantum_cost'] is not None, 'ELIGIBLE_NULL_COST')
            require(row['certified_PF_bound'] is None or isinstance(row['certified_PF_bound'],str), 'SAVED_BOUND_TYPE')
            lookup[cell] = (ri,row)
        require(set(lookup) == expected_cells, 'MISSING_METHOD_H')
        require(isinstance(c['frontiers'],dict) and set(c['frontiers']) == frontier_keys, 'FRONTIER_KEYS')
        for frontier in c['frontiers'].values():
            if frontier is not None:
                require(isinstance(frontier,dict) and set(frontier)==ROW_KEYS, 'FRONTIER_FIELDS')
                require(type(frontier['h_index']) is int and type(frontier['eligible']) is bool, 'FRONTIER_TYPES')
                number(frontier['conditional_quantum_cost'])
                cell = (frontier['method'],frontier['h_index'])
                require(cell in lookup and frontier == lookup[cell][1] and frontier['eligible'], 'FRONTIER_POINTER')
        try:
            st = seal[c['input_id']][str(c['price'])]['states'][c['state']]
        except (KeyError,TypeError):
            raise InputError('SEAL_CONTEXT_POINTER') from None
        require(isinstance(st,dict) and {'choices','coefficient_only_precision_choices','strong_known_composition_equals_C1_by_construction'} <= set(st), 'SEAL_FIELDS')
        require(st['strong_known_composition_equals_C1_by_construction'] is True, 'KNOWN_CONTROL_SEMANTICS')
        require(set(st['choices']) == set(SCORES), 'SEAL_SCORES')
        for score in SCORES:
            choices = st['choices'][score]
            require(set(choices) == set(LABELS), 'SEAL_CHOICE_LABELS')
            require(choices['C1'] == choices['strong_known_composition'] and choices['C1'] in spec.single+spec.macro,
                    'KNOWN_CONTROL_CHOICE')
            selected = c['frontiers'][score+':C1']
            require(selected == c['frontiers'][score+':strong_known_composition'], 'KNOWN_CONTROL_FRONTIER')
            require(selected is None or selected['method'] == choices['C1'], 'SAVED_SELECTOR_METHOD')
        cps = c['coefficient_only_precision_choices']
        require(isinstance(cps,dict) and set(cps) == set(LABELS), 'PRECISION_LABELS')
        for point in cps.values():
            require(isinstance(point,dict) and set(point) == {'method','h_index'} and point['method'] in spec.single+spec.macro,
                    'PRECISION_POINTER')
            require(point['h_index'] is None or type(point['h_index']) is int and 0<=point['h_index']<spec.h_levels,
                    'PRECISION_H_INDEX')
    require(seen == expected and nrows <= LIMITS['rows'], 'INPUT_COVERAGE_CAP')
    return {'status':'SCHEMA_IDENTITY_ONLY_PASS','contexts':len(seen),'rows':nrows,
            'new_minima_ratios_regrets':0,'eligibility_recomputed':False}
