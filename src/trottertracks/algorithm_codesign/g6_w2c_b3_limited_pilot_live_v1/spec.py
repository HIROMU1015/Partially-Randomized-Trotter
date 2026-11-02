"""Static API data, exact candidate parameters and budgets. Never imports core."""
import ast
from pathlib import Path
from hashlib import sha256
import json
from .guard import require

BASE = 'f6088e2cc453816b30e985e52b10696830844fa9'
SCIENCE = '76ca3c4089b201d55da0fede08e1f77733ac38a5'
INDEPENDENT = '0db83adb370d66a2f7f5254c8037dadda94ecdb8'
SOURCE = 'src/trottertracks/algorithm_codesign/'
PKG = SOURCE + 'g6_w2c_b3_limited_pilot_live_v1/'
DOC = 'docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/'
ARMS = ('ordinary', 'partial_return_tail', 'closed_P3_tail', 'full_return')
CLOSED = dict(content_adopted=False, run_ready=False, contract_grants_execution=False,
              science_release=None, scientific_adoption=None, artificial_only=True,
              mandatory_STOP=True, retry=0, resume=False)
INPUT = dict(id='P5_general_order', p=['1/5', '3/10', '1/2'], x='5/7', m=5, sigma=1)
PRECISION = dict(root_bits=256, probability_bits=160, eta='1/1000000000000',
                 rho='1/1000000000000', epsilon_axis='1/200', primitive_error='1/1000000',
                 alpha_axis='1/320', alpha_familywise='1/20')
ANGLE = dict(tau='1/1000000', K_phi=32, J=32, math_only=True)
SEED = 'PROPOSED_B3_LIMITED_G7_PILOT_20261019_V1'
API = {
 'g7_generator': ['make_generator', 'CanonicalGenerator.__init__', 'CanonicalGenerator.sample',
                  'CanonicalGenerator.event', 'FullReturnGenerator.__init__',
                  'FullReturnGenerator.sample', 'FullReturnGenerator.event', 'FullReturnGenerator.packet',
                  'DeterministicBits.bits'],
 'g6_f3_generator': ['prepare', 'GeneratorSession.sample', 'GeneratorSession.event',
                     'GeneratorSession.angle_for_event', 'GeneratorSession.parent_packet'],
 'g6_f3_contract': ['make_request', 'Evidence.create', 'Evidence.read'],
 'g6_f3_precision': ['enclose_atan'],
 'g6_w2c_b3_driver': ['LegacyEndpoint.__init__', 'LegacyEndpoint.sample', 'LegacyEndpoint.angle_for'],
 'g6_w2c_cost_preparation': ['canonical_snapshot', 'event_witness', 'AuditParity.__init__', 'AuditParity.invoke']}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def digest(value):
    return sha256(encoded(value)).hexdigest()


def route(arm, layer):
    require(arm in ARMS and layer in ('A', 'B'), 'ROUTE_CELL')
    full = arm == 'full_return'
    return dict(arm=arm, layer=layer,
        construction='F3.prepare' if full and layer == 'B' else 'G7.make_generator',
        sample='GeneratorSession.sample(bits, *, trial_id)' if full and layer == 'B'
               else ('FullReturnGenerator.sample(bits)' if full else 'CanonicalGenerator.sample(bits)'),
        audit='F3 sealed preflight + AuditParity' if full and layer == 'B'
              else 'canonical_snapshot + AuditParity + event_witness' if layer == 'B' else None,
        angle='GeneratorSession.angle_for_event(original Evidence)' if full and layer == 'B'
              else 'AuditParity.invoke -> enclose_atan' if layer == 'B' else None,
        requery='FullReturnGenerator.event(word, child) -> packet' if full and layer == 'B'
                else 'CanonicalGenerator.event(group_index, raw_word, child)' if layer == 'B' else None,
        canonical_positive_F3_boundary='CANONICAL_PREFLIGHT_NOT_IMPLEMENTED',
        live_compatibility='NOT_ESTABLISHED', executable_science_loader=False)


def signatures(root):
    result = {}
    for module, wanted in API.items():
        path = SOURCE + module + '.py'
        tree = ast.parse((Path(root)/path).read_bytes(), filename=path)
        def visit(body, prefix=''):
            for node in body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = prefix + node.name
                    if name in wanted:
                        result[module + '.' + name] = dict(path=path, line=node.lineno,
                                                           arguments=ast.unparse(node.args))
                elif isinstance(node, ast.ClassDef):
                    visit(node.body, prefix + node.name + '.')
        visit(tree.body)
        require(all(module + '.' + n in result for n in wanted), 'MISSING_API:' + module)
    return result


def budgets():
    return dict(scheduled_sampling=16896,primary_constructions=48,G7_make_generator=42,F3_prepare=6,
        Full_constructor_inside_prepare=6,canonical_snapshot=18,AuditParity_construction=24,
        B_sample_audit=8448,event_witness=512,nested_B_event_witness=8448,B_angle=8448,B_event_requery=8448,
        canonical_angle=6336,Full_angle=2112,Full_sample_packet=4224,Full_angle_packet=2112,
        bit_draw_requests_static_upper=105600,bit_width=160,additional_sampling=0,
        result_bytes=1<<36,log_bytes=1<<28,nested_internal_total_calls='FIXED_ALGORITHM_ONLY_NOT_UNLIMITED')


def tasks():
    rows=[];offset=0
    for mi,mode in enumerate(('S','P1')):
        for pi,(phase,pass_name,n) in enumerate((('correctness','baseline',16),('profile','baseline',256),('profile','instrumented',256))):
            for i in range(8):
                arm,layer=ARMS[i//2],('A','B')[i%2]
                rows.append(dict(key=f'{mode}__{phase}_{pass_name}_{arm}_{layer}',condition=mode,
                    phase_index=mi*3+pi,phase=phase,pass_name=pass_name,arm=arm,layer=layer,pair_index=i,n=n,offset=offset))
                offset+=2*n
    require(offset==16896,'GLOBAL_PLAN')
    return rows


def scope():
    return dict(schema='b3_limited_pilot_adopted_scope_v1',input=INPUT,precision=PRECISION,angle=ANGLE,
        cells_per_condition_phase=16,correctness_n=16,profile_n=256,modes=['S','P1'],
        scheduled_sampling_calls=16896,phases=6,tasks=tasks(),seed_namespace=SEED,IID_claim=False,
        budgets=budgets(),coverage='FIXED16_ACCEPTED_GE1_ELSE_INCONCLUSIVE_STOP',
        order='S_correctness_baseline_instrumented_THEN_P1_correctness_baseline_instrumented',
        subsequent_execution_authorized=False)
