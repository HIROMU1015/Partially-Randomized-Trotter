"""Static API data, exact candidate parameters and budgets. Never imports core."""
import ast
from pathlib import Path
from hashlib import sha256
import json
from .guard import require

BASE = '80b7bf61fe0a735d4031bb719c4b3e788611c3ba'
SCIENCE = '76ca3c4089b201d55da0fede08e1f77733ac38a5'
INDEPENDENT = '0db83adb370d66a2f7f5254c8037dadda94ecdb8'
SOURCE = 'src/trottertracks/algorithm_codesign/'
PKG = SOURCE + 'g6_w2c_b3_live_preflight_v1/'
DOC = 'docs/research/track_b_g6_w2c_b3_live_api_correctness_preflight_closed_20261020/'
ARMS = ('ordinary', 'partial_return_tail', 'closed_P3_tail', 'full_return')
CLOSED = dict(content_adopted=False, run_ready=False, contract_grants_execution=False,
              science_release=None, scientific_adoption=None, artificial_only=True,
              mandatory_STOP=True, retry=0, resume=False)
INPUT = dict(id='P5_general_order', p=['1/5', '3/10', '1/2'], x='5/7', m=5, sigma=1)
PRECISION = dict(root_bits=256, probability_bits=160, eta='1/1000000000000',
                 rho='1/1000000000000', epsilon_axis='1/200', primitive_error='1/1000000',
                 alpha_axis='1/320', alpha_familywise='1/20')
ANGLE = dict(tau='1/1000000', K_phi=32, J=32, math_only=True)
SEED = 'PROPOSED_B3_CORRECTNESS_ONLY_20261020_V1'
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
    # Bounds on named boundary operations; nested implementation calls are NOT summed.
    return dict(scheduled_sampling=512, primary_constructions=16, G7_make_generator=14,
        F3_prepare=2, Full_constructor_inside_prepare=2, canonical_snapshot=6,
        AuditParity_construction=8, B_sample_audit=256, event_witness=512,
        B_angle=256, B_event_requery=256, Full_sample_packet=128, Full_angle_packet=64,
        canonical_angle=192, Full_angle=64, bit_draw_requests_static_upper=3200,
        bit_width=160, diagnostic_event_queries=0, additional_sampling=0,
        result_bytes=1 << 30, log_bytes=64 << 20,
        nested_internal_total_calls='NOT_ESTABLISHED_NOT_AN_AGGREGATE_CALL_AUTHORIZATION')


def tasks():
    rows = []
    for mi, mode in enumerate(('S', 'P1')):
        for i in range(8):
            arm, layer = ARMS[i//2], ('A', 'B')[i%2]
            rows.append(dict(key=f'{mode}__correctness_baseline_{arm}_{layer}', condition=mode,
                phase_index=mi*3, phase='correctness', pass_name='baseline', arm=arm, layer=layer,
                pair_index=i, n=16, offset=len(rows)*32))
    return rows


def scope():
    return dict(schema='b3_correctness_only_candidate_v1', **CLOSED, input=INPUT,
        precision=PRECISION, angle=ANGLE, cells=16, trials_per_cell=16, modes=['S', 'P1'],
        scheduled_sampling_calls=512, phases=['correctness'], profile_calls=0,
        tasks=tasks(), seed_namespace=SEED, IID_claim=False, budgets=budgets(),
        coverage='INCONCLUSIVE_COVERAGE_IF_ANY_CELL_HAS_NO_ACCEPTED; NO_ADAPTIVE_SAMPLING',
        order='S_THEN_P1', automatic_pilot=False,
        historical=dict(Full_N=31710848, Full_N_correctness=256, pilot=16896))
