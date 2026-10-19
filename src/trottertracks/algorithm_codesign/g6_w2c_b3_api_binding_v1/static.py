"""Read bytes/AST only. Scientific names are data, never import targets."""
import ast
import hashlib
import json
import sys
from pathlib import Path
from .. import g6_w2c_b3_bundle_contract_v1 as K

DOC = 'docs/research/track_b_g6_w2c_b3_real_api_binding_preflight_candidate_20261019/'
PKG = K.SOURCE + 'g6_w2c_b3_api_binding_v1/'
BASE = 'd24cccbecdf0928fe977080dd7b28c591df0089e'
INDEPENDENT = '6bffb5b5819918698b522270ff19b255766d3072'
ARMS = ('ordinary', 'partial_return_tail', 'closed_P3_tail', 'full_return')
SCIENCE = ('g7_generator', 'g7_provider', 'return_aggregation',
           'g6_f3_contract', 'g6_f3_precision', 'g6_f3_generator')
STATIC_MODULES = SCIENCE + ('g6_w2c_b3_driver', 'g6_w2c_cost_preparation')
SEED = 'ARTIFICIAL_API_BINDING_V1'
SCHEMA = 'b3_closed_api_binding_candidate_v1'

class Rejected(RuntimeError):
    pass

def need(value, reason):
    if not value:
        raise Rejected(reason)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()

def digest(value):
    return sha(encoded(value))

def runtime():
    import dataclasses,importlib.abc,hashlib,multiprocessing.pool,pickle,resource,threading
    result=K.runtime()
    for module in (ast,dataclasses,importlib.abc,hashlib,multiprocessing.pool,pickle,resource,threading):
        p=Path(module.__file__).resolve()
        result['libraries'][module.__name__]=dict(path=str(p),sha256=sha(p.read_bytes()))
    return result

def inventory(root):
    """Every class/function, including private/nested names, with source locations.

    Return expressions are static syntax, not inferred dynamic schemas. The
    explicit mapping below supplies reviewed semantic boundaries.
    """
    result = {}
    for module in STATIC_MODULES:
        path = K.SOURCE + module + '.py'
        raw = (Path(root) / path).read_bytes()
        tree = ast.parse(raw, filename=path)
        rows = []
        def visit(node, prefix=''):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    name = prefix + child.name
                    row = dict(name=name, line=child.lineno, end_line=child.end_lineno,
                               kind=type(child).__name__, doc=ast.get_docstring(child),
                               source_sha256=sha(ast.get_source_segment(raw.decode(), child).encode()))
                    if hasattr(child, 'args'):
                        row.update(signature=ast.unparse(child.args),
                                   returns_annotation=ast.unparse(child.returns) if child.returns else None,
                                   return_syntax=sorted({ast.unparse(x.value) for x in ast.walk(child)
                                                        if isinstance(x, ast.Return) and x.value}),
                                   raise_syntax=sorted({ast.unparse(x.exc) for x in ast.walk(child)
                                                       if isinstance(x, ast.Raise) and x.exc}))
                    rows.append(row)
                    visit(child, name + '.')
                else:
                    visit(child, prefix)
        visit(tree)
        result[path] = dict(sha256=sha(raw), bytes=len(raw), entries=rows,
                           analysis='AST_ONLY_NO_IMPORT_NO_EVALUATION')
    return dict(schema='b3_static_scientific_API_inventory_v1', modules=result,
                semantics=binding(), dynamic_type_inference=False)

def binding():
    return dict(schema='b3_static_API_port_mapping_v1', dispatch_implemented=False,
        arms={arm: dict(layer_A='g7_generator.make_generator -> ' +
                       ('FullReturnGenerator.sample' if arm == 'full_return' else 'CanonicalGenerator.sample'),
                       layer_B=('g6_f3_generator.prepare -> GeneratorSession.sample/angle_for_event'
                                if arm == 'full_return' else
                                'g6_w2c_b3_driver.LegacyEndpoint: canonical_snapshot/AuditParity/event_witness; '
                                'CanonicalGenerator.event(group_index, raw_word, child) requery; '
                                'g6_f3_precision.enclose_atan'),
                       axes=['X', 'Y'], axis_semantics='execution identity and independent bits domain; not a G7 parameter')
              for arm in ARMS},
        request=dict(G7='p tuple[Fraction], x Fraction, m int, arm str; root_bits/probability_bits int; eta/rho Fraction',
                     F3='make_request exact int/Fraction/{num:int,den:positive int}; no float; angle_tau/root_bits/terms supplied',
                     sigma='provider/angle context only; G7 event retains symbolic tangent',
                     identity='GeneratorSession.sample(bits=None, *, trial_id); x=0 IDENTITY outside fixed positive input'),
        result=dict(G7='dict event or None; None only Full raw-nonreduced/coin-reject; Canonical has no stochastic zero',
                    event_fields=['word','raw_word','child','ratio','phase_i_power','coefficient','proposal',
                                  'weight','provider_calls','helper','event_unitary','circuit_time_word'],
                    exact_values='tuple[int] words; Fraction ratio/coefficient/proposal/weight; phase int; provider_calls int map',
                    F3='Evidence(document,digest), schema g6_f3_trial_v1, EVENT/ZERO/IDENTITY; trace and context IDs',
                    bridge='driver Outcome(EVENT|ZERO_R|ZERO_C,event,evidence); run_pair emits canonical chain and F01'),
        failures=dict(G7='ValueError for input/law/precision/support; never reinterpret as zero',
                      F3='F3Failure(category,reason,stage,input_id,details), immutable evidence, terminal session',
                      driver='terminal endpoint; F01 completed prefix only, retry0/resume false'),
        full_boundaries=dict(setup='ReturnKernel.__init__: first_passage/returns O(L*m^2), held fixed state',
            sample='order -> raw labels -> reduced-word test -> packet only if raw valid -> acceptance -> child only if accepted',
            local='packet -> digital_parent -> parent -> word_series/convolution/_mass; root/envelope/child law/weight',
            requery='GeneratorSession.angle_for_event -> generator.event -> packet then event fingerprint and enclose_atan',
            state='no growing packet cache; sample avoids second local query; angle intentionally requeries',
            rational='Fraction C arithmetic is attributed to Python caller, not independently metered bit operations'),
        codec=dict(F3='hex $rat/$int/$tuple/$map exact Evidence document',
                   driver='cost_preparation canonical exact representation after driver.exact',
                   compatibility='related exact codecs have distinct documents; never assume bytes interchangeable'),
        side_effects=dict(science='constructors allocate held rational state; supplied bits consume state; F3 failure stops session; source guards read files',
                          provider='IR only; cache lookup in bind_synthesized_provider_ir; excluded from this release',
                          adapter='ARTIFICIAL only; no law/evidence equivalence to real APIs'),
        canonical_F3_prepare='positive canonical arms explicitly rejected by F3.prepare; existing driver overlay required',
        bundle_injection='FIXED_BUNDLE_HAS_NO_EXTERNAL_ENDPOINT_PORT; future separate formal SOURCE required')

def monitoring():
    existing = ['parent_cpu','all_worker_cpu','outer_global_wall','condition_wall',
                'setup_online','generator_plus_audit','angle_plus_requery','serialization_hash',
                'write_flush_close','IPC','observer_call_return','coarse_fine_separate',
                'own_RSS_HWM','own_IO','physical_output_bytes','inflight_STOP_drain','UNKNOWN_credit']
    return dict(schema='b3_API_monitoring_candidate_v1',
        metrics={**{x:dict(availability='EXISTING_INSTRUMENTATION', actual_science_value=None)
                    for x in existing},
                 'port_bit_consumption':dict(availability='ADDITIONAL_ARTIFICIAL_ADAPTER', actual_science_value=None),
                 'full_local_coefficient_child_law':dict(availability='AFTER_REAL_BINDING_FUNCTION_OBSERVER',actual_science_value=None),
                 'actual_path_distribution':dict(availability='AFTER_REAL_BINDING',actual_science_value=None),
                 'rational_bit_operation_cost':dict(availability='CURRENTLY_UNIDENTIFIABLE',actual_science_value=None),
                 'isolated_observer_overhead':dict(availability='CURRENTLY_UNIDENTIFIABLE',actual_science_value=None),
                 'unique_continuous_peak_RAM':dict(availability='CURRENTLY_UNIDENTIFIABLE',actual_science_value=None)},
        attribution='parent + all workers once; pair CPU/stages not readded; global wall is outer interval, not summed pair walls',
        function_resolution='Python call/return inclusive/exclusive alternatives; local query and child_law function boundaries available; Fraction internals remain caller cost',
        STOP='parent arrival order; no new ENTER; pre-STOP inflight drains; START_RESERVED != body; unknown never refunded',
        transient_pressure='observe/reestablish monitoring; do not kill science solely for host PSI/other worktree changes')

def candidate(root):
    root = Path(root).resolve()
    old = K.candidate(root)
    paths = set(old['source_pins'])
    paths.update(str(p.relative_to(root)) for p in (root/PKG).glob('*.py'))
    paths.update([DOC+'run_acceptance.py', DOC+'build_static.py',
                  DOC+'run_checks.py',
                  'tests/tracks/algorithm_codesign/test_g6_w2c_b3_api_binding_v1.py'])
    pins = {p:sha((root/p).read_bytes()) for p in sorted(paths)}
    ref = old['scientific_reference']
    return dict(schema=SCHEMA, **K.CLOSED, status='UNADOPTED_EXECUTION_CLOSED',
                base_commit=BASE, independent_reference=INDEPENDENT,
                runtime_root=str(root), python=runtime(), source_pins=pins,
                source_release_candidate=digest(pins),
                scientific_reference=ref, scientific_reference_sha256=digest(ref),
                static_api_inventory_digest=digest(inventory(root)),
                proposal=old['proposal'], api_binding=binding(), monitoring=monitoring(),
                artificial_seed_namespace=SEED, production_dispatch=False,
                integration='STATIC_PORT_AND_FIXED_PAIR_ENGINE; UNCHANGED_BUNDLE_SEPARATE_REGRESSION',
                authority_policy=dict(real_authority=False, fresh_fake_only=True, retry=0, resume=False,
                    global_scope=True, atomic_claim=True, commit_ACK_unknown_spent=True,
                    non_refundable=True, STOP=True),
                required_future_release='new adopted source with endpoint injection and real bits/authority; no boolean unlock')

def preflight(c, root):
    need(not [n for n in sys.modules if n.rsplit('.',1)[-1] in SCIENCE], 'SCIENCE_ALREADY_LOADED')
    need(Path(__file__).resolve()==Path(root).resolve()/PKG/'static.py', 'LOADED_SOURCE_LOCATION')
    need(type(c) is dict and c.get('schema') == SCHEMA, 'UNSUPPORTED_SCHEMA')
    for key, value in K.CLOSED.items():
        need(type(c.get(key)) is type(value) and c.get(key) == value, 'CLOSED_POLICY:'+key)
    need(c == candidate(root), 'SOURCE_INPUT_RUNTIME_API_CONTRACT_MISMATCH')
    for path,h in c['source_pins'].items():
        need(sha((Path(root)/path).read_bytes()) == h, 'SOURCE_HASH:'+path)
    for path,h in c['scientific_reference']['input_pins'].items():
        need(sha((Path(root)/path).read_bytes())==h,'INPUT_HASH:'+path)
    return dict(schema='b3_static_preflight_v1', candidate_digest=digest(c),
                static_only=True, artificial_only=True, production_ready=False,
                API_inventory_digest=digest(inventory(root)))

def production_entry(*args, **kwargs):
    raise Rejected('NO_SCIENTIFIC_DISPATCH_IN_THIS_SOURCE_RELEASE')

def real_bits(*args, **kwargs):
    raise Rejected('NO_REAL_BITS_PORT')

def real_authority(*args, **kwargs):
    raise Rejected('NO_REAL_AUTHORITY_PORT')
