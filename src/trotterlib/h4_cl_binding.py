"""C/L transfer proof overlay: bytes, symbolic rules and typed obligations only.

The original40x14 binding checker is reused unchanged. Real-H4 endpoint
strings are copied and syntax-checked, never ordered, added, scaled or parsed
as rational numbers. No NumPy, FLINT, circuit builder or accountant is imported.
"""
import ast
import csv
import importlib.util
import io
from pathlib import Path


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

B=load('cl_frozen_binding','h4_evidence_binding.py')
R=load('cl_transfer_rules','h4_transfer_rules.py')
T=B.T
require=T.require
SCHEMA='H4_CL_TRANSFER_TECHNICAL_OVERLAY_V1'
CATEGORIES=('PROVED_EXISTING','SAVED_EVIDENCE_UNCHECKED','TECHNICAL_IMPLEMENTATION_NEEDED',
            'CERTIFIED_NUMERICS_NEEDED','MISSING_HISTORICAL_EVIDENCE','PROVED_INCOMPATIBLE')
L_SOURCE='eb195bf2dedd0aed9d80ad7f84948b59ea53d854'


def no_execution(action):
    require(action=='VERIFY_SAVED_ONLY','NO_SCIENCE_ACCEPTANCE_COMPILE_OR_NG_AUTHORITY')


class TransferStore:
    def __init__(self,root,manifest):
        T.keys(manifest,{'schema','domain','base_manifest_path','base_manifest_sha256','files',
                        'execution_authorized','grant_valid','dispatch_enabled','formal_certificate_accepted','N_G_authorized'},'CL_MANIFEST_SCHEMA')
        require(manifest['schema']==SCHEMA and manifest['domain']=='SAVED_H4','CL_MANIFEST_DOMAIN')
        require(all(manifest[k] is False for k in ('execution_authorized','grant_valid','dispatch_enabled','formal_certificate_accepted','N_G_authorized')),'CL_NO_AUTHORITY')
        self.root,self.manifest=Path(root),manifest
        raw=T.checked_file(root,manifest['base_manifest_path'],manifest['base_manifest_sha256'])
        self.base=B.EvidenceStore(root,T.read_json_bytes(raw));self.raw={};self.json={}
        seen=set()
        for role,ref in manifest['files'].items():
            T.identifier(role)
            T.keys(ref,{'path','sha256','source_commit','domain'},'CL_FILE_REFERENCE')
            require(ref['domain']=='SAVED_H4' and ref['path'] not in seen,'CL_DOMAIN_OR_DUPLICATE_PATH');seen.add(ref['path'])
            T.hex_digest(ref['source_commit'],40)
            self.raw[role]=T.checked_file(root,ref['path'],ref['sha256'])

    def document(self,role):
        if role not in self.json:self.json[role]=T.read_json_bytes(self.raw[role])
        return self.json[role]

    def fact(self,role,pointer):
        value=self.document(role)
        for key in pointer:value=value[key]
        return dict(role=role,**self.manifest['files'][role],pointer=pointer,value=value,value_digest=T.digest(value))

    def reference(self,role):return dict(role=role,**self.manifest['files'][role])


def verify_L_joins(store):
    """Connect saved results/CSV identities without any bound arithmetic."""
    seal=store.document('L_seal');summary=store.document('L_summary');worker=store.document('L_worker')
    require(seal['SOURCE']==summary['SOURCE']==L_SOURCE,'L_FIXED_SOURCE')
    require(worker['domain']=='SAVED_H4' and worker['precision_bits']==summary['precision_bits']==128,'L_SCOPE_PRECISION')
    require(summary['completed_calls']==dict(forward=148,inverse=148,basis=11,support=324),'L_SAVED_CALL_COUNTS')
    require(summary['formal_certificates_accepted']==0 and worker['formal_certificate_accepted'] is False,'L_ACCEPTANCE_CHANGED')
    require(worker['coverage_ledger']['status']=='COMBINATION_BLOCKED' and worker['compiler_transfer']=='MISSING_EVIDENCE','L_BLOCKERS_CHANGED')
    model_ref=store.base.manifest['files']['df_native_model'];model=store.base.document('df_native_model')
    require(seal['input_files'][seal['model_path']]==model_ref['sha256'],'L_C_MODEL_HASH_JOIN')
    local=list(csv.DictReader(io.StringIO(store.raw['L_local_csv'].decode())))
    bases=list(csv.DictReader(io.StringIO(store.raw['L_basis_csv'].decode())))
    supports=list(csv.DictReader(io.StringIO(store.raw['L_support_csv'].decode())))
    uniform=list(csv.DictReader(io.StringIO(store.raw['L_uniform_csv'].decode())))
    require((len(local),len(bases),len(supports),len(uniform))==(296,11,324,2),'L_CSV_COUNTS')
    require(len({(row['gate_hash'],row['direction']) for row in local})==296,'L_LOCAL_DUPLICATE')
    registry=store.base.document('DF_registry')['local_gates']
    require({row['gate_hash'] for row in local}==set(registry),'L_LOCAL_REGISTRY')
    for row in local:
        require(row['direction'] in ('forward','inverse') and row['precision_bits']=='128' and row['SOURCE']==L_SOURCE,'L_LOCAL_SCOPE')
        require(row['gate_name']==registry[row['gate_hash']]['name'] and T.read_json_bytes(row['parameters_json'].encode())==registry[row['gate_hash']]['parameters'],'L_LOCAL_DESCRIPTOR')
        saved=T.read_json_bytes(T.checked_file(store.root,row['result_path'],row['result_SHA256']))
        require(saved['discrepancy_upper_exact']==row['norm_interval_upper_exact'] and saved['matrix_hash']==row['gate_hash'] and saved['direction']==row['direction'] and saved['precision_bits']==128,'L_LOCAL_TOKEN_JOIN')
        B.endpoint_tokens(dict(lower=row['norm_interval_lower_exact'],upper=row['norm_interval_upper_exact']))
    specs={item['basis']['basis_hash']:item['basis'] for case in model['cases'].values() for item in case['blocks']+case['components']}
    seen=set()
    for row in bases:
        require(row['basis_hash'] not in seen,'L_BASIS_DUPLICATE');seen.add(row['basis_hash'])
        saved=T.read_json_bytes(T.checked_file(store.root,row['result_path'],row['result_SHA256']))
        require(saved['basis_hash']==row['basis_hash'] and row['basis_hash'] in specs,'L_BASIS_TOKEN_JOIN')
        for field in ('forward_upper_exact','inverse_upper_exact','control_zero_upper_exact','control_one_upper_exact'):
            require(saved[field]==row[field],'L_BASIS_ENDPOINT_TOKEN_JOIN')
        operations=[dict(gate_hash=op['description']['matrix_sha256'],qubits=op['qubits']) for op in specs[row['basis_hash']]['operations']]
        require(T.read_json_bytes(row['forward_operations_json'].encode())==operations,'L_BASIS_OPERATION_ORDER')
        forward=[op['gate_hash'] for op in operations]
        require(saved['forward_gate_order']==forward and saved['inverse_gate_order']==forward[::-1] and T.read_json_bytes(row['inverse_gate_order_json'].encode())==forward[::-1],'L_BASIS_FORWARD_INVERSE_ORDER')
    require([row['component_id'] for row in supports]==[str(i) for i in range(324)],'L_SUPPORT_IDS')
    for component_index,(row,component) in enumerate(zip(supports,model['cases']['PR']['components'])):
        require(row['basis_hash']==component['basis']['basis_hash'] and row['sign']==str(component['sign']) and T.read_json_bytes(row['qubit_support_json'].encode())==component['support'],'L_SUPPORT_JOIN')
        saved=T.read_json_bytes(T.checked_file(store.root,row['result_path'],row['result_SHA256']))
        require(saved['index']==component_index and all(saved[key]==expected for key,expected in dict(basis_hash=row['basis_hash'],sign=component['sign'],support=component['support']).items()),'L_SUPPORT_RESULT_SCOPE')
    require({r['method'] for r in uniform}=={'df_native_PF','df_native_PR'},'L_METHOD_REUSE')
    for row in uniform:
        method=row['method'].rsplit('_',1)[1]
        require(row['operator_bound_upper_exact']==worker['uniform_basis_bounds'][method]['local_basis_transfer_upper_exact'],'L_UNIFORM_TOKEN_JOIN')
    return dict(status='STATIC_L_IDENTITIES_BOUND',counts=dict(local=296,basis=11,support=324),
                numeric_bounds_compared=False,exact_unitarity_of_rounded_matrices_inferred=False)


def source_location(store,role,function):
    raw=store.base.raw[role]
    nodes=[n for n in ast.walk(ast.parse(raw)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==function]
    require(len(nodes)==1,'SOURCE_FUNCTION_UNIQUE:'+function)
    node=nodes[0]
    return dict(role=role,**store.base.manifest['files'][role],function=function,line=node.lineno,end_line=node.end_lineno)


def primitive_map(store):
    C=source_location(store,'C_ball_source','semantic_bound')
    tier=source_location(store,'C_ball_source','_tier')
    export=source_location(store,'C_export_source','build_export')
    rows=[
        dict(id='C_bias_distance',definition='abs(Re/Im((S_encoded-reference)/state_norm)-saved_signed_bias)',source=tier,quantity='NORMALIZED_RAY_AXIS_DISTANCE',nature='Arb enclosure of the stated encoded-model scalar; no literal transfer implied',overlap='Different observable/units from unscaled L operator delta',missing='Preparation and literal transfer; axis/source binding retained'),
        dict(id='C_basis_reuse',definition='N*d*c^(2N), d=max||J_b F_b-I||_F; c=max(1,||F_b||,||J_b||)',source=C,quantity='COLLAPSE_VS_EXPLICIT_ROUNDED_WORD_CANDIDATE',nature='Analytical cancellation bound for rounded factors with explicit norm amplification',overlap='Same label as L does not establish the same operator pair',missing='Concrete reduced/expanded word endpoint mapping and all event-family multiplicities'),
        dict(id='C_phase_accumulation',definition='gamma_m*(1+A)+m*2^(-1074), gamma_m=m*u/(1-m*u),u=2^(-52)',source=C,quantity='FINITE_STRAIGHT_LINE_PHASE_ERROR_CANDIDATE',nature='Analytical floating-point candidate, not an empirical fitted allowance',overlap='No equality to local RZ/XXPlusYY matrix discrepancy proved',missing='Map m and intermediate absolute magnitudes to actual builder/control DAG and exceptional cases'),
        dict(id='C_pi_rounding',definition='2*q*r*abs(binary64_pi-exact_pi) for PR; zero for PF',source=C,quantity='RTE_RELATIVE_PI_PHASE_CANDIDATE',nature='Arb interval includes exact pi; multiplicity/source-family transfer still a proposition',overlap='L retains symbolic same-angle phases; does not certify literal pi accumulation',missing='Prove count of negative product/event phases for every order0/2/4 path'),
        dict(id='C_semantic_total',definition='((reuse+phase_rounding+pi_difference)*c^(2N))*state_norm*abs(B_saved)',source=tier,quantity='SAVED_SEMANTIC_SIGNAL_ALLOWANCE',nature='Analytical aggregate already included in C axis total; primitive numeric pieces not separately saved',overlap='Aggregation and differing endpoint/normalization prevent subtraction or replacement',missing='Normalized-ray factor reconciliation: operator->raw expectation has n, then dividing by n cancels; saved n*B*delta needs separately proved dominance or a revised joint proof'),
        dict(id='C_total',definition='upper(bias_distance)+upper(circuit_semantic_allowance)',source=tier,quantity='SAVED_C_AXIS_ALLOWANCE',nature='Existing C numeric design-target evidence only',overlap='Semantic allowance is already inside; do not add it again',missing='No new aggregation or target/headroom evaluation authorized'),
        dict(id='L_local_forward_inverse',definition='||rounded_gate-exact_RZ_or_XXPlusYY(binary64_parameters)||_F',source=store.reference('L_kernel'),quantity='LOCAL_OPERATOR_NORM',nature='Existing128-bit outward Acb discrepancy,148 forward/148 inverse',overlap='Local idealization differs from cancellation defect JF-I',missing='Attach correct operator endpoints and literal parameter/decomposition identities'),
        dict(id='L_basis_uniform',definition='Telescoping local errors; conjugation f+i+fi; uniform PF/PR word propagation',source=store.reference('L_kernel'),quantity='UNSCALED_UNIFORM_BASIS_OPERATOR_DELTA',nature='Existing128-bit independent L bounds; phase/sign/angles held fixed',overlap='Physical primitive effects can be shared; consecutive triangle edges could still be valid if proved',missing='Prove source-word reduction family, diagonal/control/phase gap and B_saved transfer; no multiplication performed'),
        dict(id='export_flags',definition='semantic_error_bound_exact=0 and circuit_correspondence_proved=true',source=export,quantity='SAVED_DECLARATION_ONLY',nature='Exporter declaration; _tier actually recomputes semantic_bound',overlap='Flag/string zero is not mathematical evidence of zero literal error',missing='Never infer certificate or zero allowance from these flags')]
    return rows


def real_coverage(context):
    common=dict(domain='SAVED_H4',context=context,guarantee='SAVED_UNCHECKED')
    a=dict(common,id='saved_C_semantic',edge=dict(**{'from':'C_collapsed_rounded_native','to':'C_semantic_target_unbound'},quantity='SIGNAL_ERROR',state='saved_state_norm_multiplier',scale='B_saved'),primitives=['basis_reuse','phase_accumulation','pi_rounding'],bound_ref='saved_C.circuit_semantic_allowance')
    b=dict(common,id='saved_L_uniform',edge=dict(**{'from':'rounded_basis_factor_word','to':'ideal_same_angle_word'},quantity='OPERATOR_NORM',state='operator',scale='unscaled'),primitives=['basis_reuse','local_forward','local_inverse'],bound_ref='saved_L.uniform_basis_bounds')
    return dict(terms=[a,b],relation=R.coverage_relation(a,b),status='COMBINATION_BLOCKED',
                legacy_label_overlap='basis_reuse label shared; mathematical endpoint equality not established',
                normalized_ray_scaling_obligation='Saved C semantic uses n*B*delta while physical bias divides by n; no n>=1 or harmlessness assumption made',
                numerical_combination_performed=False,subtraction_or_cancellation_permitted=False)


def classification(gate,prior,representation,method):
    if gate in ('compiler_transfer',):return 'MISSING_HISTORICAL_EVIDENCE'
    if gate=='sampling_law' and method=='PR':return 'SAVED_EVIDENCE_UNCHECKED'
    if gate=='certificate_acceptance_implementation':return 'TECHNICAL_IMPLEMENTATION_NEEDED'
    if gate=='wrapper_transfer':return 'CERTIFIED_NUMERICS_NEEDED'
    if gate=='encoded_event_norm' and representation=='df_native':return 'SAVED_EVIDENCE_UNCHECKED'
    if gate=='final_combined_error_budget':return 'TECHNICAL_IMPLEMENTATION_NEEDED'
    if gate=='saved_B_to_shot_upper_binding' and method=='PR':return 'CERTIFIED_NUMERICS_NEEDED'
    if prior['proof_status'] in ('VERIFIED','ASSUMPTION_DEPENDENT'):return 'PROVED_EXISTING'
    return 'SAVED_EVIDENCE_UNCHECKED'


def produce(store):
    prior=B.produce_bindings(store.base);B.verify_bindings(store.base,prior)
    local_join=verify_L_joins(store);worker=store.document('L_worker')
    lemmas=dict(one_body=R.diagonal_identity('ONE_BODY'),df_squared=R.diagonal_identity('DF_SQUARED'),
                cosine=R.wrapper('cosine'),sine=R.wrapper('sine'),measurement=R.measurement_transfer(),
                controlled_diagonal={kind:R.controlled_diagonal_identity(kind) for kind in ('CRZ','RZZ','CONTROLLED_RZZ','PHASE')},
                ordered_basis=R.ordered_basis(2,['U1','U2'],['U2','U1']),
                finite_event_family=[R.event_phase(k,[1]*k,-1,identity=True) for k in (0,2,4)])
    rows=[]
    for old in prior['rows']:
        target=old['target'];representation,method=target['case'].rsplit('_',1);ctx=old['target_digest']
        T.validate_target(target)
        coverage=real_coverage(ctx) if representation=='df_native' else dict(status='COMBINATION_BLOCKED',L_relation=R.not_applicable(representation),reason='C literal/combined proof remains separate; DF L not inserted')
        Lfacts=[]
        if representation=='df_native':
            saved=worker['uniform_basis_bounds'][method]
            require(target['axis'] in saved['axes_reused'] and target['epsilon_sig'] in saved['precisions_reused'],'L_UNIFORM_AXIS_SCOPE')
            Lfacts=[store.fact('L_worker',['uniform_basis_bounds',method]),store.reference('L_local_csv'),store.reference('L_basis_csv'),store.reference('L_support_csv')]
        graph=dict(C_numerical=old['gates']['numerical_enclosure']['facts'],reference=old['gates']['reference_encoded_full_H']['facts'],
            signed_bias=old['gates']['bias_encoded_saved']['facts'],normalization=old['gates']['normalization_nominal_log_B']['facts'],
            encoded_event=old['gates']['encoded_event_norm']['facts'],local_L=Lfacts,C_L_coverage=coverage,
            phase_control_diagonal=dict(lemmas=['ONE_BODY','DF_SQUARED','SIGNED_EVENT_PHASE','CROSS_BRANCH_AND_SCALE'],real_H4_parameter_control_binding='MISSING_PROOF'),
            literal_hadamard=dict(lemma=lemmas[target['axis']],source=source_location(store,'old_wrapper_source','hadamard_wrapper'),saved_H4_transfer='MISSING_PROOF'),
            B_saved_scale=dict(facts=old['gates']['saved_B_to_shot_upper_binding']['facts'],formula='B_saved*delta (normalized ray); no numeric evaluation',nominal_B_not_substituted=True,readout_support_does_not_prove_encoded_norm=True),
            compiler=old['gates']['compiler_transfer'],combined_budget=dict(existing_total=old['error_terms'],new_total_created=False,status='COMBINATION_BLOCKED'),
            statistics=old['gates']['conditional_statistics'],acceptance=dict(authorized=False,accepted=False,N=None,G=None))
        require(len(graph)==14,'FOURTEEN_GRAPH_NODES')
        gates={}
        for gate,entry in old['gates'].items():
            gates[gate]=dict(scope=entry['scope'],proposition=entry['proposition'],target_digest=ctx,
                existing_proof_status=entry['proof_status'],category=classification(gate,entry,representation,method),
                proof_kind='Byte/declaration or conditional formal lemma only' if entry['proof_status'] in ('VERIFIED','ASSUMPTION_DEPENDENT') else 'Existing facts connected; unresolved mathematical binding',
                facts=entry['facts'],source_bindings=entry['source_bindings'],existing_obligations=entry['obligations'],conditions=entry['assumptions'],
                new_L_facts=Lfacts if gate in ('encoded_event_norm','wrapper_transfer','final_combined_error_budget') else [],
                all_required_H4_proofs_closed=False,original_gate_judgment_modified=False)
        rows.append(dict(target=target,target_digest=ctx,gates=gates,evidence_graph=graph,
            formal_certificate_accepted=False,N=None,G=None,all_scientific_gates_verified=False))
    return dict(schema=SCHEMA,domain='SAVED_H4',kind='TECHNICAL_PROOF_CONNECTION_NOT_CERTIFICATE',rows=rows,
        original_schema_reused=B.SCHEMA,original_gate_schema_unchanged=True,local_L_join=local_join,
        primitive_coverage=primitive_map(store),formal_lemmas=lemmas,C_L_combination='COMBINATION_BLOCKED',
        old_compiler='MISSING_EVIDENCE',execution_authorized=False,grant_valid=False,dispatch_enabled=False,
        formal_certificate_accepted=False,N_G_authorized=False,new_scientific_grants=0,
        new_H4_numeric_evaluations=0,new_B_scaled_bounds=0,new_40_axis_total_bounds=0,
        unconditional_PCG64_independence_required=False,mandatory_STOP=True)


def verify(store,output):
    require(output==produce(store),'CL_TARGET_SCOPE_FACT_OR_PROOF_TAMPERED')
    require(len(output['rows'])==40 and all(len(r['gates'])==14 for r in output['rows']),'CL_COMPLETE_AXES_GATES')
    return dict(status='PASS_STATIC_CL_CONNECTION_SCIENTIFIC_GATES_BLOCKED',axes=40,gates_per_axis=14,
                new_H4_numeric_evaluations=0,new_scientific_grants=0,formal_certificate_accepted=False,
                N_G_authorized=False,C_L_combination='COMBINATION_BLOCKED',compiler_transfer='MISSING_EVIDENCE',mandatory_STOP=True)
