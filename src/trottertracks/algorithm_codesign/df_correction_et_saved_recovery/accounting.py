"""Recover exactly the original post-T metadata block; no new science calls."""
import ast
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

from ..df_correction_et_phase_v7 import storage as original_storage
from ..df_correction_et_phase_v7.acceptance import verify_native_bindings
from ..df_correction_comparison.inventory_integrity import verify_inventory, verify_synthesis_result
from ..df_correction_comparison.native import t_completion, local_t_separation
from ..df_correction_comparison.angles import Angle


def original_block(text):
    main = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    body = next(n for n in main.body if isinstance(n, ast.Try)).body
    start = next(i for i,n in enumerate(body) if isinstance(n, ast.Assign)
                 and any(isinstance(x, ast.Name) and x.id == 'sequences' for x in n.targets))
    end = next(i for i,n in enumerate(body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Name) and n.value.func.id == 'dump_json'
               and any(isinstance(x, ast.Constant) and x.value == 'T_accounting.json' for x in ast.walk(n)))
    block = ast.Module(body=body[start:end+1], type_ignores=[])
    forbidden = {'synthesize','transpile','Operator','native_fixture','verified_rotation','cd_recipe','e_recipe'}
    for node in ast.walk(block):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func,ast.Name) else getattr(node.func,'attr','')
            if name in forbidden:
                raise PermissionError('Scientific evaluation in metadata recovery block')
    return ast.fix_missing_locations(block)


def recover(root, contract):
    root = Path(root)
    out = root/contract['original_result_namespace']
    path = root/contract['original_accounting_source']['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != contract['original_accounting_source']['sha256']:
        raise PermissionError('Original accounting source identity changed')
    terminal = original_storage.load_json(out/'terminal_status.json')
    if (terminal['classification'] != 'TECHNICAL_INCONCLUSIVE'
        or terminal['technical_reason'] != 'ValueError: Review file exceeds 512 KiB'
        or terminal['completed_stages'] != ['CD_SAVED_REFERENCE_ONLY','E_NATIVE','T_KEYS']
        or terminal['synthesis_inventory_integrity'] is not True):
        raise PermissionError('Original run is not fully computed storage-failure case')
    provenance = terminal['provenance']
    if provenance['source'] != contract['original_science_source'] or provenance['authorization'] != contract['original_authorization']:
        raise PermissionError('Original science/authorization identity changed')
    original_contract = json.loads((root/contract['original_contract_path']).read_text())
    binding = terminal['synthesis_inventory_binding']
    marker_hash = terminal['marker_SHA256']
    inventory = verify_inventory(out,binding,marker_sha256=marker_hash,provenance=provenance)
    verify_native_bindings(out,inventory)
    native = {r:original_storage.load_json(out/('E-r'+str(r)+'.json')) for r in (3,6)}
    if sum(n['compile_calls'] for n in native.values()) > 22 or len(inventory['unique_keys']) > 64:
        raise PermissionError('Original native/T call caps')
    circuits = {(r,row['controlled']):row['inventory'] for r in (3,6)
                for row in native[r]['native'] if row['arm']=='A'}
    jobs = []
    paths = {}
    for i in range(len(inventory['unique_keys'])):
        job = json.loads((out/('T-'+str(i)+'.job.json')).read_text())
        label = 'T-'+str(i)
        if (job['angle'] != inventory['unique_keys'][i]
            or job['marker_sha256'] != marker_hash
            or any(job[k] != v for k,v in binding.items())
            or Path(job['output']).name != label+'.json'):
            raise PermissionError('Original T job binding')
        jobs.append({'label':label,'output':job['output']})
        paths[job['output']] = out/(label+'.json')
    def load(path):
        return original_storage.load_json(paths[str(path)])
    def identity(path):
        row = original_storage.artifact_identity(paths[str(path)])
        row['logical_path'] = str(path)  # reproduce original absolute job path exactly
        return row
    captured = []
    def capture(path, value):
        if Path(path) != out/'T_accounting.json' or captured:
            raise PermissionError('Unexpected recovered output')
        captured.append(value)
    context = dict(F=F,Angle=Angle,out=out,contract=original_contract,native=native,circuits=circuits,
                   jobs=jobs,inventory=inventory,binding=binding,marker_hash=marker_hash,
                   provenance=provenance,load_json=load,artifact_identity=identity,
                   verify_inventory=verify_inventory,verify_synthesis_result=verify_synthesis_result,
                   t_completion=t_completion,local_t_separation=local_t_separation,dump_json=capture)
    block = original_block(path.read_text())
    exec(compile(block,str(path),'exec'),context)
    if len(captured) != 1 or captured[0]['CD_PR_total_winner'] is not None:
        raise PermissionError('Missing or overclaimed recovered accounting')
    verify_inventory(out,binding,marker_sha256=marker_hash,provenance=provenance)
    verify_native_bindings(out,inventory)
    return captured[0], {'original_terminal':terminal['classification'],
                        'original_terminal_reason':terminal['technical_reason'],
                        'original_science_source':provenance['source'],
                        'original_authorization':provenance['authorization'],
                        'original_marker_sha256':marker_hash,
                        'source_block_AST_sha256':hashlib.sha256(ast.dump(block,include_attributes=False).encode()).hexdigest(),
                        'native_compile_calls':sum(n['compile_calls'] for n in native.values()),
                        'T_key_evaluations':len(jobs),
                        'additional_native_synthesis_matrix_sampling_LP_calls':0,
                        'original_terminal_reclassified':False}
