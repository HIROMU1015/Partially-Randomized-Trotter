"""Explicit one-pass metadata execution; NEVER invoked by import or preparation.

The real N1 pass requires the user's separate scientific execution instruction.
No approval files/grants/managers are introduced by this module.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess

from .m_n0_header_binding import BindingError, bind_n0_headers, summarize_binding, json_metadata

N1 = '744e1a09661ebba6841fe5faf456f88c41c47780'
N1_SOURCE = 'b6dbec9f80da35a88fba28795ad02ca74c9adfa0'
P1 = Path('artifacts/o2_a1_new_target/2026-11-06/n1')
# Git object names obtained without parsing any scientific catalog.
INPUT_BLOBS = dict(zip(('SOURCE.json','CONFIG.json','LAW.json','POLICY.json','PREPARATION.json'),
    ('9cf56d6570133bde49a692ca1cb7cb064382a0a1','ad272e31d0dfc2b5087721a2ed7ab488d663724e',
     '15b40241ba648c0f8750e087425d3baeb053730f','c089717e43b3ab5b1a637e802471a2a6bb40ba9c',
     'd7cf105c31b30100ee16dc4fefe41999cd5c009f')))
SOURCE_FILES = (
    'src/trottertracks/algorithm_codesign/m_label_average.py',
    'src/trottertracks/algorithm_codesign/m_n0_header_binding.py',
    'src/trottertracks/algorithm_codesign/m_n0_metadata_execution.py',
    'scripts/tracks/algorithm_codesign/run_m_n0_metadata.py',
    'src/trottertracks/algorithm_codesign/a1_new_target_n1.py',
    'src/trottertracks/algorithm_codesign/a1_control_variates.py',
    'src/trottertracks/algorithm_codesign/parent_allocation.py',
    'src/trottertracks/algorithm_codesign/shared_cpu.py',
    'src/trottertracks/algorithm_codesign/a2_lcu_access.py',
    'src/trottertracks/algorithm_codesign/a1_engine_connection.py',
    'src/trotterlib/rpe_hadamard_interrogation.py',
    'src/trotterlib/rte.py','src/trotterlib/df_rte_tail.py',
    'src/trotterlib/df_rte_qiskit.py','src/trotterlib/df_partial_s2.py',
    'src/trotterlib/df_partial_s2_repeated.py','src/trotterlib/research_direction_sequence_policy.py',
    'src/trotterlib/df_trotter/ops.py')


def preflight_source(root, expected_source_commit):
    if not isinstance(expected_source_commit,str) or not re.fullmatch('[0-9a-f]{40}',expected_source_commit):
        raise BindingError('INPUT_OR_POLICY_MISMATCH','explicit full execution SOURCE commit required')
    git = lambda *a: subprocess.check_output(['git','-C',str(root),*a])
    if git('rev-parse','HEAD').decode().strip() != expected_source_commit:
        raise BindingError('INPUT_OR_POLICY_MISMATCH','execution SOURCE HEAD mismatch')
    hashes = {}
    for name in SOURCE_FILES:
        data = (root/name).read_bytes()
        if data != git('show',expected_source_commit+':'+name):
            raise BindingError('INPUT_OR_POLICY_MISMATCH','execution SOURCE file mismatch: '+name)
        hashes[name] = hashlib.sha256(data).hexdigest()
    env = dict(python=platform.python_version(),numpy=importlib.metadata.version('numpy'),
               qiskit=importlib.metadata.version('qiskit'))
    if env != dict(python='3.12.3',numpy='1.26.4',qiskit='1.3.0'):
        raise BindingError('INPUT_OR_POLICY_MISMATCH','fixed metadata codec environment mismatch')
    return dict(execution_source_commit=expected_source_commit,source_sha256=hashes,
                environment=env,N1_result=N1,N1_source=N1_SOURCE,scope='one n0 slot metadata pass; retry 0')


def load_fixed_inputs(root):
    records, hashes = {}, {}
    for name, expected in INPUT_BLOBS.items():
        data = (root/P1/name).read_bytes()
        blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if blob != expected:
            raise BindingError('INPUT_OR_POLICY_MISMATCH','fixed N1 input blob mismatch: '+name)
        hashes[name] = dict(git_blob=blob,sha256=hashlib.sha256(data).hexdigest())
        records[name] = json.loads(data)
    source = records['SOURCE.json']
    fixed = dict(model='linear H4',distance_angstrom=1.,basis='STO-3G',df_rank=12,ld=3,
                 m=0,q=1,delta=.02,r=1,K=2,coefficient_atol=0.,system_qubits=8,
                 sector_electrons=4,identity_policy='extract_identity_phase')
    if source.get('source_commit') != N1_SOURCE or any(source.get('scope',{}).get(k)!=v for k,v in fixed.items()):
        raise BindingError('INPUT_OR_POLICY_MISMATCH','fixed N1 SOURCE/model scope mismatch')
    # SavedEngine's fixed catalog contract: identities were extracted already.
    if any(c['is_identity'] for c in records['PREPARATION.json']['components']):
        raise BindingError('INPUT_OR_POLICY_MISMATCH','fixed extracted-identity catalog changed')
    return records,hashes


def run_metadata_pass(root, fresh_output, *, expected_source_commit):
    """Exactly one binding/partition attempt. Fresh output prevents overwrites.

    No automatic retry or downstream measurement is reachable here. Tests mock
    the two input preflights and use only invented saved-schema files/headers.
    """
    root, out = Path(root).resolve(), Path(fresh_output).resolve()
    if out.is_relative_to(root/P1):
        raise ValueError('output must not be inside preserved N1 inputs')
    out.mkdir(parents=True,exist_ok=False)
    def write(name, value):
        with (out/name).open('x',encoding='utf-8') as f:
            json.dump(json_metadata(value),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    counts = dict(source_preflight=0,input_preflight=0,metadata_pass=0,header_binding=0,group_partition=0,
                  circuit_build=0,state_or_born=0,rng_or_shots=0,compile=0,variance_N_cost=0,retries=0)
    write('CALL_COUNTS.json',counts)
    result = None
    try:
        counts['source_preflight']=1
        source=preflight_source(root,expected_source_commit);write('SOURCE.json',source)
        counts['input_preflight']=1
        records, hashes=load_fixed_inputs(root);write('INPUTS.json',hashes)
        counts['metadata_pass']=counts['header_binding']=1
        write('CALL_START.json',dict(function='bind_n0_headers',scope='n0 unique rotation slot',counts=dict(counts)))
        bound=bind_n0_headers(records)
        write('HEADER_BINDINGS.json',dict(identities=bound.identities,labels=bound.records,
            contexts=[dict(A=bound.rows[0].A.key,B=bound.rows[0].B.key,
                           system_order=bound.rows[0].system_order)],
            selected_V=[dict(label=r.label,V=r.V.key) for r in bound.rows]))
        counts['group_partition']=1
        result=summarize_binding(bound)
        write('GROUPS.json',result['groups'])
    except BindingError as exc:
        result=dict(status=exc.code,reason=exc.detail,label=exc.label,
                    obtained_label_records=exc.partial,unprocessed='UNKNOWN',scientific_retry=False)
    except Exception as exc:
        result=dict(status='BINDING_INCOMPLETE',reason=type(exc).__name__+': '+str(exc),
                    unprocessed='UNKNOWN',scientific_retry=False)
    finally:
        if result is None:
            result=dict(status='BINDING_INCOMPLETE',reason='interrupted before result',unprocessed='UNKNOWN')
        write('RESULT.json',result)
        write('CALL_COUNTS_FINAL.json',counts)
        write('STOP.json',dict(status='MANDATORY_STOP',result_status=result['status'],
                               retry=0,downstream_started=False))
    return result
