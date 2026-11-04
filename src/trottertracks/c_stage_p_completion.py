"""One fresh continuation study of missing P outputs; never resume old run.

The numerical column primitive and its aggregation order are unchanged. Progress
is observational only: saved columns cannot be loaded as a restart checkpoint.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction as F
from pathlib import Path
import importlib.metadata
import json
import multiprocessing
import resource
import signal
import time

from . import c_head_stage_p_execution as prior
from . import c_stage_p_literal as perword

io = prior.io
BASE = '5e1450b1071359b81819fe5de481cdeccc9d3f35'
SCOPE = 'C_PR_STAGE_P_MISSING_OUTPUTS_V1'
FAMILY = 'artifacts/research_c_stage_p_completion/2026-11-04'
CONTRACT = 'docs/research/research_c_stage_p_completion_contract_20261104.json'
ENTRY = 'scripts/run_research_c_stage_p_completion.py'
PARENT_CONTRACT_SHA = 'ab40f08c28c32bf803a95571f296779875864db2ba6b32a9d4b7eea1d8223cab'
LIMITS = dict(saved_evidence=1, saved_records_load=1, identity_gate=1,
              compiled_adjoint_certificate=1, shell_load=6, shell_certificate=3,
              finite_signal_profile=1, summary=1)
MAXIMUM = sum(LIMITS.values())


def save_atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    io.save(temp, value)
    temp.replace(path)


def preflight(root, path, *, launched=False):
    root = Path(root).resolve()
    c = io.read(path)
    old = io.read(root/c['parent_contract'])
    typed=dict(call_limits=LIMITS,maximum_logical_units=MAXIMUM,precision_bits=192,
               retry=0,resume=0,mandatory_stop=True,old_unused_units_transferred=0,
               new_build=0,new_compile=0,new_sampling=0,Stage_C_executable=False)
    if not prior.schema.same_json({k:c[k] for k in typed},typed):
        raise ValueError('TYPED_COMPLETION_POLICY')
    if (c['base_commit'] != BASE or c['scope'] != SCOPE
            or c['call_limits'] != LIMITS or c['maximum_logical_units'] != MAXIMUM
            or c['fresh_output'] != FAMILY+'/run01' or c['launch_output'] != FAMILY+'/launch01'
            or c['precision_bits'] != 192 or c['retry'] != 0 or c['resume'] != 0
            or c['mandatory_stop'] is not True or c['old_unused_units_transferred'] != 0
            or c['new_build'] != 0 or c['new_compile'] != 0 or c['new_sampling'] != 0
            or c['Stage_C_executable'] is not False
            or not prior.schema.same_json(c['identity'], old['identity'])
            or c['word'] != old['word'] or c['shells'] != old['shells']
            or c['parent_contract'] != prior.CONTRACTS['P']
            or io.sha(root/c['parent_contract']) != PARENT_CONTRACT_SHA
            or c['packages'] != old['packages']):
        raise ValueError('FIXED_COMPLETION_CONTRACT')
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']:
        raise ValueError('COMPLETION_SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'], c['saved_inputs']):
        for name, want in group.items():
            p = (root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p) != want:
                raise ValueError('SOURCE_OR_INPUT:'+name)
    if any(c['saved_inputs'].get(p) != h for p,h in old['saved_inputs'].items()):
        raise ValueError('ALL_PARENT_INPUTS_PRESERVED')
    for name, version in c['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('PACKAGE:'+name)
    expected = {name:old['fresh_output']+'/'+name for name in (
        'compiled_record.json','compiled_adjoint_record.json','raw_record.json',
        'compiled_raw.json','raw_literal.json','per_word.json','literal_reference.json',
        'raw_actual_inverse.json','science_calls.jsonl')}
    if c['parent_files'] != expected:
        raise ValueError('PRESERVED_PARENT_ROLES')
    if (c['parent_STOP'] != prior.FAMILY+'/P_execution_STOP.json'
            or c['population'] != old['population']):
        raise ValueError('PRESERVED_METADATA_ROLES')
    for name in [c['parent_contract'],c['parent_STOP'],c['population'],*expected.values(),
                 *(p for pair in c['shells'].values() for p in pair.values())]:
        if name not in c['saved_inputs']:
            raise ValueError('UNPINNED_ROLE:'+name)
    stop = io.read(root/c['parent_STOP'])
    if (stop['scientific_units_attempted'] != 30 or stop['COMPLETE'] != 29
            or stop['call_FAIL'] != 0 or stop['interrupted_without_terminal_event'] != 1
            or stop['runner_STOP_written'] is not False or stop['retry'] or stop['resume']
            or stop['mandatory_stop'] is not True):
        raise ValueError('PRESERVED_INTERRUPTION')
    prior.stage_r(root, old)
    launch=root/c['launch_output']
    if ((root/c['fresh_output']).exists()
            or (launch.exists() if not launched else not launch.is_dir() or (launch/'exit_status.json').exists())):
        raise ValueError('FRESH_NO_RETRY_RESUME')
    return dict(status='PASSIVE_READY',scientific_calls=0,arrays_decoded=0,
                maximum_logical_units=MAXIMUM,scope=SCOPE,Stage_C_ready=False,
                actual_CA_achieved=None,mandatory_stop=True)


def evidence(root, c):
    names = ('compiled_raw.json','raw_literal.json','per_word.json',
             'raw_actual_inverse.json','literal_reference.json','raw_record.json')
    return dict({n:io.read(root/c['parent_files'][n]) for n in names},
                population=io.read(root/c['population']),
                journal=[json.loads(s) for s in (root/c['parent_files']['science_calls.jsonl']).read_text().splitlines()])


def identity(e, records, c):
    from . import c_pr_aware_df_fixed_certificate as fixed
    C, inverse = records
    for r in records:
        fixed._validate(r)
        if r['qubits'] != 9:
            raise ValueError('FULL512_RECORD_REQUIRED')
    cr, rd, b, raw_inv = [e[n] for n in ('compiled_raw.json','raw_literal.json','per_word.json','raw_actual_inverse.json')]
    for x in (cr, rd, raw_inv):
        if (x['dimension'] != 512 or x['columns_evaluated'] != 512
                or x['precision_bits'] != 192 or x['phase_fit'] is not False
                or len(x['columns']) != 512
                or {r['column'] for r in x['columns']} != set(range(512))):
            raise ValueError('PRESERVED_COMPLETE_FULL512_CERTIFICATE')
    if (cr['status'] != 'VALIDATED_FIXED_CIRCUIT_PAIR'
            or rd['status'] != 'VALIDATED_RAW_LITERAL_OPERATOR'
            or b['status'] != 'VALIDATED_PER_WORD_COMPILED_LITERAL_OPERATOR'
            or cr['left_id'] != b['compiled_id'] or cr['left_id'] != perword.literal._digest(C)
            or cr['right_id'] != rd['raw_id'] or rd['raw_id'] != b['raw_id']
            or rd['raw_id'] != perword.literal._digest(e['raw_record.json'])
            or b['compiled_raw'] != cr or b['raw_literal'] != rd
            or F(b['upper_exact']) != F(cr['upper_exact'])+F(rd['upper_exact'])
            or b['all_word_uniform'] is not False or b['Head_bridge_to_B2_included'] is not False
            or b['phase_fit'] is not False or raw_inv['literal_inverse_claimed'] is not False):
        raise ValueError('SAME_FORWARD_CERTIFICATES_AND_PHASE')
    reference=e['literal_reference.json']
    if (reference['binding'] != b['binding'] or rd['target_id'] != b['binding']['target_id']
            or rd['word_id'] != b['binding']['word_id']
            or [x[0] for x in b['binding']['word']] != c['word']):
        raise ValueError('SAME_LITERAL_WORD_TARGET')
    binding=perword.bind_population(reference,e['population'])
    counts=Counter(x['event'] for x in e['journal'])
    last=e['journal'][-1]
    completed=Counter((x['kind'],x['label']) for x in e['journal'] if x['event']=='COMPLETE')
    if (counts != Counter(attempt=30,COMPLETE=29)
            or last['event'] != 'attempt' or last['kind'] != 'adjoint_certificate' or last['label'] != 'compiled'
            or any(completed[k] != 1 for k in (('records','compiled'),('adjoint_record','compiled'),
               ('compiled_raw_certificate','full512 columns'),('raw_literal_certificate','full512 columns;no cap2'),
               ('perword_triangle','two legs once;Head-to-B2 not included')))):
        raise ValueError('PARENT_COMPLETE_VS_INTERRUPTED_CALLS')
    return binding


def compare_progress(left, right, output, *, parallel=True):
    """Same fixed.compare arithmetic, sorted original order, durable row progress."""
    from . import c_pr_aware_df_fixed_certificate as fixed
    from . import c_pr_aware_df_zsxx_enclosure as z
    from flint import ctx, arb
    z.runtime_model();fixed._validate(left);fixed._validate(right)
    if left['qubits'] != right['qubits']:
        raise ValueError('FIXED_DIMENSION_MISMATCH')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    d=1<<left['qubits'];plan=fixed.resource_plan(d);rows=[];tick=time.monotonic()
    def accept(row, stream):
        rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
        progress=dict(completed_columns=len(rows),total_columns=d,last_column=row['column'],
                      elapsed_seconds=time.monotonic()-tick,restart_allowed=False)
        save_atomic(output/'progress.json',progress)
        print(json.dumps(dict(event='COLUMN_COMPLETE',**progress)),flush=True)
    with (output/'columns.jsonl').open('x') as stream:
        if parallel and d*d*(len(left['gates'])+len(right['gates']))>100000 and plan['workers']>1:
            with ProcessPoolExecutor(max_workers=plan['workers'],mp_context=multiprocessing.get_context('spawn'),
                    initializer=fixed._init_pair,initargs=(left,right)) as pool:
                futures=[pool.submit(fixed._pair_column,i) for i in range(d)]
                try:
                    for future in as_completed(futures):accept(future.result(),stream)
                except BaseException:
                    for future in futures:future.cancel()
                    raise
        else:
            fixed._init_pair(left,right)
            for i in range(d):accept(fixed._pair_column(i),stream)
            plan.update(workers_used=1,reason='tiny workload or artificial serial test')
    rows.sort(key=lambda x:x['column'])
    if len(rows)!=d or [x['column'] for x in rows]!=list(range(d)):
        raise ValueError('ALL_COLUMNS_EXACTLY_ONCE')
    with z._BALL_LOCK,ctx.workprec(z.PRECISION):
        squared=sum((z._real(F(row['upper_exact']))**2 for row in rows),arb(0))
        upper=z._fraction(squared.sqrt().upper())
    return dict(status='VALIDATED_FIXED_CIRCUIT_PAIR',norm='full_operator_2',phase_fit=False,
        precision_bits=z.PRECISION,dimension=d,columns_evaluated=d,
        upper_exact=str(min(F(2),upper)),lower_exact=str(max(F(x['lower_exact']) for x in rows)),
        upper_method='Frobenius of all certified column2 bounds;exact ideal gates plus recentered evaluation radii',
        columns=rows,resource_plan=plan,whole_law_bound=None,raw_Hamiltonian_reference_bound=None,
        CA_achieved=None,shots=None,total_resources=None)


def execute(root,c,out,call):
    from . import c_pr_aware_df_fixed_certificate as fixed
    e=call('saved_evidence','completed certificates,exact population and original journal',lambda:evidence(root,c))
    records=call('saved_records_load','same compiled and actual inverse records',lambda:tuple(
        io.read(root/c['parent_files'][n]) for n in ('compiled_record.json','compiled_adjoint_record.json')))
    binding=call('identity_gate','same C/R/D,phase,word,old completed vs missing calls',lambda:identity(e,records,c))
    C,inverse=records
    def certify():
        value=compare_progress(inverse,fixed.adjoint(C),out/'compiled_adjoint_columns')
        return dict(value,to_literal_adjoint_upper_exact=str(F(value['upper_exact'])+F(e['per_word.json']['upper_exact'])),
            same_realized_forward_record=perword.literal._digest(C),
            literal_target='D dagger;not inverse of nonunitary D',literal_inverse_claimed=False)
    inv=call('compiled_adjoint_certificate','all512 columns from start;no old checkpoint',certify)
    io.save(out/'compiled_actual_inverse.json',inv)
    from qiskit import qpy
    shells={}
    for label in ('head','X','Y'):
        pair={}
        for kind in ('raw','compiled'):
            def load(label=label,kind=kind):
                with (root/c['shells'][label][kind]).open('rb') as f:items=qpy.load(f)
                if len(items)!=1 or items[0].num_qubits!=9:raise ValueError('SHELL_SCHEMA')
                return items[0]
            pair[kind]=call('shell_load',label+kind,load)
        shells[label]=call('shell_certificate',label,lambda:perword.shell_certificate(pair['raw'],pair['compiled'],label))
        io.save(out/(label+'_shell.json'),shells[label])
    value=call('finite_signal_profile','one saved word/one occurrence;no expectation calibration',lambda:perword.finite_signal(
        e['per_word.json'],e['population'],{k:v['compiled_ideal']['upper_exact'] for k,v in shells.items()}))
    io.save(out/'finite_signal.json',value)
    return call('summary','old run remains interrupted;new missing-output study complete',lambda:dict(
        status='STAGE_P_MISSING_OUTPUTS_COMPLETE',parent_result_commit=BASE,
        reused_per_word=e['per_word.json'],reused_raw_actual_inverse=e['raw_actual_inverse.json'],
        compiled_actual_inverse=inv,shells=shells,finite_signal=value,population_binding=binding,
        original_41_call_run_complete=False,missing_outputs_obtained=True,
        all_q_compile_success_qualified=False,calibration_ready=False,Stage_C_ready=False,
        actual_CA_achieved=None,mandatory_stop=True))


def run(root,path,*,execute_science=False,executor=None,launched=False):
    if not execute_science:raise PermissionError('EXPLICIT_MISSING_OUTPUTS_SCIENTIFIC_PERMISSION')
    root=Path(root).resolve();preflight(root,path,launched=launched);c=io.read(path)
    from .c_pr_aware_df_zsxx_enclosure import runtime_model
    from .c1_stage3_integrated import Calls
    runtime_model();out=root/c['fresh_output'];out.mkdir(parents=True,exist_ok=False)
    call=Calls(out,LIMITS,MAXIMUM);tick=time.monotonic();previous={}
    io.save(out/'contract.json',c);io.save(out/'SOURCE.json',io.read(root/c['source_manifest']))
    def interrupted(number,frame):
        save_atomic(out/'termination_signal.json',dict(signal=number,elapsed_seconds=time.monotonic()-tick))
        raise InterruptedError('TERMINATION_SIGNAL:'+str(number))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        previous[sig]=signal.signal(sig,interrupted)
    try:
        result=(executor or execute)(root,c,out,call);io.save(out/'result.json',result);return result
    except BaseException as exc:
        call.stop.set()
        if call.first_failure is None:call.first_failure=dict(error_type=type(exc).__name__,error=str(exc))
        io.save(out/'partial_failure.json',dict(error_type=type(exc).__name__,error=str(exc)));raise
    finally:
        io.save(out/'STOP.json',dict(call_counts=call.counts,scientific_units=sum(call.counts.values()),
            failed=call.first_failure,retry=0,resume=0,mandatory_stop=True,Stage_C_executed=False,actual_CA_achieved=None))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-tick,
            parent_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            child_CPU_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime))
        for sig,handler in previous.items():signal.signal(sig,handler)
