"""Fixed synthetic pilot controller. One compile, one solve/fixture, no retries."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
PRIVATE=Path('/tmp/ra-d0-v4-exact-backend-pilot-v2-20261009')
ART=ROOT/'artifacts/track_b_ra_d0_v4_exact_backend_pilot_v2/2026-10-09'
CONTRACT=json.loads((ART/'execution_contract_v1.json').read_text())
STOP=PRIVATE/'phase_b_STOP.json'
LEDGER=PRIVATE/'phase_b_ledger'
RESULT={'classification':'V4_EXACT_BACKEND_TECHNICAL_INCONCLUSIVE','rows':[],
        'LP_calls':0,'harness_compile_calls':0,'rational_io':'NOT_RUN','retries':0,'mandatory_STOP':True}

def save():
    (PRIVATE/'phase_b_result.json').write_text(json.dumps(RESULT,indent=2)+'\n')

def blocked(classification,reason):
    RESULT.update(classification=classification,technical_reason=reason)
    if not STOP.exists():
        with STOP.open('x') as f:json.dump({'failure':reason,'classification':classification,'retry':0},f)
    save()
    raise RuntimeError(reason)

def launch(name,command,wall):
    if STOP.exists():raise RuntimeError('attempt after STOP refused')
    if time.time()>=CONTRACT['pilot_deadline_epoch']:blocked('V4_EXACT_BACKEND_RESOURCE_INCONCLUSIVE','PILOT_WALL_CAP')
    spec={'id':name,'ledger':str(LEDGER),'stop_path':str(STOP),'command':command,'cwd':str(PRIVATE),
          'stdout':str(PRIVATE/'phase_b_output'/f'{name}.stdout.json'),
          'stderr':str(PRIVATE/'phase_b_output'/f'{name}.stderr.txt'),'TMPDIR':str(PRIVATE/'compiler_tmp'),
          'pilot_deadline_epoch':CONTRACT['pilot_deadline_epoch'],'wall_seconds':wall,
          'RSS_bytes':CONTRACT['RSS_bytes'],'address_space_bytes':CONTRACT['RSS_bytes'],
          'output_bytes':CONTRACT['output_bytes'],'output_roots':CONTRACT['output_roots'],'sample_seconds':.025}
    specs=PRIVATE/'phase_b_specs';specs.mkdir(exist_ok=True)
    path=specs/f'{name}.json'
    with path.open('x') as f:json.dump(spec,f,indent=2)
    p=subprocess.run([sys.executable,'-B',str(HERE/'guard.py'),str(path)],capture_output=True,text=True,
                     timeout=min(wall, max(.001,CONTRACT['pilot_deadline_epoch']-time.time()))+8)
    record=json.loads((LEDGER/f'{name}.result.json').read_text())
    result_path=Path(spec['stdout'])
    return record,result_path

def wire(problem):
    lines=[f"{len(problem['c'])} {len(problem['A'])} {len(problem['H'])}",problem['c0'],
           ' '.join(problem['c']),' '.join(problem['U'])]
    for key,rhs in [('A','b'),('H','f')]:
        lines+=[' '.join(row+[value]) for row,value in zip(problem[key],problem[rhs])]
    return '\n'.join(lines)+'\n'

def acquire(problem,echo=False):
    prefix='rational_io' if echo else problem['id']
    inputs=PRIVATE/'phase_b_inputs';inputs.mkdir(exist_ok=True)
    input_path=inputs/f'{prefix}.input.json';wire_path=inputs/f'{prefix}.input.rational.txt'
    with input_path.open('x') as f:json.dump(problem,f,indent=2)
    with wire_path.open('x') as f:f.write(wire(problem))
    if not echo:
        if RESULT['LP_calls']>=CONTRACT['LP_call_cap']:blocked('V4_EXACT_BACKEND_RESOURCE_INCONCLUSIVE','LP_CALL_CAP')
        RESULT['LP_calls']+=1;save() # conservatively consume before launch
    command=[str(PRIVATE/'pilot_harness'),str(wire_path)]+(['--echo-only'] if echo else [])
    guard,output=launch(prefix,command,30)
    entry={'id':problem['id'],'gate':'I/O' if echo else CONTRACT['gate_by_id'][problem['id']],
           'dimensions':problem['dimensions'],'bit_length':problem['input_bit_length'],'solver_guard':guard,
           'exact_input_path':str(input_path),'solver_output_path':str(output),'certificate':'NOT_RUN'}
    if echo:RESULT['rational_io_record']=entry
    else:RESULT['rows'].append(entry)
    if guard['failure']:
        classification='V4_EXACT_BACKEND_RESOURCE_INCONCLUSIVE' if guard['failure'] in ['WALL_CAP','PILOT_WALL_CAP','RSS_CAP','OUTPUT_CAP'] else 'V4_EXACT_BACKEND_TECHNICAL_INCONCLUSIVE'
        blocked(classification,f"{prefix}: {guard['failure']}")
    try:data=json.loads(output.read_text())
    except Exception as e:blocked('V4_EXACT_BACKEND_TECHNICAL_INCONCLUSIVE',f'{prefix}: non-JSON output: {e}')
    entry['backend']=data
    verify_guard,verify_output=launch(prefix+'_verify',[sys.executable,'-B',str(HERE/'verify.py'),str(input_path),str(output)],30)
    entry['verifier_guard']=verify_guard
    try:verdict=json.loads(verify_output.read_text())
    except Exception as e:blocked('V4_EXACT_BACKEND_CERTIFICATE_FAIL',f'{prefix}: verifier could not certify: {e}')
    entry['verification']=verdict;entry['certificate']='PASS' if verdict['PASS'] else 'FAIL'
    if verify_guard['failure'] or not verdict['PASS']:
        missing=any('unavailable' in x for x in verdict.get('failures',[]))
        blocked('V4_EXACT_BACKEND_PARTIAL' if missing else 'V4_EXACT_BACKEND_CERTIFICATE_FAIL',f'{prefix}: {verdict.get("failures",verify_guard["failure"])}')
    if echo:RESULT['rational_io']='PASS'
    save()

def execute():
    marker=PRIVATE/'phase_b_execution_started.json'
    with marker.open('x') as f:json.dump({'started_epoch':time.time(),'retry':0,'contract_SHA256':hashlib.sha256((ART/'execution_contract_v1.json').read_bytes()).hexdigest()},f)
    assert json.loads((PRIVATE/'phase_a_result.json').read_text())['classification']=='GUARD_V2_PASS'
    for name,sha in CONTRACT['frozen_files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    for name,sha in CONTRACT['backend_files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
    manifest=json.loads((ART/'synthetic_fixture_manifest_v1.json').read_text())
    by_id={p['id']:p for p in manifest['fixtures']}
    RESULT['harness_compile_calls']=1;save()
    record,_=launch('harness_compile',CONTRACT['compile_command'],1200)
    RESULT['build']=record
    if record['failure']:
        cls='V4_EXACT_BACKEND_RESOURCE_INCONCLUSIVE' if record['failure'] in ['WALL_CAP','PILOT_WALL_CAP','RSS_CAP','OUTPUT_CAP'] else 'V4_EXACT_BACKEND_BUILD_BLOCKED'
        blocked(cls,f'harness compile: {record["failure"]}')
    RESULT['harness_binary_SHA256']=hashlib.sha256((PRIVATE/'pilot_harness').read_bytes()).hexdigest();save()
    acquire(by_id['rational_roundtrip'],echo=True)
    for name in CONTRACT['LP_order']:acquire(by_id[name])
    RESULT.update(classification='V4_EXACT_BACKEND_PILOT_V2_PASS',technical_reason=None)
    save()

if __name__=='__main__':
    try:execute()
    except Exception as e:
        if RESULT.get('technical_reason') is None:RESULT['technical_reason']=f'{type(e).__name__}: {e}'
        save()
    print(json.dumps({k:v for k,v in RESULT.items() if k not in ['rows','build','rational_io_record']}))
    sys.exit(0 if RESULT['classification']=='V4_EXACT_BACKEND_PILOT_V2_PASS' else 1)
