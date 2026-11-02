#!/usr/bin/env python3
"""One authorized saved-result scalar batch; no simulation/sampling/compile."""
import argparse
import csv
from datetime import datetime
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import types
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
BASE = 'e4a9b1345a46f1bccccbf451f4e2d26521350fe3'
CONTRACT = 'docs/research/h4_p3_stage4_conditional_accounting_contract_20261102.md'
OUT = ROOT/'artifacts/df_pauli_matched_partition_validation/2026-11-02/stage4/run01'
OWN = ('scripts/run_h4_p3_stage4_accounting.py', 'src/trotterlib/matched_partition_stage4.py',
       'tests/test_matched_partition_stage4.py', 'docs/research/h4_p3_stage4_execution_20261102.md',
       'src/trotterlib/fixed_signal_accounting.py', 'src/trotterlib/h4_conditional_cng.py')


def modules():
    if 'p3_scalar' not in sys.modules:
        package = types.ModuleType('p3_scalar'); package.__path__ = [str(ROOT/'src/trotterlib')]
        sys.modules['p3_scalar'] = package
    return importlib.import_module('p3_scalar.matched_partition_stage4')


def sha(raw): return hashlib.sha256(raw).hexdigest()


def gitbytes(ref, path): return subprocess.check_output(['git', 'show', ref+':'+path], cwd=ROOT)


def write(name, value):
    path = OUT/name
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def inputs():
    raw_contract = gitbytes(BASE, CONTRACT)
    if (ROOT/CONTRACT).read_bytes() != raw_contract: raise ValueError('FIXED_CONTRACT_CHANGED')
    rows = re.findall(r'\| `([0-9a-f]{40})` \| `([^`]+\.json)` \| `([0-9a-f]{64})`', raw_contract.decode())
    if len(rows) != 8: raise ValueError('FIXED_8_INPUT_TABLE')
    saved, bindings = {}, {}
    for ref, path, digest in rows:
        raw = gitbytes(ref, path)
        if sha(raw) != digest or (ROOT/path).read_bytes() != raw: raise ValueError('INPUT_BYTES:'+path)
        kind = 'stage2' if '/stage2/' in path else 'stage3'
        saved[kind+'/'+Path(path).name] = json.loads(raw)
        bindings[path] = dict(ref=ref, sha256=digest)
    return saved, bindings, sha(raw_contract)


def provenance(saved):
    s2, s3 = saved['stage2/source_and_inputs.json'], saved['stage3/source_and_inputs.json']
    c2, c3 = s2['condition'], s3['conditions']
    for key in ('molecule', 'basis', 'charge', 'multiplicity', 'system_qubits', 'T_exact', 'q', 'r', 'K', 'arms'):
        if c2[key] != c3[key]: raise ValueError('SCIENTIFIC_CONDITION_MISMATCH:'+key)
    if (c2['dimension'] != 256 or c3['full_dimension'] != 256 or c2['distance_Angstrom'] != 1.
            or c3['distance_A'] != 1. or c3['partition'] != 'A' or c3['state_preparation'] != 'P=0; no state signal'):
        raise ValueError('FIXED_H4_CONDITIONS')
    if 'rank10' not in c2['DF'] or 'rank10' not in c3['DF']: raise ValueError('FIXED_DF_RANK_POLICY')
    for path, row in s2['inputs']['files'].items():
        if s3['input_files'].get(path) != row: raise ValueError('ORIGINAL_SCIENTIFIC_INPUT_MISMATCH:'+path)
    if saved['stage2/STOP.json']['status'] != 'STOP_STAGE2_UNRESOLVED_COMPONENTS': raise ValueError('STAGE2_STOP')
    if saved['stage3/STOP.json']['status'] != 'STOP_STAGE3_PRICE_CONFIRMATION_COMPLETE_NO_AUTOMATIC_FOLLOWUP':
        raise ValueError('STAGE3_STOP')
    if saved['stage3/price_statistics.json']['arms']['P']['n'] != 32 or saved['stage3/price_statistics.json']['arms']['DF']['n'] != 32:
        raise ValueError('FIXED_32_PRICE_SAMPLES')
    return dict(Stage2_SOURCE=s2['commit'], Stage3_SOURCE=s3['commit'], conditions=c3,
                state_hash=s2['inputs']['state_hash'], raw_norm_exact=s2['inputs']['raw_norm_exact'],
                original_scientific_input_bindings=len(s2['inputs']['files']))


def csv_outputs(result):
    columns = ['arm','epsilon_sig','status','N_cosine','N_sine','N_total']
    columns += [metric+'_'+field for metric in ('RZ','CX','depth') for field in ('G','G_exact','price_SE')]
    with (OUT/'accounting.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=columns, lineterminator='\n'); writer.writeheader()
        for row in result['rows']:
            entry={k:row[k] for k in ('arm','epsilon_sig','status','N_total')}
            entry.update({f'N_{a}':row['axes'].get(a,{}).get('shots') for a in ('cosine','sine')})
            for metric, cost in row['costs'].items():
                entry.update({metric+'_'+key:(cost.get(value) if cost is not None else None)
                    for key,value in [('G','G'),('G_exact','G_exact'),('price_SE','price_standard_error')]})
            writer.writerow(entry)
    with (OUT/'axes.csv').open('w', newline='') as stream:
        columns=['arm','epsilon_sig','axis','status','headroom_exact','bias_upper_operand','shots','legacy_shots','ceil_adjustment']
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n'); writer.writeheader()
        for row in result['rows']:
            for axis in ('cosine','sine'):
                values=row['axes'].get(axis,{})
                writer.writerow(dict(arm=row['arm'],epsilon_sig=row['epsilon_sig'],axis=axis,
                                     **{key:values.get(key) for key in columns[3:]}))


def execute():
    OUT.mkdir(parents=True, exist_ok=False)
    start=time.monotonic()
    counts=dict(scientific_scalar_batches=0,accounting_rows=0,axis_rows=0,
        new_sampling=0,new_compile=0,new_Gaussian=0,new_reference=0,new_signal=0,
        finite_bias_high_precision=0,automatic_retries=0)
    source={}
    try:
        kernel=modules(); saved, bindings, contract_hash=inputs(); lineage=provenance(saved)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        code={path:sha((ROOT/path).read_bytes()) for path in OWN}
        for path,digest in code.items():
            if sha(gitbytes(commit,path))!=digest: raise ValueError('UNFIXED_SOURCE:'+path)
        source=dict(commit=commit, contract_commit=BASE,contract_path=CONTRACT,contract_sha256=contract_hash,
            code_sha256=code,inputs=bindings,scientific_lineage=lineage,python=sys.version,executable=sys.executable,
            execution_JST=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
            assumptions=dict(finite_bit_gap=0,compiled_rho=0,B_upper_policy='saved engineering candidate assumed',
                             empirical_price_tail_guaranteed=False),scope='2 arms x 5 precision rows, saved scalar inputs only')
        write('source_and_inputs.json',source)
        counts['scientific_scalar_batches']=1
        def checkpoint(rows):
            counts['accounting_rows']=len(rows); counts['axis_rows']=2*len(rows)
            write('accounting_partial.json',rows)
            write('execution_status.json',dict(state='ACCOUNTING',counts=counts))
        result=kernel.evaluate(saved['stage2/results.json'],saved['stage3/prices.json'],
                               saved['stage3/price_statistics.json'],checkpoint)
        if len(result['rows'])!=10: raise ValueError('FIXED_TEN_ROWS')
        write('results.json',result); csv_outputs(result)
        for path,digest in code.items():
            if sha((ROOT/path).read_bytes())!=digest: raise ValueError('SOURCE_CHANGED:'+path)
        for path,row in bindings.items():
            if sha((ROOT/path).read_bytes())!=row['sha256']: raise ValueError('INPUT_CHANGED:'+path)
        write('verification.json',dict(status='SCALAR_CHECKS_PASS',input_files_unchanged=len(bindings),
            code_files_unchanged=len(code),contract_unchanged=sha((ROOT/CONTRACT).read_bytes())==contract_hash,
            scalar_sufficient_inequalities_verified=True,weighted_covariance_identity_verified=True,
            independent_price_cohorts=True,pilot_pooled=False,simulation_or_compiler_imported=False,
            interpretation='conditional engineering assumptions; no implemented accuracy certificate'))
        finish=dict(state='COMPLETED',counts=counts,elapsed_seconds=time.monotonic()-start,
            eligible_rows=sum(row['status']=='ELIGIBLE_CONDITIONAL' for row in result['rows']))
        write('execution_status.json',finish)
        write('STOP.json',dict(status='STOP_STAGE4_CONDITIONAL_ACCOUNTING_COMPLETE_NO_AUTOMATIC_FOLLOWUP',
              **finish,automatic_followup=False,implemented_accuracy_guaranteed=False))
        print(json.dumps(dict(source=commit,**finish)))
    except Exception:
        failure=dict(state='FAILED_NO_RETRY',counts=counts,source_commit=source.get('commit'),
            error=traceback.format_exc(),elapsed_seconds=time.monotonic()-start)
        write('execution_status.json',failure)
        write('STOP.json',dict(status='STOP_STAGE4_CONDITIONAL_ACCOUNTING_FAILURE_NO_RETRY',
              **failure,automatic_followup=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',required=True,action='store_true')
    parser.parse_args();execute()
