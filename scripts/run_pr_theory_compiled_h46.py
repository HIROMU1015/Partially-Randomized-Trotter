#!/usr/bin/env python3
"""No real data are read numerically without a new sealed H4/H6 grant."""
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pr_theory_compiled',ROOT/'src/trotterlib/pr_theory_compiled.py')
lib=importlib.util.module_from_spec(spec);sys.modules[spec.name]=lib;spec.loader.exec_module(lib)


def run(plan,grant,input_path,output):
    source=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    # Reject invalid drafts before opening even the numeric input path.
    lib.verify_grant(plan,grant,source=source,input_sha=plan.get('input_sha256'))
    if not plan.get('source_hashes') or not plan.get('rows'):
        raise ValueError('SEALED_SOURCE_AND_EXACT_ROW_SET_REQUIRED')
    if not {'src/trotterlib/pr_theory_compiled.py','src/trotterlib/native_df_gate_model.py',
            'scripts/run_pr_theory_compiled_h46.py'} <= set(plan['source_hashes']):
        raise ValueError('COMPLETE_RUNTIME_CLOSURE_REQUIRED')
    for name,sha in plan['source_hashes'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or lib.file_hash(p)!=sha:
            raise ValueError('SOURCE_CLOSURE')
    input_path=Path(input_path).resolve();output=Path(output).resolve()
    if not output.is_relative_to('/home/AbeHiromu') or str(output)!=plan.get('output'):
        raise ValueError('OWN_HOME_OUTPUT_BINDING')
    if lib.file_hash(input_path)!=plan['input_sha256']:
        raise ValueError('INPUT_BYTES')
    output.mkdir(exist_ok=False)  # Exclusive, retained claim including on STOP.
    terminal=dict(status='STOP',first_stop=None,mandatory_stop=True,next_stage_authorized=False,
                  SOURCE=source,plan_digest=lib.digest(plan),rows_completed=0)
    try:
        payload=json.loads(input_path.read_bytes())
        if isinstance(payload,dict) and payload.get('format')=='H46_SAVED_TABLES_V1':
            tables={}
            for name in ('theory','connections'):
                ref=payload[name]
                # The new plan seals this descriptor, including original Git
                # commit/path/blob-byte SHA, before any numeric table is read.
                if len(ref['commit'])!=40 or not ref['path'].startswith('artifacts/') or '..' in Path(ref['path']).parts:
                    raise ValueError('FIXED_SAVED_TABLE_REFERENCE')
                raw=subprocess.check_output(['git','-C',str(ROOT),'show',ref['commit']+':'+ref['path']])
                import hashlib
                if hashlib.sha256(raw).hexdigest()!=ref['sha256']:
                    raise ValueError('SAVED_TABLE_BYTES')
                tables[name]=json.loads(gzip.decompress(raw) if ref['path'].endswith('.gz') else raw)
            rows=lib.saved_table_rows(tables['theory'],tables['connections'],payload['cases'],models=plan['models'],metrics=plan['metrics'])
            if len(rows)!=plan.get('registered_row_count'):
                raise ValueError('REGISTERED_ROW_COUNT')
            # Exact input bytes + selected cases, models and metrics determine
            # this row set; no floating arithmetic is used for its selection.
            row_set=[{k:r[k] for k in ('model','axis','metric','compiled_observation_id')} for r in rows]
            if row_set!=plan['rows']:
                raise ValueError('EXACT_SAVED_ROW_IDENTITY_SET')
        else:
            rows=payload
            if [lib.digest(r) for r in rows]!=plan['rows']:
                raise ValueError('EXACT_ROW_SET_BINDING')
        unique=lib.unique_observations(rows)
        with (output/'comparisons.jsonl').open('x') as stream:
            for row in rows:
                if row['model'] not in plan['models'] or row['metric'] not in plan['metrics']:
                    raise ValueError('REGISTERED_MODEL_METRIC')
                # Reserve each analysis operation durably before its arithmetic.
                with (output/'journal.jsonl').open('a') as journal:
                    journal.write(json.dumps(dict(row_digest=lib.digest(row)))+'\n');journal.flush();os.fsync(journal.fileno())
                value=lib.compare_row(row,conditional=plan.get('allow_conditional_metadata') is True)
                stream.write(json.dumps(dict(row_digest=lib.digest(row),comparison=value))+'\n');stream.flush()
                terminal['rows_completed']+=1
        terminal.update(status='COMPLETE_MANDATORY_STOP',unique_compiled_observations=unique)
    except BaseException as e:
        terminal['first_stop']=repr(e)
        raise
    finally:
        (output/'terminal.json').write_text(json.dumps(terminal,indent=2)+'\n')
    return terminal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--plan');p.add_argument('--grant')
    p.add_argument('--input');p.add_argument('--output');a=p.parse_args()
    if not a.execute:
        print(json.dumps(dict(status='PREPARATION_ONLY',namespace=lib.NAMESPACE,numerical_execution=False)))
        return
    if not all((a.plan,a.grant,a.input,a.output)):
        p.error('new plan/grant/input/output required')
    print(json.dumps(run(json.loads(Path(a.plan).read_bytes()),json.loads(Path(a.grant).read_bytes()),a.input,a.output)))

if __name__=='__main__':main()
