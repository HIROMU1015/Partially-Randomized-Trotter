#!/usr/bin/env python3
"""Static execution inventory audit. Never regenerates a result or terminal."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h6_matched_execution_v1 import bounded_json
from trottertracks.resource_applicability.h6_matched_contract_v1 import file_hash, verify_parent, verify_sources
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.h6_matched_accounting_v1 import tasks, confirmation_ids, seed


def audit(root,output):
    output=Path(output);m=bounded_json(output/'frozen_manifest.json')
    verify_sources(root,m['source_commit'],m['source_hashes'])
    if verify_parent(root)!=m['input_identity']:raise ValueError('AUDIT_INPUT_CHANGED')
    inv=bounded_json(output/'execution_inventory.json');terminal=bounded_json(output/'execution_terminal.json')
    g=bounded_json(output/'authorization.json');binding=bounded_json(output/'launch_binding.json')
    if (g.get('schema')!='h6_matched_authorization_v1' or g.get('kind')!=m['kind'] or
            g.get('approved_by_user') is not True or g.get('one_shot') is not True or
            g.get('retry') is not False or g.get('resume') is not False or
            g.get('source_commit')!=m['source_commit'] or g.get('manifest_digest')!=digest(m) or
            bounded_json(output/'authorization_source.json')!=g or binding!={
                'manifest_digest':digest(m),'authorization_digest':digest(g),
                'authorization_source_sha256':file_hash(output/'authorization_source.json')}):
        raise ValueError('AUDIT_GRANT_BINDING')
    if inv['source_commit']!=m['source_commit'] or terminal['source_commit']!=m['source_commit']:
        raise ValueError('AUDIT_SOURCE_BINDING')
    expected={str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()
              and '.runtime_cache' not in p.parts and p.name not in ('execution_inventory.json','execution_terminal.json','STOP_REQUEST')}
    if set(inv['file_hashes'])!=expected:raise ValueError('AUDIT_INVENTORY_CLOSURE')
    for p,sha in inv['file_hashes'].items():
        path=(output/p).resolve()
        if not path.is_relative_to(output.resolve()) or file_hash(path)!=sha:
            raise ValueError('AUDIT_BYTES:'+p)
    if (terminal.get('mandatory_stop') is not True or terminal.get('next_stage_authorized') is not False
            or terminal.get('H6_status')!='H6_NOT_AUTHORIZED'):
        raise ValueError('AUDIT_STOP_FLAGS')
    if terminal['status']=='H6_MATCHED_RESOURCE_COMPLETE_MANDATORY_STOP':
        signals=bounded_json(output/'signal/signal_summary.json')['signals']
        if [s['cell'] for s in signals]!=m['plan']['cells']:raise ValueError('AUDIT_SIGNAL_COVERAGE')
        exploratory=[]
        for phase in ('exploration','confirmation'):
            registration=bounded_json(output/(phase+'_tasks.json'));registered=registration['tasks']
            selected=confirmation_ids(signals,exploratory,m['plan']) if phase=='confirmation' else None
            generated=tasks(signals,m['plan'],phase,selected)
            if phase=='exploration':
                by_id={s['cell']['id']:s for s in signals}
                for t in generated:
                    if t['cell']['method']!='B2':t['expected_signal']=by_id[t['cell_id']]['signals']['corrected']
            elif registration['selected']!=selected:raise ValueError('AUDIT_CONFIRMATION_SELECTION')
            if registered!=generated:raise ValueError('AUDIT_TASK_RULE')
            for i,t in enumerate(registered):
                path=output/f'cost_{phase}_{i:04d}'
                result=bounded_json(path/'cost_result.json');trajectory=bounded_json(path/'trajectory.json')
                if result['task']!=t or bounded_json(path/'worker_terminal.json')['status']!='TASK_COMPLETE':
                    raise ValueError('AUDIT_COST_COVERAGE')
                if trajectory['task']!=t or trajectory['event_digest']!=digest(trajectory['events']):
                    raise ValueError('AUDIT_EVENT_BINDING')
                keys=set()
                for row in result['rows']:
                    if any(row[k]!=t[k] for k in ('cell_id','phase','replica','seed')) or row['event_digest']!=trajectory['event_digest']:
                        raise ValueError('AUDIT_COST_TASK_BINDING')
                    keys.add((row['control'],row['axis']))
                expected_keys={(p,a) for p in (['symmetric_directional','ordinary'] if t['ordinary'] else ['symmetric_directional']) for a in ('cosine','sine')}
                if keys!=expected_keys or len(keys)!=len(result['rows']):raise ValueError('AUDIT_PAIRED_WRAPPERS')
                if phase=='exploration':exploratory+=result['rows']
    return dict(status='STATIC_IDENTITY_AND_COVERAGE_PASS',execution_status=terminal['status'],
                numerical_certificate_verified=False,scientific_GO_issued=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(audit(ROOT,a.output))


if __name__=='__main__':raise SystemExit(main())
