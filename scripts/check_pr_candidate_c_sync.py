"""Read a published PR export through the unchanged C discovery/reconcile code.

The existing C checkout is never written. Discovery is scoped to the export's
published directory; the exact C source functions still classify and reconcile
those immutable Git blobs. This is a saved-evidence check, never a producer run.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import types
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
C_SOURCE='06e0b67634265d6dc818d4032f0cdb8fe8fd5629'
DEFAULT_BASELINE=C_SOURCE+':docs/research/h4_saved_rpe_cost_catalog_20261116/INPUTS.json'
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)


def parse_locator(text):
    if text.startswith('{'):return json.loads(text)
    commit,path=text.split(':',1)
    return dict(commit=commit,path=path)


def blob(root,commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=root)


def fixed_modules(root):
    catalog=types.ModuleType('trottertracks._pr_candidate_c_catalog');catalog.__package__='trottertracks'
    catalog.__file__=str(Path(root)/'src/trottertracks/saved_rpe_cost_catalog.py')
    exec(compile(blob(root,C_SOURCE,'src/trottertracks/saved_rpe_cost_catalog.py'),
                 C_SOURCE+':src/trottertracks/saved_rpe_cost_catalog.py','exec'),catalog.__dict__)
    sync=types.ModuleType('_pr_candidate_c_sync');sync.__file__=str(Path(root)/'scripts/sync_saved_rpe_publications.py')
    with patch.dict(sys.modules,{'trottertracks.saved_rpe_cost_catalog':catalog}):
        exec(compile(blob(root,C_SOURCE,'scripts/sync_saved_rpe_publications.py'),
                     C_SOURCE+':scripts/sync_saved_rpe_publications.py','exec'),sync.__dict__)
    return catalog,sync


def check(root,source,baseline,*,comparison_only=False):
    root=Path(root);C,S=fixed_modules(ROOT);reader=C.Reader(root)
    source=dict(source,commit=reader.commit(source['commit']));directory=str(PurePosixPath(source['path']).parent)
    original_git=S.git
    def focused_git(repo,*args):
        # Restrict file enumeration, without changing C's schema/reconcile rules.
        if args[:2]==('ls-tree','-r') and args[-2:]==('--','docs/research'):
            args=(*args[:-1],directory)
        return original_git(repo,*args)
    with patch.object(S,'git',focused_git):
        discovered=S.discover(reader,{'D2_PUBLISHED_PR_CANDIDATE_EXPORT':source['commit']})
    published=reader.read(source)
    if isinstance(published.get('summary'),str):
        summary_path=published['summary'];summary=reader.read(dict(commit=source['commit'],path=summary_path))
    else:summary_path=source['path'];summary=published
    input_source=summary['execution_source'];own_inputs=reader.read(input_source)
    expected={c.get('original_configuration_id')or c.get('configuration_id')or reader.read(c['configuration'])['id']for c in own_inputs['cases']}
    inputs=deepcopy(reader.read(baseline))
    if comparison_only:inputs['sources']=[]
    result,records,accepted=S.reconcile(reader,inputs,discovered,{})
    def from_export(row):
        return any(s.get('path')==summary_path and reader.document(s['commit'],s['path'])==summary
                   for s in row['result_sources'])
    own=[r for r in result['candidates']if r['configuration_id']in expected and from_export(r)]
    failures=[dict(source=r['source'],reason=r.get('reason'))for r in records.values()if r['status']=='PENDING']
    valid=(expected=={r['configuration_id']for r in own} and bool(own)
           and all(r['status']!='MISMATCH'and len(r['price_axes'])==26 for r in own)and not failures)
    return dict(schema='D2_PR_PUBLIC_C_SYNC_READ_CHECK_V1',status='PASS'if valid else'HELD',
        consumer_SOURCE=C_SOURCE,source=source,baseline=baseline,baseline_sources_included=not comparison_only,
        discovery_scope=directory,original_discover_and_reconcile=True,discovered=len(discovered),
        accepted_sources=accepted,pending=failures,expected_configuration_ids=sorted(expected),
        candidates=[{k:r.get(k)for k in ('method','configuration_id','total_configuration_id','binding_id','N_X','N_Y',
            'q3_status','status','conditional_cost','MSE_lower','MSE_upper','issues','price_axes','adopted_evaluation')}for r in own],
        method_minima=result['method_minima'],consumer_catalog_candidates=len(result['candidates']),
        new_scientific_calls=0,C_checkout_modified=False,
        qualification='READ_ONLY_PUBLIC_GIT_CONSUMER_CHECK_NOT_A_C_SCIENTIFIC_ADMISSION')


def main():
    p=argparse.ArgumentParser(description=__doc__);group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--join');group.add_argument('--summary');p.add_argument('--baseline',default=DEFAULT_BASELINE)
    p.add_argument('--comparison-only',action='store_true',help='Use baseline comparison contract without replaying its already accepted source list')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    value=check(ROOT,parse_locator(a.join or a.summary),parse_locator(a.baseline),comparison_only=a.comparison_only)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(status=value['status'],configurations=value['expected_configuration_ids'],
                         candidates=len(value['candidates']),pending=len(value['pending']))))
    return 0 if value['status']=='PASS'else 2


if __name__=='__main__':raise SystemExit(main())
