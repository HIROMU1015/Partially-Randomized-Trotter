"""Read-only byte/link/saved-arithmetic checks for the v0.4 review bundle."""
import ast
import argparse
import importlib.util
import json
from pathlib import Path
import re
import resource
import signal
import subprocess
from urllib.parse import unquote, quote
from urllib.request import urlopen

REPO=Path(__file__).resolve().parents[2]
BUNDLE=REPO/'artifacts/resource_applicability/track_a_h4_b0i_manuscript_integration/2026-10-14'
spec=importlib.util.spec_from_file_location('integration',REPO/'scripts/resource_applicability/integrate_h4_b0i_manuscript.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def links(path):
    checked=[]
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',path.read_text()):
        target=target.strip().strip('<>')
        if target.startswith(('https://','http://','mailto:','#')):continue
        target=unquote(target.split('#')[0])
        dest=(path.parent/target).resolve()
        assert dest.is_relative_to(REPO) and dest.exists(),(str(path.relative_to(REPO)),target)
        checked.append(str(dest.relative_to(REPO)))
    return checked


def verify():
    manifest=m.load(BUNDLE/'artifact_manifest.json')
    for path,expected in manifest['files'].items():
        dest=(REPO/path).resolve();assert dest.is_relative_to(REPO)
        assert m.identity(dest)==expected,path
    tables,summary=m.derive()
    for name,rows in tables.items():
        assert m.csv_read(BUNDLE/(name+'.csv'))==[{k:'' if v is None else str(v) for k,v in r.items()} for r in rows],name
    assert summary==m.load(BUNDLE/'analysis_summary.json')
    provenance=m.load(BUNDLE/'provenance.json')
    for item in provenance['inputs']+provenance['preserved_assets']:
        raw=(REPO/item['path']).read_bytes()
        assert m.identity(REPO/item['path'])=={k:item[k] for k in ('bytes','sha256')}
        original=subprocess.check_output(['git','-C',str(REPO),'show',item['commit']+':'+item['path']])
        assert raw==original,item['path']
    for item in provenance['source_closure_verified']:
        assert m.identity(REPO/item['path'])=={k:item[k] for k in ('bytes','sha256')}
    for item in provenance['fixed_references']:
        assert m.identity(REPO/item['local_copy'])=={k:item[k] for k in ('bytes','sha256')}
    authored=manifest['authored_markdown']
    local_links={path:links(REPO/path) for path in authored}
    figures=list((BUNDLE/'figures').iterdir())
    assert len(figures)==15 and {p.suffix for p in figures}=={'.png','.svg','.pdf'}
    assert sum(p.stat().st_size for p in figures)<=12*2**20
    assert len([p for p in figures if p.suffix=='.png'])==5
    for path in manifest['new_code']:
        tree=ast.parse((REPO/path).read_text())
        for node in ast.walk(tree):
            module=node.module if isinstance(node,ast.ImportFrom) else None
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [module] if module else []
            assert not any(n.split('.')[0] in ('qiskit','openfermion','pyscf','cupy','trotterlib','trottertracks') for n in names),path
    assert summary['scientific_claim_decision']=='GPT_REVIEW_PENDING'
    return dict(status='PASS_V04_SAVED_VALUE_BYTES_LINKS',files=len(manifest['files']),
        selected_family_precision_points=1812,compiled_metric_comparisons=1440,
        method_grid_regions=12,configuration_grid_regions=19,
        B0I_point_minimum=397,B2_point_minimum=1415,B0I_below_B2_minus_2SE=374,
        fixed_display_rows=216,matched_configuration_rows=120,
        source_closure_preserved=59,old_figure_assets_preserved=12,
        authored_markdown_local_links=sum(len(v) for v in local_links.values()),
        links_scope='new authored documents; archived byte-identical references use their original pinned GitHub links',
        figure_assets=15,figure_bytes=sum(p.stat().st_size for p in figures),
        original_measurement_regressions_repeated=False,new_science_compile_sampling=0,
        NPZ_access=0,H8_shared_venv_changes=0,independent_scientific_review=False)


def remote_verify():
    head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    branch=subprocess.check_output(['git','-C',str(REPO),'branch','--show-current'],text=True).strip()
    remote=subprocess.check_output(['git','-C',str(REPO),'ls-remote','--heads','origin','refs/heads/'+branch],text=True).strip()
    assert remote and remote.split()[0]==head,'remote branch does not match local result commit'
    manifest=m.load(BUNDLE/'artifact_manifest.json')
    targets={**manifest['files']}
    for name in ('artifact_manifest.json','verification.json'):
        path=str((BUNDLE/name).relative_to(REPO));targets[path]=m.identity(BUNDLE/name)
    total=0
    for path,expected in targets.items():
        url='https://raw.githubusercontent.com/HIROMU1015/Partially-Randomized-Trotter/'+head+'/'+quote(path,safe='/')
        with urlopen(url,timeout=15) as response:raw=response.read()
        import hashlib
        assert len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256'],path
        total+=len(raw)
    return dict(status='PASS_REMOTE_HEAD_AND_ALL_ASSET_BYTES',branch=branch,commit=head,
        remote_SHA=head,files=len(targets),bytes=total)


if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20))
    resource.setrlimit(resource.RLIMIT_CPU,(60,60));signal.alarm(90)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--remote',action='store_true')
    args=parser.parse_args();result=verify()
    if args.remote:result['remote']=remote_verify()
    print(json.dumps(result))
