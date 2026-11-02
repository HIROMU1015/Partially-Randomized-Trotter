"""Check saved output and artifacts, without rerunning the one-shot science."""
import argparse
import csv
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
from urllib.parse import unquote

from build_figures import HERE, ROOT, RAW, check, figure_data


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    result=check();d=figure_data()
    for entry in read(HERE/'input_sources.json'):
        assert sha(ROOT/entry['local_path'])==entry['sha256'],entry['label']
        if entry.get('commit'):
            content=subprocess.check_output(['git','show',entry['commit']+':'+entry['source_path']],cwd=ROOT)
            assert hashlib.sha256(content).hexdigest()==entry['sha256']
    plan=read(RAW/'execution_plan.json');grant=read(RAW/'grant.json')
    assert grant['plan_sha256']==sha(RAW/'execution_plan.json')
    assert grant['user_instruction_sha256']==sha(RAW/'user_instruction.txt')
    assert grant['SOURCE']==read(RAW/'output/terminal.json')['SOURCE']
    for entry in plan['inputs']+plan['source_closure']:assert sha(ROOT/entry['path'])==entry['sha256']
    assert read(RAW/'one_shot.marker.json')['grant_sha256']==sha(RAW/'grant.json')
    ledger=[json.loads(line) for line in (RAW/'output/ledger.jsonl').read_text().splitlines()]
    assert [r['kind'] for r in ledger]==['H4_EPS_0p001_S2_PF_SELECTION','SAVED_S2_TOTAL_ACCOUNTING_REGRESSION']
    selection=read(RAW/'output/candidate_selection_audit.json')[0]
    chosen=selection['selected'];assert chosen['cell_id']=='B0_p7_q8_2nd'
    assert chosen['G_RZ_exact']=='1666873406028' and chosen['eligible'] is True
    assert selection['eligible_S2_candidates']==21
    assert all(F(chosen['G_RZ_exact'])<=F(c['G_RZ_exact']) for c in selection['eligible_records'])
    assert len(read(RAW/'output/saved_theory_regression.json'))==17
    assert all(r['status']=='PASS' and r['use_in_current_main'] is False for r in read(RAW/'output/saved_theory_regression.json'))
    assert len(read(RAW/'output/theory_tasks.json'))==27
    assert all(t['native_theory_counter_calls']==0 and t['status']=='NOT_RUN_MISSING_FIXED_INPUT' for t in read(RAW/'output/theory_tasks.json'))
    actual=list(csv.DictReader((HERE/'figure_data.csv').open()))
    expected=[{k:v for k,v in r.items() if k not in ['PR_original_record','PF_original_record']} for r in d['rows']]
    normalize=lambda v:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else '' if v is None else str(v)
    assert actual==[{k:normalize(v) for k,v in r.items()} for r in expected]
    assert list(csv.DictReader((HERE/'coverage_matrix.csv').open()))==actual
    links=[]
    for path in HERE.glob('*.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',path.read_text()):
            target=target.strip().strip('<>')
            if target.startswith(('http:','https:','#','mailto:')):continue
            assert (path.parent/unquote(target.split('#',1)[0])).exists(),(path.name,target)
            links.append([path.name,target])
    for coord in read(HERE/'display_coordinates.json'):
        stem=coord['figure']; tree=ET.parse(HERE/(stem+'.svg'))
        texts=[''.join(e.itertext()) for e in tree.iter() if e.tag.endswith('}text')]
        assert any('Compiledonly' in ''.join(t.split()) for t in texts)
        assert not any(re.search('[\u3040-\u30ff\u4e00-\u9fff]',t) for t in texts)
        assert (HERE/(stem+'.png')).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert (HERE/(stem+'.pdf')).read_bytes().startswith(b'%PDF-')
        text=subprocess.check_output(['pdftotext',str(HERE/(stem+'.pdf')),'-'],text=True)
        assert 'Compiled only' in text and 'H8' in text
    prov=read(HERE/'provenance.json')
    for entry in prov['outputs']:
        f=ROOT/entry['path'];assert sha(f)==entry['sha256'] and f.stat().st_size==entry['bytes']
    stop=read(HERE/'STOP.json');assert stop['provenance_sha256']==sha(HERE/'provenance.json')
    assert stop['current_theory_G_acquired']==0 and stop['complete_four_total_comparison'] is False
    assert stop['push_executed'] is False
    result.update(fixed_input_SHA_checks='PASS',SOURCE_grant_ledger_binding='PASS',
                  H4_selected_candidate_and_eligibility='PASS',all_CSV_JSON_fields_match=True,
                  scientific_SCOPE='PARTIAL_MISSING_INPUT_NO_CURRENT_THEORY_EVALUATION',
                  local_links=len(links),manifest_files=len(prov['outputs']),STOP_binding=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true',required=True);p.parse_args()
    print(json.dumps(verify(),ensure_ascii=False))
