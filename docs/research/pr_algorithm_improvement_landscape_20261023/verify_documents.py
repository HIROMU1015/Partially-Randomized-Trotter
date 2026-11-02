"""Validate this survey's documents and cross-references using only the standard library.

This does not import, run, reproduce or re-analyze any scientific SOURCE or result.
It checks the survey's metadata, local links and publication-file identities only.
"""
import hashlib
import json
import pathlib
import re
import sys
import urllib.parse

def verify(directory):
    root = pathlib.Path(directory).resolve()
    errors = []
    catalog = json.loads((root / 'STUDY_CATALOG.json').read_text())
    sources = json.loads((root / 'SOURCE_AND_PROVENANCE_INDEX.json').read_text())['sources']
    studies = catalog['studies']
    ids = [s['study_id'] for s in studies]
    evidence = {e['evidence_id']: e for e in sources}
    if len(ids) != len(set(ids)):
        errors.append('duplicate study IDs')
    if len(evidence) != len(sources):
        errors.append('duplicate evidence IDs')
    for s in studies:
        for field in ['objective','modification','improvement_mechanism','comparison',
                      'scientific_conditions','metrics','historical_validation_level',
                      'actual_result','evidence_strength','stop_reason','open_questions',
                      'reusable_assets','pr_applicability','branch','period','evidence_sufficiency_reason']:
            if not s.get(field):
                errors.append(f'{s["study_id"]}: missing {field}')
        if s['evidence_sufficiency'] not in ['A', 'B', 'C']:
            errors.append(f'{s["study_id"]}: invalid sufficiency')
        for e in s['evidence_ids'] + s['implementation_evidence_ids'] + s['saved_result_evidence_ids']:
            if e not in evidence:
                errors.append(f'{s["study_id"]}: unresolved {e}')
        if s['evidence_sufficiency'] == 'C' and not s['gap_ids']:
            errors.append(f'{s["study_id"]}: C without GAP')
    gaps = set(re.findall(r'<a id="(gap-\d+)"', (root / 'EVIDENCE_GAPS.md').read_text()))
    for s in studies:
        for g in s['gap_ids']:
            if g.lower() not in gaps:
                errors.append(f'{s["study_id"]}: unresolved GAP {g}')
    opportunities = json.loads((root / 'OPPORTUNITIES.json').read_text())
    if opportunities['adopted'] or opportunities['scientific_priority_assigned']:
        errors.append('opportunities falsely adopted/ranked')
    for op in opportunities['opportunities']:
        for i in op['studies']:
            if i not in ids:
                errors.append(f'{op["id"]}: unknown study {i}')
    for e in sources:
        if not e['repository'].startswith('HIROMU1015/'):
            errors.append(f'{e["evidence_id"]}: unexpected repository')
        if not re.fullmatch('[0-9a-f]{40}', e['commit']):
            errors.append(f'{e["evidence_id"]}: commit not fixed')
        if not e['retrieval']['local_git_object_bytes']:
            errors.append(f'{e["evidence_id"]}: source falsely registered')
    for file in root.iterdir():
        if not file.is_file():
            continue
        raw = file.read_bytes()
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            errors.append(f'{file.name}: not UTF-8')
            continue
        if file.suffix == '.json':
            json.loads(text)
        # Construct patterns so this verification source does not contain a secret/path example.
        for prefix in ['/'+'home'+'/', '/'+'mnt'+'/', '/'+'tmp'+'/', 'github'+'_pat_', 'gh'+'p_']:
            if prefix in text:
                errors.append(f'{file.name}: forbidden path/credential literal')
        if file.suffix != '.md':
            continue
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if '://' in target or target.startswith('mailto:'):
                continue
            name, _, anchor = target.partition('#')
            dest = root / urllib.parse.unquote(name) if name else file
            if not dest.is_file():
                errors.append(f'{file.name}: missing local link {target}')
            elif anchor and f'id="{anchor}"' not in dest.read_text():
                errors.append(f'{file.name}: unresolved anchor {target}')
    return {'scope':'survey metadata and local document references only',
            'study_count':len(studies),'source_count':len(sources),
            'gap_count':len(gaps),'opportunity_count':len(opportunities['opportunities']),
            'http_source_byte_checks':sum(e['retrieval']['github_http_bytes_checked'] for e in sources),
            'errors':errors,'passed':not errors}

if __name__ == '__main__':
    report = verify(sys.argv[1] if len(sys.argv)>1 else pathlib.Path(__file__).parent)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    raise SystemExit(0 if report['passed'] else 1)
