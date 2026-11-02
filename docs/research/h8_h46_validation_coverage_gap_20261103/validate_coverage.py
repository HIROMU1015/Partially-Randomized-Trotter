"""Check document metadata and literal transfers only, using the prior
validate_synthesis.py CSV/Git/literal-check pattern. No scientific imports,
array loading, RNG, numerical statistics, circuit operations, or network calls.

Run from any directory: python3 validate_coverage.py --repo REPOSITORY
The fixed evidence Git objects must already be available in that repository.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
FILES = {'REPORT.md', 'H468_VALIDATION_COVERAGE.csv', 'H8_MINIMAL_NEXT_PLAN.md',
         'PAPER_IMPLICATIONS.md', 'VALIDATION.json', 'validate_coverage.py'}
CLASSES = {'EXISTING_EVIDENCE_AVAILABLE', 'METADATA_RECOVERY_NEEDED',
           'SCIENTIFIC_REANALYSIS_NEEDED', 'NEW_SCIENTIFIC_COMPUTATION_NEEDED',
           'NOT_REQUIRED_FOR_O1', 'UNRESOLVED'}


def check(repo):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repo)
    def read_evidence(key):
        s = validation['evidence'][key]
        return git('show', s['commit']+':'+s['path']).decode()
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require({p.name for p in HERE.iterdir() if p.is_file()} == FILES, 'unexpected/missing output file')
    validation = json.loads((HERE/'VALIDATION.json').read_text())
    for key, entry in validation['evidence'].items():
        require(re.fullmatch('[0-9a-f]{40}', entry['commit']) is not None, 'nonfixed ref: '+key)
        actual = git('rev-parse', entry['commit']+':'+entry['path']).decode().strip()
        require(actual == entry['git_blob'], 'fixed Git evidence changed: '+key)
        require(entry['url'] == 'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'
                +entry['commit']+'/'+entry['path'], 'bad fixed URL: '+key)
    with (HERE/'H468_VALIDATION_COVERAGE.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    require(len({r['coverage_id'] for r in rows}) == len(rows), 'duplicate coverage row')
    required = {'physical_input','candidate_set','price_status','scientific_SOURCE',
                'fixed_url','original_status','gap_class','additional_validation_needed'}
    for row in rows:
        require(all(row.get(k) for k in required), 'missing required field: '+row['coverage_id'])
        require(row['gap_class'] in CLASSES, 'invalid gap class')
        s = validation['evidence'][row['evidence_id']]
        require(all(row[a] == s[b] for a,b in [('evidence_commit','commit'),
                    ('evidence_path','path'),('fixed_url','url')]), 'unbound evidence: '+row['coverage_id'])
    require({r['gap_class'] for r in rows} == CLASSES, 'classification coverage')
    originals = json.loads(read_evidence('MAIN'))['rows']
    scope = {r['comparison_id']:r for r in csv.DictReader(io.StringIO(read_evidence('PROTOCOL')))}
    by_id = {r['coverage_id']:r for r in rows}
    require(len([r for r in rows if r['aspect']=='fixed nominee and measured price']) == 6, 'fixed comparison count')
    for old in originals:
        r = by_id[old['comparison_id']]
        for field in ['epsilon','q','delta','K']:
            require(r[field] == str(old[field]), 'changed saved condition: '+field)
        require(r['prefix'] == str(old['L_D']) and r['R_or_r'] == str(old['r']), 'changed split/r')
        require(r['candidate_set'] == old['PR_id']+' vs '+old['PF_id'], 'candidate substitution')
        require(r['scientific_SOURCE'] == old['SOURCE'], 'source substitution')
        saved = json.loads(r['N_G_SE'])
        require(saved == {k:old[k] for k in ['n','G_PR_exact','G_PF_exact','paired_SE','lower_saved_point_cost']},
                'changed saved numerical token')
        require(r['original_status'] == old['saved_direction_status']+'; '+scope[old['comparison_id']]['TECHNICAL_PUBLIC'],
                'original status changed')
        if old['system']=='H8':
            require(old['mechanism_core_alias'] in r['confirmation'], 'H8 alias lost')
    original_claims = {r['claim_id']:r for r in csv.DictReader(io.StringIO(read_evidence('CLAIMS')))}
    for cid, fields in validation['original_phase3_and_synthesis_status'].items():
        require(all(original_claims[cid][k] == v for k,v in fields.items()), 'original claim status changed')
    original_gap = next(r for r in csv.DictReader(io.StringIO(read_evidence('PHASE3_GAPS'))) if r['gap_id']=='G04')
    require(validation['original_G04'] == original_gap, 'G04 original changed')
    require(by_id['H8_G04']['original_status'] == original_gap['decision_status'], 'G04 CSV status changed')
    for cid in ['H8_LONG_GAP','H8_HIGH_PR_GAP']:
        r = by_id[cid]
        require(all(r[k]=='NOT_FIXED_FOR_NEW_WORK' for k in ['epsilon','T','q','delta','prefix','R_or_r','K']),
                'new scientific execution conditions fixed')
        require(r['additional_validation_needed'].startswith('NO_FOR_CURRENT_O1;'), 'extension made mandatory')
    require(validation['G04_public_availability']['status']=='NOT_LOCATED_IN_CURRENT_PUBLIC_LOOKUP', 'unacquired supplement claimed')
    require(validation['scope']['mandatory_stop_after_publication'] is True, 'missing mandatory STOP')
    require(all(v is False for k,v in validation['scope'].items() if k!='mandatory_stop_after_publication'),
            'scope boundary changed')
    for p in HERE.glob('*.md'):
        content = p.read_text()
        definitions = dict(re.findall(r'^\[([A-Z][A-Z0-9_]+)\]: (https://\S+)$', content, re.M))
        keys = set(re.findall(r'\[([A-Z][A-Z0-9_]+)\]', content))
        require(keys == set(definitions), 'unresolved reference in '+p.name)
        for key, url in definitions.items():
            require(url == validation['evidence'][key]['url'], 'wrong evidence link')
        for target in re.findall(r'\]\(([^)]+)\)', content):
            if '://' not in target:
                require((HERE/target.split('#')[0]).exists(), 'broken local link: '+target)
        require('mandatory STOP' in content, 'missing STOP statement')
    for p in HERE.iterdir():
        if p.is_file():
            content = p.read_text()
            require(re.search(r'/(?:home|mnt|srv)/|(?:ghp_|github_pat_)[A-Za-z0-9_]+|PRIVATE[ ]KEY',content) is None,
                    'private locator or credential pattern: '+p.name)
    return dict(status='PASS',fixed_evidence_objects=len(validation['evidence']),coverage_rows=len(rows),
                fixed_comparisons=len(originals),classification_labels=sorted(CLASSES),
                meaning='metadata/literal/source-link checks only; no scientific validation or independent review')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(check(args.repo),ensure_ascii=False,indent=2))
