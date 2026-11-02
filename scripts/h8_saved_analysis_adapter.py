"""Read-only, hash-pinned public Git blob adapter. Never executes old science."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/trotterlib'))
from h8_saved_analysis import CANDIDATES, FIXED, MODES, RARE, EPS, validate_trajectories
from h8_rare_tail_selection import require
from audit_h8_rare_tail_selection_evidence import audit, COMPILER, PROFILE

MANIFEST='references/h8_saved_analysis_inputs_v1.json'


def read_fixed(root=ROOT):
    m=json.loads((root/MANIFEST).read_text())
    require(m['schema']=='H8_SAVED_ANALYSIS_INPUTS_V1','INPUT_MANIFEST_SCHEMA')
    raw={}
    for e in m['inputs']:
        path=Path(e['path'])
        require(len(e['commit'])==40 and not path.is_absolute() and '..' not in path.parts,
                'EXACT_PUBLIC_GIT_PATH')
        b=subprocess.check_output(['git','--no-optional-locks','-C',str(root),'show',
                                   e['commit']+':'+e['path']],stderr=subprocess.PIPE)
        require(len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256']
                and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==e['git_blob'],
                'FIXED_INPUT_DRIFT:'+e['role'])
        require(e['role'] not in raw,'DUPLICATE_INPUT_ROLE')
        raw[e['role']]=b
    return raw


def projection(raw):
    """Copy saved fields and identities only. Called after authorization gates."""
    J=lambda role:json.loads(raw[role],parse_float=str)
    C=lambda role:list(csv.DictReader(io.StringIO(raw[role].decode())))
    prior=audit()  # Existing source/seed/pair/law checks, no statistical arithmetic.
    require(J('saved_laws')['laws']==prior['laws'],'SAVED_LAW_PROJECTION')
    require(J('selection_metadata')==prior['seed_and_selection'],'SELECTION_METADATA_PROJECTION')
    groups=C('map_groups');wrappers=C('map_wrappers')
    by={c:[] for c in CANDIDATES};exploration=set()
    ws={}
    for w in wrappers:
        if w['control']=='symmetric_directional':
            k=(w['phase'],w['cell_id'],w['replica'],w['axis'])
            require(k not in ws,'DUPLICATE_WRAPPER');ws[k]=w
    for g in groups:
        if g['phase']=='exploration':exploration.add(int(g['seed']))
        if g['phase']!='confirmation':continue
        c=g['cell_id'];pair=[ws[('confirmation',c,g['replica'],a)] for a in ('cosine','sine')]
        require(all(w['event_digest']==g['event_digest'] and w['seed']==g['seed']
                    and w['compiler_hash']==COMPILER and w['source']==g['source'] for w in pair),
                'PAIRED_WRAPPER_PROVENANCE')
        by[c].append(dict(candidate=c,replica=int(g['replica']),seed=int(g['seed']),
            phase='confirmation',law='unconditional',cosine=int(pair[0]['RZ']),sine=int(pair[1]['RZ']),
            event_digest_cosine=pair[0]['event_digest'],event_digest_sine=pair[1]['event_digest']))
    for rows in by.values():rows.sort(key=lambda r:r['replica'])
    validate_trajectories(by,CANDIDATES,expected_n=32,exploration_seeds=exploration)
    usual={}
    for r in J('original:D0')['H8_CONFIRMATION_UNION6']:
        if r['cell_id'] not in by:continue
        require(r['sample_phase']=='confirmation' and r['sample_count']==32,'D0_CONFIRMATION_ONLY')
        usual[(r['cell_id'],r['epsilon'],r['factor'])]=r['accounting']
    noncancel={}
    packed=gzip.decompress(raw['noncancel:accounting_rows_v1.jsonl.gz'])
    receipt=J('noncancel:accounting_rows_receipt_v1.json')
    require(len(packed)==receipt['original']['bytes'] and
            hashlib.sha256(packed).hexdigest()==receipt['original']['sha256'],'LOSSLESS_NONCANCEL')
    for l in packed.splitlines():
        r=json.loads(l,parse_float=str)
        if r['cell_id'] in by:noncancel[(r['cell_id'],r['epsilon'],r['factor'])]=r['accounting']
    saved={}
    for r in C('current_accounting'):
        if r['cell_id'] in by:
            key=(r['bias_kind'],r['epsilon'],r['factor'],r['cell_id'])
            require(key not in saved,'DUPLICATE_SAVED_PR_ROW');saved[key]=r
    endpoints=[]
    for mode,accounting in zip(MODES,(usual,noncancel)):
        for epsilon in EPS:
            for factor in ('1.0','10.0'):
                candidates={}
                for c in CANDIDATES:
                    a=accounting[c,epsilon,factor]
                    old=saved[mode,epsilon,factor,c];eligible=old['eligible']=='True'
                    require(eligible==(a['cost_acquisition_eligible'] is True),'SAVED_ELIGIBILITY_JOIN')
                    shots=None
                    if eligible:
                        shots=dict(cosine=a['axes']['real']['shots'],sine=a['axes']['imag']['shots'])
                        require(all(type(x) is int and x>=0 for x in shots.values()),'SAVED_SHOTS_REQUIRED')
                        require(old['G_after_exact']!='','SAVED_ELIGIBLE_COST_REQUIRED')
                    candidates[c]=dict(eligible=eligible,shots=shots,
                        saved_G_exact=old['G_after_exact'] or None)
                match=[r for r in C('current_comparisons') if r['bias_kind']==mode and
                       r['epsilon']==epsilon and r['factor']==factor and r['scope']=='strengthened184']
                require(len(match)==1,'FIXED_184_COMPARATOR_JOIN');det=match[0]
                endpoints.append(dict(id='|'.join((mode,epsilon,factor)),bias_kind=mode,epsilon=epsilon,
                    factor=factor,candidates=candidates,saved_PR_winner=det['PR_saved_id'],
                    comparator=dict(candidate=det['winner_after'],cost_exact=det['G_after_exact'],
                        cost_kind=det['winner_cost_status'],scope='strengthened184',
                        result_commit='96e69a01d1fae7a402aac2c4efa570a59f8cc7a0')))
    pilot={c:[] for c in RARE}
    for t in J('pilot_costs')['trajectories']:
        c=t['task']['cell_id'];pair={r['axis']:r for r in t['rows']}
        require(t['native_exit']==0 and t['reaped'] is True and
                all(r['compiler_hash']==COMPILER and r['event_digest']==t['event_digest'] for r in pair.values()),
                'PILOT_REAP_PAIR_COMPILER')
        pilot[c].append(dict(cosine=pair['cosine']['metrics']['RZ'],sine=pair['sine']['metrics']['RZ']))
    require(all(len(v)==8 for v in pilot.values()),'PILOT_N8')
    for c in RARE:
        require(all(int(x['counts'].get('2',0))==0 and int(x['counts'].get('4',0))==0
                    for x in prior['laws'][c]['original_observation'] if x['phase']=='confirmation'),
                'S0_CONFIRMATION_ALL_ZERO_ORDER')
    fixed={r['epsilon']:r['exploration_fixed_candidate'] for r in prior['original_fixed_candidate_evaluations']}
    require(fixed==FIXED,'ORIGINAL_EXPLORATION_FIXED_SELECTION')
    return dict(by=by,pilot=pilot,laws=prior['laws'],endpoints=endpoints,fixed=fixed,
        provenance=dict(fixed_input_files=len(raw),compiler=COMPILER,profile=PROFILE,
            preparation_and_runtime='PUBLISHED_NATIVE_RECEIPTS_NOT_CURRENT_COMPILER_EXECUTION',
            private_scientific_input_required=False,original_selection_receipt_bytes_verified=False,
            seed_streams='SOURCE_SUPPORTED_DISTINCT_STREAMS_NOT_MATHEMATICAL_INDEPENDENCE_PROOF',
            previous_STOP_and_ledgers='READ_ONLY_PRESERVED_NO_REFUND_NO_RESET'))
