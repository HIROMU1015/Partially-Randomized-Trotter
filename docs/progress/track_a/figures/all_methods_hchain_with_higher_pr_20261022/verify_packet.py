#!/usr/bin/env python3
"""Seal/check display artifacts and preservation; never run a science stage."""
from pathlib import Path
import argparse
from collections import Counter
import datetime
import hashlib
import json
import re
import subprocess
from zoneinfo import ZoneInfo

P=Path(__file__).resolve().parent;ROOT=P.parents[4]
ADDITION='\n- [高次PRを含む保存比較図：A/B/C/D・52図（2026-10-22）](figures/all_methods_hchain_with_higher_pr_20261022/README.md) — 旧Aとouter32を分けたoverview、PR内部比較、同一系列m5部分集合比較、カバレッジ。ローカル保存、未commit・未公開。\n'
def sha(b):return hashlib.sha256(b).hexdigest()
def read(n):return json.loads((P/n).read_text())
def dump(n,o):(P/n).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def git(*a):return subprocess.check_output(['git',*a],cwd=ROOT)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--check',action='store_true');ap.add_argument('--seal-stop',action='store_true');args=ap.parse_args()
 if args.check:
  prov=read('provenance.json')
  for x in prov['outputs']:assert sha((P/x['path']).read_bytes())==x['sha256'],x['path']
  assert read('STOP.json')['provenance_sha256']==sha((P/'provenance.json').read_bytes())
  for x in read('preservation_check.json')['files']:assert sha((ROOT/x['path']).read_bytes())==x['after_sha256'],x['path']
  print(json.dumps(dict(status='PASS',outputs=len(prov['outputs']),mandatory_STOP=True)));return
 before=read('preservation_baseline.json');protected=[]
 for path,h in before.items():
  b=(ROOT/path).read_bytes();after=sha(b)
  if path=='docs/progress/track_a/README.md':
   assert b.endswith(ADDITION.encode());assert sha(b[:-len(ADDITION.encode())])==h
   status='ORIGINAL_PREFIX_PRESERVED_ONE_INDEX_ENTRY_APPENDED'
  else:assert h==after,path;status='BYTE_IDENTICAL'
  protected.append(dict(path=path,before_sha256=h,after_sha256=after,status=status))
 old='docs/progress/track_a/figures/all_methods_hchain_20261022/'
 old_commit='5d53698534a1f325c8b7ed7be1bca6fa5192a77d'
 original_figures=[]
 for f in sorted((ROOT/old).glob('*_cost_eps_*')):
  if f.suffix not in ('.png','.svg','.pdf'):continue
  b=git('show',old_commit+':'+old+f.name);assert b==f.read_bytes()
  original_figures.append(dict(path=old+f.name,sha256=sha(b),commit=old_commit))
 assert len(original_figures)==30
 dump('preservation_check.json',dict(status='PASS',files_checked=len(protected),
   byte_identical=sum(x['before_sha256']==x['after_sha256'] for x in protected),index_additions_only=1,
   old_10_figures_30_exports_equal_fixed_5d536985=True,old_exports=original_figures,files=protected))
 sources=read('input_sources.json')
 for x in sources:
  raw=(P/x['local_path']).read_bytes();assert sha(raw)==x['sha256']
  assert raw==git('show',x['commit']+':'+x['source_path'])
 data=read('figure_data.json');cov=read('coverage_audit.json')['rows'];rc=read('rendering_checks.json')
 assert read('verification.json')['status']=='PASS'
 assert read('display_check.json')['status']=='PASS'
 assert read('display_checks/pdf_render_check.json')['count']==52
 assert len(rc['figures'])==52 and len(data['display_points'])==740 and len(cov)==600
 points=data['display_points'];assert sum(x['display_exact'] is not None for x in points)==648
 for row in points:
  if row['denominator_record'] and data['records'][row['record_id']]['domain']!='LEGACY_A':
   assert data['records'][row['denominator_record']]['domain']!='LEGACY_A'
 for group in ('A','B','C'):
  assert set(f['epsilon'] for f in data['figures'] if f['group']==group)==set(('0.05','0.01','0.005','0.001','0.0005'))
 for f in rc['figures']:
  for x in f['outputs']:assert sha((P/x['file']).read_bytes())==x['sha256']
 links=[]
 for name in ('README.md','captions_ja.md','figure_notes_ja.md','figure_mapping.md','coverage_audit.md'):
  for target in re.findall(r'\]\(([^)]+)\)',(P/name).read_text()):
   if target.startswith(('http://','https://','#')):continue
   links.append(dict(document=name,target=target))
 dump('packet_verification.json',dict(status='PASS',sources_equal_fixed_git_blobs=len(sources),
   local_links=len(links),links=links,coverage_rows=600,canonical_records=420,display_points=740,numeric_draw_points=648,
   missing_draw_slots=92,PR_internal_ratios_equal_saved_exact_ratios=60,
   usual_m5_subset_denominators_equal_saved_summary=10,
   no_unjoined_higher_PR_over_legacy_PF_ratios=True,old_figures_preserved=10,
   figures_by_group=dict(A=10,B=20,C=20,D=2),exports=156,
   coverage_status_counts=dict(Counter(x['status'] for x in cov)),
   all_five_requested_epsilons_covered=True,commit_created=False,pushed=False,
   remaining_unavailable_comparisons=['Higher PR vs stronger legacy S2/S4 PF: not identity-joined',
    'H2 outer32 vs deterministic PF: no same-identity signal set',
    'H8 higher PR: not saved','Same-native deterministic S2 controls: not saved/joined',
    'Noncancellation B0/B0I: PF/discard components missing',
    'Higher-PR independent confirmation: readiness only'],
   authorized_work_remaining=[],independent_scientific_review=False))
 outputs=[]
 for f in sorted(P.rglob('*')):
  if not f.is_file() or '.mpl-cache' in f.parts or '__pycache__' in f.parts:continue
  if f.name in ('provenance.json','STOP.json'):continue
  outputs.append(dict(path=str(f.relative_to(P)),bytes=f.stat().st_size,sha256=sha(f.read_bytes())))
 now=datetime.datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()
 dump('provenance.json',dict(schema='HIGHER_PR_SAVED_DISPLAY_PACKET_V1',created_at_jst=now,
   repository='HIROMU1015/Partially-Randomized-Trotter',checkout_root=str(ROOT),
   checkout_HEAD=git('rev-parse','HEAD').decode().strip(),preexisting_dirty_checkout=True,
   inputs=sources,output_hash_exclusions=['provenance.json self-reference','STOP.json binds provenance hash'],outputs=outputs,
   pinned_commits=dict(old_figures=old_commit,saved_evaluation='8613b3ecda2030a677b1690fe7aa3da23dec4873',
                      confirmation_readiness='c7c1d8229508253390212588ae2b25e504eb4915'),
   acquired_result_commits=dict(H2='a05f4ed19e87e1eb65f7fc346385b5b4eae69845',H4_H6='605ca07a653241db49348fbd7f4e091f9f18ced7'),
   acquisition_SOURCEs=dict(H2='4ec26921c6872ed6e8667f167074e1e405cf1a3f',H4_H6='5a448d94ea389ada6e2b2755de6a215ed2034679'),
   saved_evaluation_SOURCE='9a77d2b0a0b87e246e85f085b37f5923c18e27bb',
   confirmation_preparation_SOURCE='8825391ff73b531be5834bf852b9bf2479d1799b',
   method='Read pinned saved totals and existing formula bests; display-group minima only; format/normalize for plots',
   A_policy='Separate legacy A and outer32 panels; distinct relative denominators; no complete 8-series direct overlay',
   B_policy='Outer32 PR S2/Yoshida4/m5_best; modes separate',
   C_policy='H4/H6 same-native m5 subset only; noncancellation full-m5-only diagnostic',
   higher_independent_confirmation='NOT_EXECUTED_AT_PINNED_READINESS',
   new_scientific_calls=dict(sampling=0,compile=0,signal=0,fit=0,bootstrap=0,
       shot_generation=0,total_cost_accounting=0,scientific_reranking=0,reoptimization=0),
   verification_scope='Codex local data/export/visual/preservation verification; not independent scientific review or immutable CI',
   artifact_commit=None,new_commit_created=False,pushed=False,newly_published=False,
   state='LOCAL_UNCOMMITTED_DISPLAY_ARTIFACTS',next_stage_authorized=False))
 if args.seal_stop:
  dump('STOP.json',dict(status='COMPLETE_MANDATORY_STOP',date_jst='2026-10-22',completed_at_jst=now,
   figures=52,exports=156,groups=dict(A=10,B=20,C=20,D=2),precisions=['0.05','0.01','0.005','0.001','0.0005'],
   coverage_rows=600,source_snapshots=len(sources),verification='PASS',display_check='PASS',preservation='PASS',
   old_10_figures_preserved=True,scientific_computation=0,new_scientific_calls=0,
   new_shot_or_mean_SE_cost_accounting=0,independent_confirmation_executed=False,
   complete_8_series_direct_overlay='NOT_CREATED_UNJOINED_LEGACY_PF_AND_OUTER32; separated-panel A saved instead',
   unavailable_scope_is_explicit_in_audit=True,commit_created=False,pushed=False,newly_published=False,
   provenance_sha256=sha((P/'provenance.json').read_bytes()),automatic_retry=False,
   next_stage_authorized=False,authorized_work_remaining=[],
   next_action=dict(owner='user',status='OPTIONAL_PROPOSAL_NOT_AUTHORIZED',action='Select figures and publication scope if desired'),
   Codex_action='STOP; do not run confirmation, join new science comparisons, or publish'))
 for x in links:
  f=P/x['target'].split('#')[0]
  if f.name=='STOP.json' and not args.seal_stop:continue
  assert f.exists(),x
 print(json.dumps(dict(status='PASS',figures=52,exports=156,old_files_checked=len(protected),
    source_snapshots=len(sources),local_links=len(links),output_files=len(outputs),STOP_sealed=args.seal_stop)))

if __name__=='__main__':main()
