#!/usr/bin/env python3
"""Saved-value comparison, focused preservation checks and lossless raw bundle."""
import csv,gzip,hashlib,importlib.util,json,math,os,subprocess,sys,tarfile
from fractions import Fraction as Q
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('hx_package_refinement',ROOT/'scripts/pauli_h2_h6_refinement.py');M=importlib.util.module_from_spec(s);s.loader.exec_module(M);R=M.R
def csvsave(path,rows):
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def unresolved():
 records=[]
 for n in [2,6]:
  rows=R.read(R.OUT/f'H{n}/candidate_results.json');reg={c['candidate_id']:c for c in R.registry(n)}
  for x in rows:
   if x['eligibility']!='NUMERICALLY_UNDETERMINED':continue
   c=reg[x['candidate_id']];scale=R.S.accounting_scale(c['law']);ea=Q(float(x['epsilon_sig'])/math.sqrt(2))
   bound=Q(scale['A_lower'])/ea**2;ceil=-(-bound.numerator//bound.denominator);st=x['price_statistics']
   price_sum=sum((Q(v) for v in st['mean_exact'].values()),Q(0)) if st else None
   records.append(dict(molecule=n,candidate_id=x['candidate_id'],epsilon_sig=x['epsilon_sig'],
     status='NUMERICALLY_UNDETERMINED_PRESERVED',actual_N=None,actual_G=None,
     diagnostic_only_optimistic_shot_lower_exact=str(ceil),optimistic_G_lower_using_saved_empirical_prices=None if price_sum is None else str(price_sum*ceil),
     exceeds_existing_shot_safe_integer_domain=ceil>=2**52,
     reason='even zero bias/u cannot reduce sufficient-shot formula below this lower bound; no population-cost claim',
     source_target=c.get('target_path',c['case_path']+'/target.json')))
 R.save(R.OUT/'unresolved_candidates.json',records)
 return records
def comparisons():
 combined=R.read(R.OUT/'H2_H6_comparison_core.json')
 with (ROOT/'artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/best_by_precision.csv').open() as f:
  for x in csv.DictReader(f):
   ratio=Q(x['PR_G_exact'])/Q(x['best_G_exact'])
   combined.append(dict(molecule='H4',epsilon_sig=x['epsilon_sig'],deterministic_candidate=x['best_deterministic_ids'],
     deterministic_G_exact=x['best_G_exact'],PR_candidate='PR_p0027_q8_r4_K4',PR_G_exact=x['PR_G_exact'],
     PR_over_D_exact=str(ratio),PR_over_D=float(ratio),PR_scope='fixed historical candidate, 256 saved trajectories; no PR optimization',
     H4_PR_optimized=False,actual_rho=None,PR_population_tail_bound=None,source='fbfbfd21a252a4a32f52e3b40e50acd91b3c626a/artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/best_by_precision.csv'))
 combined.sort(key=lambda x:(int(x['molecule'][1:]),-float(x['epsilon_sig'])))
 R.save(R.OUT/'H2_H4_H6_comparison.json',combined);csvsave(R.OUT/'H2_H4_H6_comparison.csv',combined)
 flat=[]
 for n in [2,6]:
  for x in R.read(R.OUT/f'H{n}/candidate_results.json'):
   st=x['price_statistics'];cfg=x['config'];t=x['target']
   flat.append(dict(molecule=f'H{n}',candidate_id=x['candidate_id'],method=cfg['method'],L_D=cfg['L_D'],q=cfg['q'],r=cfg['r'] if cfg['method']=='PR' else None,K=cfg['K'] if cfg['method']=='PR' else None,
      epsilon_sig=x['epsilon_sig'],eligibility=x['eligibility'],B_exact=t['B_saved_exact'],
      bias_cos_exact=t['axes']['cosine']['signed_bias_exact'],bias_sin_exact=t['axes']['sine']['signed_bias_exact'],
      u_cos_exact=t['axes']['cosine']['u_enc_exact'],u_sin_exact=t['axes']['sine']['u_enc_exact'],
      C_cos_exact=st['mean_exact']['cosine'] if st else None,C_sin_exact=st['mean_exact']['sine'] if st else None,
      price_n=st['n'] if st else None,N_cos=x['N_cos'],N_sin=x['N_sin'],G_exact=x['G_exact'],G_SE=x['G_SE'],
      cohort=x['cohort'],case_path=x['case_path']))
 csvsave(R.OUT/'registered_candidate_precision_table.csv',flat)
 return combined
def verification():
 records=[];errors=[];parents=[];scientific=[]
 for n in [2,6]:
  reg=R.registry(n);oldsha={}
  for c in reg:
   batch=ROOT/c['batch'];p=R.read(batch/'plan.json');R.verify(batch)
   for file in ['targets_native.json','prices_native.json']:
    key=str(batch/file)
    if key not in oldsha:oldsha[key]=True;records+=R.read(batch/file)['results']
  for folder in [R.OUT/f'H{n}/confirmation01',*sorted((R.OUT/f'H{n}').glob('precision_refinement_*'))]:
   for p in [folder/'native.json',*folder.glob('extension_*/native.json')]:
    if p.exists():records+=R.read(p)['results']
  science=dict(molecule=n,registered_candidates=len(reg),B0=sum(c['config']['method']=='B0' for c in reg),PF=sum(c['config']['method']=='PF' for c in reg),PR=sum(c['config']['method']=='PR' for c in reg),
    original_target_actions=len(reg),refinement_actions=sum((ROOT/c['case_path']/'target_refined.json').exists() for c in reg),reference_residual_actions=1,
    price_trajectories=sum(len(list((ROOT/c['case_path']/'prices').glob('*/*/result.json'))) for c in reg))
  science['top_level_compiles']=2*science['price_trajectories'];scientific.append(science)
  for c in reg:
   for p in (ROOT/c['case_path']/'prices').glob('*/*/result.json'):
    d=R.read(p)
    for a in R.AXES:
     for kind in ['input','compiled']:
      q=p.parent/f'{a}_{kind}.qpy.gz';b=q.read_bytes();entry=d['axes'][a][kind+'_qpy']
      if hashlib.sha256(b).hexdigest()!=entry['gzip_sha256'] or hashlib.sha256(gzip.decompress(b)).hexdigest()!=entry['raw_sha256']:errors.append(str(q))
 for p in [R.OUT/'input_native.json']:
  records+=R.read(p)
 records+=R.read(R.OUT/'reference_native.json')['results']
 if any(x['exit_code']!=0 or x.get('reaped')!=True for x in records):errors.append('native_exit_or_reap')
 h2=R.read(R.OUT/'H2/inputs/model.json')['provenance']
 for p,h in h2['old_files'].items():
  if R.sha(ROOT/p)!=h:errors.append('protected H2 '+p)
 old_h4=ROOT/'artifacts/non_df_h4_pauli_b0_comparison/2026-10-25'
 for p in ['result_summary.json','best_by_precision.csv','STUDY_STOP.json']:
  saved=subprocess.check_output(['git','show','fbfbfd21a252a4a32f52e3b40e50acd91b3c626a:'+str((old_h4/p).relative_to(ROOT))],cwd=ROOT)
  if (old_h4/p).read_bytes()!=saved:errors.append('protected H4 '+p)
 result=dict(status='PASS' if not errors else 'FAIL',errors=errors,native_children_waited_reaped=len(records),
   all_recorded_native_exit_zero=not any(x['exit_code'] for x in records),scientific_counts=scientific,
   historical_H2_inputs_preserved=True,historical_H4_results_STOP_preserved=True,
   old_grant_ledger_marker_STOP_modified=False,AI_subagents=0,independent_science_review=False,
   QPY_compressed_and_uncompressed_hashes_verified=True)
 R.save(R.OUT/'verification.json',result)
 if errors:raise ValueError(errors)
 return result
def archives():
 """Split one lossless tar.gz per molecule into <=8 MiB Git blobs."""
 inventory=[]
 for n in [2,6]:
  root=R.OUT/f'H{n}';folders=sorted([*root.glob('batch*'),*root.glob('precision_refinement_*'),root/'confirmation01'])
  path=R.OUT/f'H{n}_evidence.tar.gz';members={}
  with tarfile.open(path,'w:gz',compresslevel=6) as tar:
   for folder in folders:
    for p in sorted(folder.rglob('*')):
     if p.is_file():
      rel=str(p.relative_to(R.OUT));members[rel]=dict(sha256=R.sha(p),bytes=p.stat().st_size);tar.add(p,arcname=rel,recursive=False)
  parts=[]
  with path.open('rb') as f:
   i=0
   while data:=f.read(8*1024*1024):
    p=R.OUT/f'H{n}_evidence.tar.gz.part{i:03d}';p.write_bytes(data);parts.append(dict(path=p.name,sha256=R.sha(p),bytes=len(data)));i+=1
  inventory.append(dict(molecule=n,format='concatenate listed parts in order, then gzip tar extract in an empty directory',
    archive_sha256=R.sha(path),archive_bytes=path.stat().st_size,parts=parts,members=members))
 R.save(R.OUT/'data_archives.json',inventory)
 return inventory
if __name__=='__main__':
 if sys.argv[1]=='analyze':comparisons();unresolved();verification()
 elif sys.argv[1]=='archives':archives()
 else:raise ValueError(sys.argv[1])
