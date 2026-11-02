#!/usr/bin/env python3
"""Precision-refined views over preserved v1 raw outputs."""
import importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('hx_refined_runner',ROOT/'scripts/pauli_h2_h6_study.py');R=importlib.util.module_from_spec(s);s.loader.exec_module(R)
original_registry=R.registry
def registry(n):
 rows=original_registry(n)
 for c in rows:
  p=ROOT/c['case_path']/'target_refined.json'
  if p.exists():c['target']=R.read(p);c['target_path']=str(p.relative_to(ROOT));c['target_sha256']=R.sha(p)
 return rows
R.registry=registry
R.CODE+=['scripts/pauli_h2_h6_refinement.py','scripts/refine_pauli_h2_h6_numeric.py','src/trotterlib/pauli_vector_tight_enclosure.py']
def refine(n):
 rows=[c for c in original_registry(n) if c['target'] is not None and not (ROOT/c['case_path']/'target_refined.json').exists()]
 if not rows:return
 parent=R.OUT/f'H{n}';k=len(list(parent.glob('precision_refinement_*')))+1;d=parent/f'precision_refinement_{k:02d}';d.mkdir()
 plan=dict(molecule=n,SOURCE=R.source(n),reason='tight correlated unitary/contraction input error; old bounds remain valid',
           state_actions_max=len(rows),new_compile=0,sampling=0,
           candidates=[dict(candidate_id=c['candidate_id'],old_target_sha256=R.sha(ROOT/c['case_path']/'target.json'),spec_sha256=R.sha(ROOT/c['case_path']/'spec.json'),case_path=c['case_path']) for c in rows])
 R.save(d/'plan.json',plan)
 for name in ['candidate_results.json','best_exploration.json']:
  if (parent/name).exists():(d/('prior_'+name)).write_bytes((parent/name).read_bytes())
 jobs=[]
 for c in rows:
  path=ROOT/c['case_path'];jobs.append(([R.ARB,str(ROOT/'scripts/refine_pauli_h2_h6_numeric.py'),str(parent/'inputs/model.json'),str(parent/'inputs/reference.json'),str(path/'spec.json'),str(path/'target_refined.json')],d/'native'/c['candidate_id']))
 R.run_jobs(jobs,d/'native.json')
 for c in rows:
  old=c['target'];new=R.read(ROOT/c['case_path']/'target_refined.json')
  for axis in R.AXES:
   # The old and new enclosures contain the same mathematical target.
   from fractions import Fraction as Q
   a=old['axes'][axis]['bias_interval'];b=new['axes'][axis]['bias_interval']
   if max(Q(a['lower']),Q(b['lower']))>min(Q(a['upper']),Q(b['upper'])):raise ValueError('NUMERICAL_REFINEMENT_INCONSISTENT')
 R.save(d/'verification.json',dict(status='PASS',refined=len(rows),old_raw_unchanged=True,old_new_intervals_overlap=True))
 R.summarize(n)
if __name__=='__main__':
 mode=sys.argv[1];n=int(sys.argv[2]) if len(sys.argv)>2 else None
 if mode=='refine':refine(n)
 elif mode=='adapt':
  round_no=int(sys.argv[3]);b=R.adapt(n,f'batch{round_no+1:02d}',round_no)
  if b:R.execute(b);refine(n)
 elif mode=='confirm':R.confirm(n)
 elif mode in ['extend','tables']:
  s=importlib.util.spec_from_file_location('hx_refined_finalizer',ROOT/'scripts/finalize_pauli_h2_h6_study.py');F=importlib.util.module_from_spec(s);s.loader.exec_module(F);F.R=R
  if mode=='extend':F.extend(n)
  else:R.save(R.OUT/'H2_H6_comparison_core.json',F.tables())
 else:raise ValueError(mode)
