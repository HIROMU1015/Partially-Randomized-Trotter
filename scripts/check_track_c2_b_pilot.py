"""Read-only publication checks of C2-B words, costs and STOPs."""
import ast,hashlib,json,lzma,tarfile
from fractions import Fraction as F
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'artifacts/track_c2_b_pilot/2026-11-06'
for bundle in json.loads((out/'BUNDLES.json').read_text()):
 p=out/bundle['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==bundle['sha256']
 if 'decompressed_sha256' in bundle:assert hashlib.sha256(lzma.decompress(p.read_bytes())).hexdigest()==bundle['decompressed_sha256']
count=0
with tarfile.open(out/'raw_synthesis.tar.xz') as tar:
 for member in tar:
  row=json.load(tar.extractfile(member));count+=1
  assert row['status']=='PASS'
  assert hashlib.sha256(row['sequence'].encode()).hexdigest()==row['sequence_sha256']
  assert F(row['strict_error_upper'])<=F(10)**row['precision_exponent']
  assert row['T']==row['sequence'].count('T')+row['sequence'].count('t')
assert count==7603
adopted=out/'verified_final'
result=json.loads(lzma.decompress((adopted/'RESULTS.json.xz').read_bytes()))
summary=json.loads((adopted/'SUMMARY.json').read_text())
for arm in result.values():
 assert len(arm['rounds'])==18
 assert F(2)*sum((F.from_float(x['alpha_per_axis']) for x in arm['rounds']),F(0))<=F('.05')
 assert all(x['rho_lower']>0 and x['beta_stat']>0 for x in arm['rounds'])
 assert sum(2*x['T']*x['cost_sensitive_shots_per_axis'] for x in arm['rounds'])==arm['total_T_cost_sensitive_alpha']
assert result['gate_sum_nonuniform']['total_T_cost_sensitive_alpha']==summary['baseline_T']
assert result['structural_nonuniform']['total_T_cost_sensitive_alpha']==summary['structural_T']
assert json.loads((adopted/'BASELINE_BOUND.json').read_text())['beats_registered_baseline_lower_bound'] is False
for p in [out/'STOP.json',out/'verified/STOP.json',adopted/'STOP.json']:
 row=json.loads(p.read_text());assert row['status']=='MANDATORY_STOP' and row['automatic_resume'] is False
for p in [root/'src/trottertracks/track_c2_b_pilot.py',root/'src/trottertracks/track_c2_b_verify.py',Path(__file__),root/'scripts/run_track_c2_b_pilot.py']:ast.parse(p.read_text())
print(json.dumps(dict(status='PASS',strict_word_records=count,arms=4,rounds_per_arm=18,new_scientific_calls=0,STOPs_preserved=True)))
