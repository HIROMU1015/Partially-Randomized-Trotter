import pathlib,json,subprocess,hashlib
import argparse
P=pathlib.Path(__file__).resolve().parent;W=P.parents[2];H='82896d53fa97235fa848d6c0833eb193af8b99d6';M='78c571005fe766c61b32dc867e006531099ff608'
Q=P/'fixed_reference'/H/'docs/research/track_b_g6_w2c_b3_real_api_binding_preflight_candidate_20261019';candidate=json.loads((Q/'technical_contract_candidate.json').read_text())
p=argparse.ArgumentParser();p.add_argument('--new-runtime',required=True);a=p.parse_args();dest=pathlib.Path(a.new_runtime).resolve();assert not dest.exists() and not dest.is_relative_to(W) and not W.is_relative_to(dest)
c=dict(R=str(W),M=M,H=H);dest.mkdir();subprocess.run(['git','init','-q',str(dest)],check=True)
common=subprocess.check_output(['git','-C',c['R'],'rev-parse','--git-common-dir'],text=True).strip();(dest/'.git/objects/info/alternates').write_text(str((pathlib.Path(c['R'])/common).resolve()/'objects')+'\n');subprocess.run(['git','-C',str(dest),'update-ref','refs/heads/frozen-artificial-source',c['M']],check=True);subprocess.run(['git','-C',str(dest),'symbolic-ref','HEAD','refs/heads/frozen-artificial-source'],check=True)
paths=dict(candidate['source_pins']);paths.update(candidate['scientific_reference']['input_pins']);rows=[]
for p,h in paths.items():
 b=(P/'fixed_reference'/c['H']/p).read_bytes();assert hashlib.sha256(b).hexdigest()==h
 d=dest/p;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(b);rows.append(dict(path=p,sha256=h,bytes=len(b)))
(dest/'runtime_preparation.json').write_text(json.dumps(dict(measured_SOURCE=c['M'],files=rows,science_imports=0),indent=2));print('runtime files',len(rows))
