"""Materialize exact measured bytes and an owned Git object view; no execution."""
from pathlib import Path
from hashlib import sha256
import argparse,json,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
p=argparse.ArgumentParser();p.add_argument('--new-runtime',required=True);a=p.parse_args();dest=Path(a.new_runtime).resolve()
assert not dest.is_relative_to(ROOT) and not ROOT.is_relative_to(dest)
tree=HERE/'fixed_reference/bridge';doc=tree/'docs/research/track_b_g6_w2c_b3_v2_production_bundle_bridge_candidate_20261019'
c=json.loads((doc/'technical_contract_candidate.json').read_text());measured='e099dcc461247e81eff981867dc28c3f3b16df5e'
subprocess.run(['git','-C',str(ROOT),'cat-file','-e',measured+'^{commit}'],check=True)
dest.mkdir(exist_ok=False);subprocess.run(['git','init','-q',str(dest)],check=True)
objects=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','--git-common-dir'],text=True).strip();objects=(ROOT/objects).resolve()/'objects'
(dest/'.git/objects/info/alternates').write_text(str(objects)+'\n')
subprocess.run(['git','-C',str(dest),'update-ref','refs/heads/frozen-artificial-source',measured],check=True);subprocess.run(['git','-C',str(dest),'symbolic-ref','HEAD','refs/heads/frozen-artificial-source'],check=True)
paths=dict(c['source_pins']);paths.update(c['scientific_reference']['input_pins']);rows=[]
for path,h in sorted(paths.items()):
 b=(tree/path).read_bytes();assert sha256(b).hexdigest()==h
 actual=subprocess.check_output(['git','-C',str(ROOT),'show',measured+':'+path]);assert actual==b
 out=dest/path;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(b);rows.append(dict(path=path,sha256=h,bytes=len(b)))
(dest/'materialized_runtime_inventory.json').write_text(json.dumps(dict(measured_SOURCE=measured,files=rows,sparse_materialization=True,scientific_calls=0),indent=2)+'\n');print(json.dumps(dict(runtime=str(dest),files=len(rows))))
