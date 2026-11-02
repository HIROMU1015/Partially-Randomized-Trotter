import pathlib,json,subprocess,hashlib
import argparse
P=pathlib.Path(__file__).resolve().parent;W=P.parents[2];H='62967269e611b61e73b70abe20e998d9c7eaaf50';M='deade78a19f5b23859d16816d3a7bfb37765dd80';DOC='docs/research/track_b_g6_w2c_b3_closed_port_bundle_integration_20261019'
ap=argparse.ArgumentParser();ap.add_argument('--new-runtime',required=True);a=ap.parse_args();D=pathlib.Path(a.new_runtime).resolve();assert not D.exists() and not D.is_relative_to(W) and not W.is_relative_to(D)
c=dict(R=str(W),H=H,M=M);C=json.loads((P/'fixed_reference'/H/DOC/'technical_contract_candidate.json').read_text());D.mkdir();subprocess.run(['git','init','-q',str(D)],check=True);common=subprocess.check_output(['git','-C',str(W),'rev-parse','--git-common-dir'],text=True).strip();(D/'.git/objects/info/alternates').write_text(str((W/common).resolve()/'objects')+'\n');subprocess.run(['git','-C',str(D),'update-ref','refs/heads/frozen-artificial',M],check=True);subprocess.run(['git','-C',str(D),'symbolic-ref','HEAD','refs/heads/frozen-artificial'],check=True)
paths=dict(C['source_pins']);paths.update(C['scientific_reference']['input_pins'])
for p,h in paths.items():
 b=(P/'fixed_reference'/c['H']/p).read_bytes();assert hashlib.sha256(b).hexdigest()==h;d=D/p;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(b)
print('sparse runtime',len(paths))
