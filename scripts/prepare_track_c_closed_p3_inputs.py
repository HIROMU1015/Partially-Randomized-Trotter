"""Restore the published fixed scientific inputs; no scientific recomputation."""
from pathlib import Path
import hashlib
import json
import pickle
import subprocess
from trottertracks.track_c_closed_p3_inputs import load_model,model_identity
root=Path(__file__).resolve().parents[1]
source=root/'artifacts/track_c_closed_p3/2026-11-08'
cache=root/'.cache/closed_p3/inputs';cache.mkdir(parents=True,exist_ok=True)
if not (cache/'common.pkl').exists():
    c=load_model(source)
    with (cache/'common.pkl').open('wb') as f:pickle.dump(c,f)
    print('Restored fixed model',model_identity(c))
for name,path in [('policy','artifacts/track_c_diagonal_fusion/2026-11-08/POLICY.json'),('final','artifacts/track_c_diagonal_fusion/2026-11-08/FINAL.json'),('joint_certificate','artifacts/track_c_joint_isometry/2026-11-08/CERTIFICATE.json')]:
    if not (cache/(name+'.json')).exists():(cache/(name+'.json')).write_bytes(subprocess.check_output(['git','show','94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9:'+path],cwd=root))
if not (cache/'catalogue.json').exists():
    import io,tarfile
    raw=subprocess.check_output(['git','show','94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9:artifacts/track_c_joint_isometry/2026-11-08/RAW_DATA.tar.gz'],cwd=root)
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tf:
        candidates=[m for m in tf.getmembers() if m.name.endswith('/CATALOGUE_FINAL.json') or m.name=='CATALOGUE_FINAL.json']
        assert len(candidates)==1
        (cache/'catalogue.json').write_bytes(tf.extractfile(candidates[0]).read())
print('Inputs ready; original private pickle was not required or modified.')
