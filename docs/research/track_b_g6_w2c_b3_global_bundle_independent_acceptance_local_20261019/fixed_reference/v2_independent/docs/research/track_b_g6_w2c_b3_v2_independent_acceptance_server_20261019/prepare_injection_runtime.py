"""Materialize exact fixed bytes for fresh artificial faults; no execution."""
from hashlib import sha256
from pathlib import Path
import json,subprocess
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    inventory=json.loads((HERE/'injection_runtime_inventory.json').read_text());dest=HERE/'injection_runtime';dest.mkdir(exist_ok=False)
    for row in inventory['files']:
        raw=subprocess.check_output(['git','show',inventory['measured_SOURCE']+':'+row['path']],cwd=HERE)
        assert len(raw)==row['bytes'] and sha256(raw).hexdigest()==row['sha256']
        path=dest/row['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    print('Fixed artificial runtime files:',len(inventory['files']))
