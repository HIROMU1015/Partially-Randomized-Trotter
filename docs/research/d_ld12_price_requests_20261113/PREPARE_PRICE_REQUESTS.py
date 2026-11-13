"""Publish fixed-menu request selectors and coverage. No scientific entrypoints."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import ld12_price_requests as L


def generate():
    files = ('src/trottertracks/candidate_price_adapter.py', 'src/trottertracks/ld12_price_requests.py')
    source = dict(purpose='TECHNICAL_METADATA_PREPARATION_ONLY_NOT_A_SCIENCE_RUN',
        D_adapter_base=L.D_BASE, A_menu=L.A_SOURCE,
        source_blobs={name: subprocess.check_output(['git','hash-object',name],cwd=ROOT).decode().strip() for name in files})
    result = L.prepare(ROOT, execution_source=source)
    common = {k: v for k, v in result.items() if k not in ('request_index','coverage')}
    artifacts = {'PRICE_REQUEST_INDEX.json':dict(common, requests=result['request_index']),
        'PRICE_REQUEST_COVERAGE.json':dict(common, cells=result['coverage'])}
    for name, value in artifacts.items():
        target=OUT/name
        if target.exists():
            if json.loads(target.read_text()) != value:
                raise ValueError('Existing metadata differs; publish to a new SOURCE/output')
        else:
            with target.open('x') as f:
                json.dump(value,f,indent=2,allow_nan=False,ensure_ascii=False);f.write('\n')
    print('26 paired request selectors / 52 axis coverage cells; no scientific calls')


if __name__=='__main__':
    generate()
