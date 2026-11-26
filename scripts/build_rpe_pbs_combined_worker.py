#!/usr/bin/env python3
"""Add shared preparation acquisition to a fixed PBS Q3-collection zipapp."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
PBS_TRANSPORT = 'trottertracks/_collection_pbs_worker.py'
Q3_TRANSPORT = 'trottertracks/_collection_q3_worker.py'
PREPARATION = 'trottertracks/_pipeline_pbs_preparation_batch.py'


def build(base, base_sha256, output_directory, python):
    if not Path(python).is_absolute() or any(c.isspace() for c in str(python)):
        raise ValueError('Use an absolute PBS Python path without whitespace')
    original = Path(base).read_bytes()
    if hashlib.sha256(original).hexdigest() != base_sha256:
        raise ValueError('PBS worker base hash differs')
    result = io.BytesIO()
    result.write(f'#!{python}\n'.encode())

    def write(target, name, content):
        target.writestr(ZipInfo(name), content, compress_type=ZIP_DEFLATED)

    with ZipFile(io.BytesIO(original)) as source, ZipFile(result, 'w', ZIP_DEFLATED) as target:
        transport = source.read(PBS_TRANSPORT)
        if (b'batch_q3_completions' not in transport or b'def _pump_collection(' not in transport
                or b'def _launch_buffered(' not in transport
                or b'pbs_evidence_status_batch' not in source.read(Q3_TRANSPORT)):
            raise ValueError('Expected the PBS worker with batched Q3 result collection')
        for item in sorted(source.namelist()):
            if item not in ('__main__.py', PREPARATION):
                write(target, item, source.read(item))
        write(target, '__main__.py', (ROOT/'scripts/run_rpe_pbs_storage_worker.py').read_bytes())
        write(target, PREPARATION, (ROOT/'src/trottertracks/rpe_pbs_preparation_batch.py').read_bytes())
    content = result.getvalue()
    sha256 = hashlib.sha256(content).hexdigest()
    directory = Path(output_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'pilot-worker-{sha256[:12]}.pyz'
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('Existing PBS worker package differs')
    else:
        with path.open('xb') as output:
            output.write(content)
        path.chmod(0o700)
    return {'path': str(path), 'sha256': sha256, 'base_sha256': base_sha256}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--base-sha256', required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True, help='Fixed PBS interpreter for the executable launcher')
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.base_sha256, args.output_directory, args.python), indent=2))


if __name__ == '__main__':
    main()
