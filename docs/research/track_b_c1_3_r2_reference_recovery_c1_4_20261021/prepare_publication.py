"""Git/text transport only, after science. No scientific import/recalculation.

Create bounded text batches to carry exact public files through the connector.
The caller stages explicit new-namespace paths before invoking this utility.
"""
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNTIME = ROOT.parent / (ROOT.name + '.runtime')

def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()

def main():
    names = git('diff', '--cached', '--name-only', 'HEAD').splitlines()
    assert names and all(name.startswith(str(HERE.relative_to(ROOT)) + '/') for name in names)
    directory = RUNTIME / 'result_transport'; directory.mkdir(exist_ok=False)
    metadata, parts, batches = [], [], 0
    for index, name in enumerate(names):
        content = (ROOT / name).read_text()
        metadata.append({'path': name, 'chars': len(content), 'blob': git('hash-object', name)})
        for offset in range(0, len(content), 12000):
            parts.append({'fi': index, 'offset': offset, 'content': content[offset:offset+12000]})
            if len(parts) == 6:
                (directory / f'batch_{batches:04d}.json').write_text(json.dumps(parts, ensure_ascii=False))
                batches += 1; parts = []
    if parts:
        (directory / f'batch_{batches:04d}.json').write_text(json.dumps(parts, ensure_ascii=False)); batches += 1
    result = {'files': metadata, 'parent': git('rev-parse', 'HEAD'), 'base_tree': git('rev-parse', 'HEAD^{tree}'),
              'tree': git('write-tree'), 'batches': batches, 'transport_directory': str(directory)}
    (RUNTIME / 'result_payload_meta.json').write_text(json.dumps(result))
    print(json.dumps({'files': len(names), 'text_batches': batches, 'tree': result['tree']}))

if __name__ == '__main__': main()
