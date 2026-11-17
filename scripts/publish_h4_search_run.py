"""Prepare a controller's saved-result index for the ordinary C 10-minute sync.

This does not run, modify, or stop the controller, nor commit or push for it.
Publish the generated file with the referenced metadata via ordinary Git.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

FILES = ('ROUND_RESULT.json', 'ROUND_RESULT_INTERRUPTED.json', 'EVALUATIONS.json',
         'NEEDS.json', 'SCOPE_METADATA.json', 'SEARCH_STATE.json', 'STATE.json',
         'ROUND.json', 'STATUS.json', 'LIVE_START_SNAPSHOT.json')


def prepare(root, run_directory, output, *, run_id=None, execution_hosts=()):
    root=Path(root).resolve(); run=Path(run_directory).resolve(); output=Path(output).resolve()
    relative=run.relative_to(root).as_posix()
    if not relative.startswith('docs/research/'):
        raise ValueError('Publish owned controller metadata under docs/research; private paths are not scanned')
    out_relative=output.relative_to(root).as_posix()
    if not out_relative.startswith('docs/research/')or output.name!='SEARCH_PUBLICATION.json':
        raise ValueError('Public index must be docs/research/.../SEARCH_PUBLICATION.json')
    files={}
    for name in FILES:
        p=run/name
        if p.exists():
            value=json.loads(p.read_text())
            if value.get('artificial_only') or value.get('artificial'):
                raise ValueError('Artificial fixtures cannot be published to the production map')
            files[name]=dict(path=p.relative_to(root).as_posix(),
                            sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    if not files:raise ValueError('No saved controller metadata found')
    # No host guess from absolute run paths or usernames. A producer may list
    # several executor hosts, but it publishes just one controller adoption.
    value=dict(schema='H4_SEARCH_PUBLICATION_V1',role='CONTROLLER',controller_adopted=True,
        run_id=run_id or relative,run_directory=relative,execution_hosts=sorted(set(execution_hosts)),
        files=files,scope='SAVED_METADATA_ONLY_NO_SCIENTIFIC_EXECUTION',
        new_scientific_calls=0)
    text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if output.exists()and output.read_text()==text:return 'UNCHANGED',value
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(text)
    return 'UPDATED',value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository',type=Path,default=Path.cwd())
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,
                   help='Publish as docs/research/.../SEARCH_PUBLICATION.json')
    p.add_argument('--run-id');p.add_argument('--execution-host',action='append',default=[])
    a=p.parse_args();root=a.repository.resolve()
    def path(v):return v if v.is_absolute()else root/v
    status,value=prepare(root,path(a.run_dir),path(a.output),run_id=a.run_id,execution_hosts=a.execution_host)
    print(json.dumps(dict(status=status,run_id=value['run_id'],files=len(value['files']),
        next='Commit/push this index and referenced controller metadata on any public branch; C ordinary sync detects it')))


if __name__=='__main__':main()
