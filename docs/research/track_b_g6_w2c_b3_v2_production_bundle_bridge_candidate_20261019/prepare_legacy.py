"""Restore only fixed legacy artificial test dependencies into a new directory."""
from pathlib import Path
from hashlib import sha256
import json,subprocess,sys,tarfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];FIXED='49a0dcc9c8ec6dfc96eba5bb862cfdcb3d0fc866'
def main():
    target=HERE/'legacy_runtime';target.mkdir(exist_ok=False)
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',FIXED,'--','src/trottertracks','tests/tracks/algorithm_codesign','docs/research','artifacts/track_b_g7_budget_control_result/2026-10-10/v1/result_v1.json'],cwd=ROOT,text=True).splitlines()
    paths=[p for p in paths if p.startswith('src/trottertracks/') or (p.startswith('tests/tracks/algorithm_codesign/')and (p.endswith('.py')or'/fixtures/'in p)) or p.startswith('docs/research/track_b_g6_w2c_')or p.startswith('artifacts/track_b_g7_budget_control_result/')]
    inventory=[]
    proc=subprocess.Popen(['git','archive',FIXED,'--',*paths],cwd=ROOT,stdout=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout,mode='r|')as tar:
        for m in tar:
            if not m.isfile():continue
            if m.name not in paths or Path(m.name).is_absolute()or'..'in Path(m.name).parts:raise ValueError('UNSAFE_ARCHIVE')
            b=tar.extractfile(m).read();p=target/m.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);inventory.append(dict(path=m.name,bytes=len(b),sha256=sha256(b).hexdigest()))
    assert proc.wait()==0
    (HERE/'legacy_restoration.json').write_text(json.dumps(dict(commit=FIXED,files=inventory),indent=2)+'\n');print(len(inventory),'fixed files restored')
if __name__=='__main__':main()
