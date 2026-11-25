"""Git transport layout only; no scientific provider is invoked."""
from pathlib import Path
import json
import runpy
import subprocess


def test_profile_flattening_preserves_fixed_heads_and_unrelated_paths(tmp_path):
    api=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/stage_rpe_pbs_flat_transport.py'))
    def git(cwd,*args):return subprocess.check_output(['git','-C',str(cwd),*args],text=True).strip()
    base=tmp_path; first=base/'source';first.mkdir();git(first,'init','-q');git(first,'config','user.name','test');git(first,'config','user.email','test@example.invalid')
    (first/'data').write_text('fixed');git(first,'add','data');git(first,'commit','-qm','source');head=git(first,'rev-parse','HEAD')
    root=first
    for i in range(4):
        child=base/str(i);git(base,'clone','--shared','-q',str(root),str(child));root=child
    alternate=root/'.git/objects/info/alternates';alternate.write_text(alternate.read_text()+'\n')
    profile=base/'profile.json';profile.write_text(json.dumps({'root':str(root),'route_views':{'old':str(first)},'work_dir':str(base/'spool')}))
    report=api['stage_profile'](base,base/'flat',profile,minimum_depth=2)
    assert list(report['paths'])==[str(root)] and report['alternate_depth']==1
    result=json.loads(Path(report['profile']).read_text());assert result['route_views']['old']==str(first) and result['work_dir']==str(base/'spool')
    assert git(result['root'],'rev-parse','HEAD')==head and (Path(result['root'])/'data').read_text()=='fixed'
    assert git(root,'rev-parse','HEAD')==head and not report['issues']
