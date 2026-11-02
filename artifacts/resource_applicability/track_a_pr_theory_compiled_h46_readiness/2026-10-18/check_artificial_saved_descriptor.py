"""Additional artificial end-to-end descriptor check. No real saved costs read."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[4]
spec=importlib.util.spec_from_file_location('prtc_test_fixture',ROOT/'tests/test_pr_theory_compiled_h46.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
m=t.m;runner=t.runner
case,theory,connections=t.SavedAdapter().fixture()
raw={'fake-theory':json.dumps(theory).encode(),'fake-connections':json.dumps(connections).encode()}
source=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
with tempfile.TemporaryDirectory(prefix='h46-descriptor-artificial-',dir='/home/AbeHiromu') as tmp:
    base=Path(tmp);output=base/'output';path=base/'descriptor.json'
    descriptor=dict(format='H46_SAVED_TABLES_V1',cases=[case])
    for kind,name in [('theory','fake-theory'),('connections','fake-connections')]:
        descriptor[kind]=dict(commit='1'*40,path='artifacts/'+name,sha256=hashlib.sha256(raw[name]).hexdigest())
    path.write_text(json.dumps(descriptor))
    rows=m.saved_table_rows(theory,connections,[case]);keys=[{k:r[k] for k in ('model','axis','metric','compiled_observation_id')} for r in rows]
    plan=dict(SOURCE=source,namespace=m.NAMESPACE,sealed=True,input_sha256=m.file_hash(path),mandatory_stop=True,next_stage_authorized=False,
        operations=['COMPARE_SAVED_VALUES'],source_hashes=t.source_hashes(),models=['A','B'],metrics=['RZ','CX'],
        output=str(output),registered_row_count=8,rows=keys)
    grant=dict(status='APPROVED',approved=True,runtime_authorization=True,SOURCE=source,namespace=m.NAMESPACE,
        plan_digest=m.digest(plan),input_sha256=plan['input_sha256'],one_shot=True,automatic_retry=False)
    def fake_git(args,**kwargs):
        if 'rev-parse' in args:return source
        return raw[args[-1].split('/')[-1]]
    with patch.object(runner.subprocess,'check_output',side_effect=fake_git):
        result=runner.run(plan,grant,path,output)
    assert result['status']=='COMPLETE_MANDATORY_STOP' and result['rows_completed']==8
    print(json.dumps(dict(status='PASS_ARTIFICIAL_SAVED_DESCRIPTOR',rows=8,real_saved_tables_opened=False,new_scientific_calls=0)))
