"""Reuse the controller's saved LD2 model through make_executor; no science.

The real store is opened read-only and copied. Only the normal Service and
factory execute; the optimizer, compiler dispatch and evidence scan are blocked.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from unittest.mock import patch


def records(path):
    connection=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
    try:return connection.execute('SELECT id,identity,record FROM models ORDER BY id').fetchall()
    finally:connection.close()


def check(code_root,source,output):
    code_root=Path(code_root).resolve();source=Path(source).resolve()
    sys.path[:0]=[str(code_root/'src'),str(code_root/'scripts')]
    import run_rpe_method_optimizer as CLI
    from trottertracks import rpe_price_validation as V
    before=records(source)
    saved=next(json.loads(raw)for _,_,raw in before if json.loads(raw).get('registry')=='PAULI_LD2')
    assert saved['status']=='INDEPENDENT_MEASURED'and saved['evidence_parser_version']==4
    desc=deepcopy(saved)
    desc.update(configuration_id='ARTIFICIAL_REUSE_OF_SAVED_LD2',points=deepcopy(saved['validation_points']))
    candidate=dict(configuration_id=desc['configuration_id'],method='PR_PAULI',family='PR',
        price_model_validations=[desc],caps=[4]*13,ready=True,needs=[],
        prices=[1]*26)  # Artificial observer context; never priced or published.
    class Executor:
        def __init__(self,*args,**kwargs):pass
        def finish(self):pass
    def artificial_search(*args,**kwargs):
        assert kwargs['context']['candidates'][0]['ready']
        return {'PR_PAULI':dict(stop_reason='ARTIFICIAL_SEARCH_ONLY',new_calls=0)}
    with tempfile.TemporaryDirectory(prefix='rpe-saved-ld2-')as temporary:
        directory=Path(temporary);target=directory/'models.sqlite3'
        old=sqlite3.connect(source.as_uri()+'?mode=ro',uri=True);new=sqlite3.connect(target)
        try:old.backup(new)
        finally:new.close();old.close()
        views=[]
        for label,vector in [('first',[1]*26),('another-allocation',[7]*26)]:
            run=directory/label
            spec=dict(methods=['PR_PAULI'],max_new_q3_per_method={'PR_PAULI':0},science_approved=False,
                price_validation=dict(store=str(target),publish_shared_models=True,
                    source='a18d5a04df5ec8c789f274730bff87a2c0a7cc89'))
            candidate['validation_vector']=vector
            with patch.object(CLI.O,'Executor',Executor),patch.object(CLI.O,'optimize',side_effect=artificial_search), \
                    patch.object(V.Service,'_submit',side_effect=AssertionError('Compiler dispatch forbidden'))as submit, \
                    patch.object(V.Service,'resolve',side_effect=AssertionError('Saved evidence re-scan forbidden'))as resolve:
                executor,optimize=CLI.make_executor(code_root,run,{}, {}, {},spec)
                try:result=optimize(code_root,{},context={'candidates':[candidate]})
                finally:executor.finish()
                assert submit.call_count==resolve.call_count==0
                assert result['PR_PAULI']['new_calls']==0
            view=json.loads((run/'PRICE_MODEL_VALIDATIONS.json').read_text())
            model=next(record for record in view['models']if record['model_id']==saved['model_id'])
            assert model==saved and 'job_id'not in model
            assert view['model_scope']=='CONTROLLER_SHARED_MODEL_STORE'
            assert candidate['price_validation'][saved['model_id']]['status']=='INDEPENDENT_MEASURED'
            assert len(view['applications'])==1 and view['applications'][0]['model_id']==saved['model_id']
            views.append(dict(run=label,models=len(view['models']),applications=len(view['applications']),
                              saved_model_unchanged=True,new_compile_requests=0,new_q3_calls=0))
        assert records(target)==before
    assert records(source)==before
    result=dict(status='PASS',kind='SAVED_METADATA_WITH_ARTIFICIAL_OPTIMIZER_ONLY',
        code_root=str(code_root),code_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code_root,text=True).strip(),
        implementation_worktree_changed=bool(subprocess.check_output(['git','diff','HEAD','--name-only','--','src','scripts','tools','configs','schemas'],cwd=code_root,text=True).strip()),
        source_store=str(source),model_id=saved['model_id'],registry=saved['registry'],
        saved_record_sha256=V.digest(saved),source_record_bytes_unchanged=True,all_shared_models_preserved=len(before),
        runs=views,new_evidence_scans=0,new_compile_requests=0,new_scientific_q3_calls=0,new_signal_calls=0,
        saved_reuse_is_new_independent_validation=False,performed_by='Codex sub-agent price_store_acceptance')
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root',type=Path,default=Path(__file__).resolve().parents[4])
    parser.add_argument('--source-store',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    arguments=parser.parse_args()
    print(json.dumps(check(arguments.code_root,arguments.source_store,arguments.output)))
