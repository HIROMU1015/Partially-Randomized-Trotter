"""Prove forward-only q/R metadata enrichment on a read-only saved-store copy.

Reads fixed configuration and receipts; no scientific parser, compiler, fitting,
optimizer or scientific evaluation executes. The live store is never opened RW.
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

RULES='c1f95e0806f5478a1ee5ce9ecb148ce9d10461aa'
PUBLICATION='ca5905e79d7654bdfe42a679b7b5e74aef80b1be'
INDEX='docs/research/h4_search_map_20261117/INDEX.json'


def records(path):
    connection=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
    try:return connection.execute('SELECT id,identity,record FROM models ORDER BY id').fetchall()
    finally:connection.close()


def check(code_root,source,publication_repo,output):
    code_root=Path(code_root).resolve();source=Path(source).resolve()
    sys.path[:0]=[str(code_root/'src'),str(code_root/'scripts')]
    from trottertracks import rpe_price_validation as V
    scope={'__name__':'fixed_price_rules','__file__':'fixed_price_rules'}
    exec(compile(subprocess.check_output(['git','show',RULES+':src/trottertracks/pr_price_rules.py'],cwd=code_root),'fixed','exec'),scope)
    reader=scope['GitReader'](code_root)
    registry=reader.read(dict(commit=RULES,path='configs/pr_price_rules/h4_saved_rules.json'))
    entry=next(e for e in registry['entries']if e['name']=='PAULI_LD2')
    rules=scope['PriceRules'](reader,dict(entries=[entry]));configuration=rules.entries[0]['configuration']
    quote=rules.quote(dict(configuration=configuration,implementation=entry['implementation'],policy='MODEL'))
    desc=V.describe(rules,configuration,entry['name'],quote)[0]
    before=records(source);saved=next(json.loads(raw)for mid,_,raw in before if mid==desc['model_id'])
    published=json.loads(subprocess.check_output(['git','show',PUBLICATION+':'+INDEX],cwd=publication_repo))
    accepted=published['price_validation_models'][desc['model_id']]
    assert saved['identity']==accepted['identity']==desc['identity']
    updates=V.saved_point_metadata_updates(saved,desc,reader)
    expected=deepcopy(saved)
    for update in updates:
        point=next(p for p in expected['validation_points']if p['m']==update['m']and p['source']==update['source'])
        point.update(update['fields'])
    assert expected['validation_points']==accepted['validation_points']
    with tempfile.TemporaryDirectory(prefix='rpe-saved-point-metadata-')as temporary:
        directory=Path(temporary);target=directory/'models.sqlite3'
        old=sqlite3.connect(source.as_uri()+'?mode=ro',uri=True);new=sqlite3.connect(target)
        try:old.backup(new)
        finally:new.close();old.close()
        service=V.Service(code_root,directory/'artificial-observer',dict(science_approved=False,
            price_validation=dict(store=str(target),publish_shared_models=True,enrich_saved_evidence_metadata=True)))
        candidate=dict(configuration_id='ARTIFICIAL_METADATA_OBSERVER_ONLY',family='PR',method='PR_PAULI',
            prices=[1]*26,price_model_validations=[desc],ready=True,needs=[])
        with patch.object(service,'resolve',side_effect=AssertionError('Independent evidence scan forbidden'))as resolve, \
                patch.object(service,'_submit',side_effect=AssertionError('Compiler dispatch forbidden'))as submit:
            service.observe([candidate]);service.observe([candidate])
            assert resolve.call_count==submit.call_count==0
        assert service.model_holds==set()
        after=records(target)
        assert len(after)==len(before)==13
        for original,current in zip(before,after):
            assert original[:2]==current[:2]
            if original[0]==desc['model_id']:assert json.loads(current[2])==expected
            else:assert original==current
        view=json.loads((service.run/'PRICE_MODEL_VALIDATIONS.json').read_text())
        for model in view['models']:
            public=published['price_validation_models'][model['model_id']]
            for key in ('identity','status','qualified_for_candidate','validation_points'):
                assert model.get(key)==public.get(key),(model['model_id'],key)
    assert records(source)==before
    result=dict(status='PASS',kind='SAVED_SOURCE_METADATA_ON_COPY_ONLY',
        code_root=str(code_root),code_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code_root,text=True).strip(),
        implementation_worktree_changed=bool(subprocess.check_output(['git','diff','HEAD','--name-only','--','src','scripts','tools','configs','schemas'],cwd=code_root,text=True).strip()),
        source_store=str(source),source_store_record_bytes_unchanged=True,live_writes=0,
        fixed_price_rules_source=RULES,configuration_source=desc['configuration_source'],
        model_id=desc['model_id'],saved_status=saved['status'],saved_record_sha256=V.digest(saved),
        enriched_record_sha256=V.digest(expected),copy_only_updates=updates,
        all_other_records_byte_identical=True,advanced_m9_evidence_preserved=True,
        observations_predictions_coefficients_qualification_owners_attempts_partials_preserved=True,
        accepted_C_source=PUBLICATION,accepted_C_index=INDEX,all_13_models_match_C_science_and_metadata=True,
        saved_reuse_is_new_independent_validation=False,new_scientific_calls=0,new_compile_requests=0,
        performed_by='Codex sub-agent price_store_acceptance')
    Path(output).write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root',type=Path,default=Path(__file__).resolve().parents[4])
    parser.add_argument('--source-store',type=Path,required=True)
    parser.add_argument('--publication-repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();print(json.dumps(check(a.code_root,a.source_store,a.publication_repo,a.output)))
