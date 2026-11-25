"""Technical codec/dispatch checks: synthetic events, no sampling or science."""
import ast
from contextlib import ExitStack
from pathlib import Path
import hashlib
import json
import subprocess
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from trottertracks import rpe_df_price_flanks as E
from trottertracks import pr_candidate_completion as C
from trottertracks.s0a_saved_structure import SavedEvents


def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)


def event(i):
    return dict(taylor_order=0, component_id=f'fragment-{i}', rotation_angle=-0.0 if i==0 else .125,
                phase={'real':1.,'imag':-0.0}, application_sequence=[dict(role='rotation',is_identity=False,
                    basis_id='基底', basis_change_operations=[['x',j] for j in range(12)])])


def test_dictionary_exact_roundtrip_and_legacy_reader(tmp_path):
    rows=[[event(0),event(1)]*15,[],[event(1)]*3,[event(0)]*20]
    doc=dict(events=rows,seeds=[1,2,3,4],laws=[{'parameter':-0.0}],conditional=False)
    expected=hashlib.sha256(canonical(rows).encode()).hexdigest()
    path=tmp_path/'events_PRIVATE.json'
    footer=E.write_event_dictionary(path,doc,block_size=8)
    assert footer['event_digest']==expected
    assert path.stat().st_size<len(canonical(doc).encode())
    reader=E.DictionaryEvents(path,expected,chunk=11)
    restored=[list(row) for row in reader.batches()]
    assert canonical(restored)==canonical(rows)  # includes signed zero and all fields
    assert reader.metadata=={k:v for k,v in doc.items() if k!='events'}
    assert reader.counts=={'rows':4,'events':53}
    restored[0][0]['phase']['real']=123
    assert restored[0][2]['phase']['real']==1
    legacy=tmp_path/'legacy.json'
    assert E.expand_event_dictionary(path,legacy,expected_digest=expected)==expected
    assert canonical(json.loads(legacy.read_text()))==canonical(doc)
    for file in (path,legacy):
        calls=[];r=SavedEvents(file,1,[30,0,3,20],expected,lambda i,e:calls.append((i,e['component_id'])),chunk=19)
        assert canonical([list(batch) for batch in r.batches()])==canonical(rows)
        assert r.digest==expected and len(calls)==53
    with pytest.raises(FileExistsError):E.write_event_dictionary(path,doc)


def test_dictionary_rejects_corrupted_reference_and_digest(tmp_path):
    path=tmp_path/'events_PRIVATE.json';doc=dict(events=[[event(0)]*4],seeds=[1],laws=[],conditional=False)
    E.write_event_dictionary(path,doc,block_size=2)
    bad=json.loads(path.read_text());bad['blocks'][0]['segments'][0][0][0]=999
    # Keep header order, as the versioned writer does.
    path.write_text(json.dumps(bad,separators=(',',':')))
    with pytest.raises(ValueError,match='indices'):[list(row) for row in E.DictionaryEvents(path).batches()]
    good=tmp_path/'good.json';E.write_event_dictionary(good,doc)
    with pytest.raises(ValueError,match='digest'):[list(row) for row in E.DictionaryEvents(good,'0'*64).batches()]
    reader=E.DictionaryEvents(good);rows=reader.batches();next(rows)
    with pytest.raises(ValueError,match='not fully consumed'):next(rows)


def test_event_views_do_not_materialize_until_consumed():
    items=[Mock(to_dict=Mock(return_value=event(i))) for i in range(4)]
    requests=[SimpleNamespace(rte_occurrence=SimpleNamespace(events=items[:2])),
              SimpleNamespace(rte_occurrence=SimpleNamespace(events=items[2:]))]
    view=E.EventRows(requests);flat=E.FlatEvents(view)
    assert len(flat)==4 and all(not x.to_dict.called for x in items)
    assert view[1][0]==event(2) and items[2].to_dict.call_count==1
    expected=[[event(0),event(1)],[event(2),event(3)]]
    assert E.streaming_digest(view)==hashlib.sha256(canonical(expected).encode()).hexdigest()
    assert canonical(list(flat))==canonical([e for row in expected for e in row])


def test_known_fixed_sources_match_lazy_representation_only():
    root=Path(__file__).resolve().parents[1]
    read=lambda rev,path:subprocess.check_output(['git','show',rev+':'+path],cwd=root)
    raw=read('ea9e66092465d39dfcd4aeb6fea8e9f09808cabb','scripts/run_pr_df_ld_split.py')
    source=ast.parse(raw);updated=E._legacy_event_tree(source)
    assert updated is not source
    fn=next(n for n in updated.body if isinstance(n,ast.FunctionDef) and n.name=='acquire_path')
    assert '_event_rows(requests)' in ast.unparse(fn) and '_flat_events(eventrows)' in ast.unparse(fn)
    raw=read('e827367629ca6c2920d9d908fc1631314399db86','src/trottertracks/compiled_price_fast.py')
    fn=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='make_backend_class')
    _,counts=E._event_materialization(fn)
    assert counts==(1,0)
    compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'fixture','exec')


def test_empty_and_unique_event_rows_are_lossless(tmp_path):
    for i,rows in enumerate(([],[[],[]],[[event(j) for j in range(33)]])):
        path=tmp_path/str(i);doc=dict(events=rows,seeds=[],laws=[],conditional=False)
        E.write_event_dictionary(path,doc,block_size=3)
        assert canonical([list(r) for r in E.DictionaryEvents(path).batches()])==canonical(rows)


def test_pairing_respects_existing_axis_owner_plan(tmp_path):
    import run_pr_df_candidate_prices as runner
    paired=E.pair_pending_axis_jobs(runner.pending_axis_jobs)
    both={a:{'status':'MISSING'} for a in ('X','Y')}
    cell=dict(m=3,request=dict(request_id='test',axes=['X','Y']))
    jobs=paired(cell,0,tmp_path,deepcopy(both),True)
    assert [job[0]['request']['axes'] for job in jobs]==[['X','Y']]
    assert not jobs[0][0].get('parallel_axis_group')
    (tmp_path/'STARTED_m3_p0_axesX.json').touch()
    state=deepcopy(both);jobs=paired(cell,0,tmp_path,state,True)
    assert [job[0]['request']['axes'] for job in jobs]==[['Y']]
    assert jobs[0][0]['parallel_axis_group'] and state['X']['status'].startswith('HELD_PRIOR')
    (tmp_path/'STOP_m3_p0_axesY.json').touch()
    assert paired(cell,0,tmp_path,deepcopy(both),True)==[]


def test_exploratory_defers_all_df_mappers_but_keeps_original_owner(tmp_path):
    quote={'complete':False}
    for request in ({'method':'PR_DF'},{'method':'PR_DF','df_max_width_source':'mapper'}):
        assert C.defer_new_df_price(request,quote,tmp_path,exploratory=True)
        assert not C.defer_new_df_price(request,quote,tmp_path,exploratory=False)
    assert not C.defer_new_df_price({'method':'PR_PAULI'},quote,tmp_path,exploratory=True)
    assert not C.defer_new_df_price({'method':'PR_DF'},{'complete':True},tmp_path,exploratory=True)
    owned=tmp_path/'df_price_groups'/'group';owned.mkdir(parents=True);(owned/'DF_CALIBRATION.DISTRIBUTED.json').write_text('{}')
    assert not C.defer_new_df_price({'method':'PR_DF'},quote,tmp_path,exploratory=True)
