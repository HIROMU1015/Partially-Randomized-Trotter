"""Selection and resume boundaries without scientific providers or numeric H4 calls."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from trottertracks import pf_candidate_selection as S


def candidate(cid, p, formula='2nd', shift='100', t2=()):
    return dict(configuration_id=cid,method='TRUNCATED_PF',key=dict(p=p,formula=formula,delta_over_h='2'),
                ready=False,needs=['SIGNAL_13ROUND_PACKET','PRICE_26AXIS'],
                configuration_body=dict(id=cid,spec=dict(split=dict(original_PF_method='B0I'))),
                priority_evidence=dict(T1=dict(delta_E_trunc_B0I_ha=shift),T2=list(t2)))


def context():
    return dict(catalog=dict(candidates=[dict(method='TRUNCATED_PF',L_D=7,
        q3_status='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD',status='FEASIBLE',conditional_cost=100,
        configuration_id='saved7')]),candidates=[candidate('saved7',7),candidate('missing',0)])


class Selection(unittest.TestCase):
    def test_feasible_unpriced_candidate_recovers_audit_and_remains_selectable(self):
        ctx = context()
        ctx['candidates'][0].update(ready=True, needs=[], prices=None)
        chosen = S.select(ctx, ['TRUNCATED_PF'], excluded={'missing'}, price_rechecks={'saved7'})['selected']
        self.assertEqual((chosen['configuration_id'], chosen['needs']), ('saved7', ['PRICE_26AXIS']))
        self.assertTrue(ctx['candidates'][0]['ready'])
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            folder = run/'pf_auto/candidates/saved7/pf_auto/pf_completion/provider_workspace/docs/research/distributed_selected_pf/saved7'
            folder.mkdir(parents=True)
            (folder/'PF_COMPILE.DISTRIBUTED.json').touch()
            (folder/'PF_COMPILE.RETURN.json').touch()
            marker = folder/'PF_AUDIT.DISTRIBUTED.json'
            marker.touch()
            bridge = Mock()
            S.recover_candidate_price_receipts(run, {'saved7'}, bridge)
            bridge.import_received_stage.assert_called_once()
            self.assertEqual(bridge.import_received_stage.call_args.args[1:], (marker, 'PF_AUDIT'))

    def test_legacy_price_summary_only_restores_matching_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            wave = Path(tmp)
            selected = candidate('missing', 7)
            folder = wave/'candidates/missing/pf_auto'
            leaf = folder/'pf_completion'
            leaf.mkdir(parents=True)
            held = dict(status='PRIOR_OWNER_HOLD', configuration_id='missing', reason='original receipt')
            for path, obj in (
                (folder/'SELECTION.json', dict(selected=selected)),
                (leaf/'RESPONSE.json', dict(items=[held])),
                (leaf/'ORIGINAL_OWNER_CHECK.json', dict(rows=[held])),
                (leaf/'ORIGINAL_OWNER_REQUEST.json', dict(selected=[selected], output_root=tmp)),
            ):
                path.write_text(json.dumps(obj))
            record = dict(status='PREPARATION_INCOMPLETE', reason='PF_EVIDENCE_INCOMPLETE',
                          key=selected['key'], configurations={})
            rows = {'missing': dict(record, key={'p': 8})}
            S.restore_price_checks([wave], rows)
            self.assertEqual(rows['missing']['status'], 'PREPARATION_INCOMPLETE')
            rows['missing'] = record.copy()
            S.restore_price_checks([wave], rows)
            self.assertEqual(rows['missing']['price_reconciliation']['reason'], held['reason'])
            self.assertEqual(rows['missing']['price_prior_owner_directories'], [str(Path(tmp)/'missing')])

    def test_large_shift_is_priority_only_and_p_is_not_supplied(self):
        d=S.select(context(),['TRUNCATED_PF'])
        self.assertEqual(d['selected']['configuration_id'],'missing')
        self.assertEqual(d['evaluated_prefixes'],[7])
        self.assertEqual({c['key']['p']for c in d['candidates']},{0,7})
        self.assertEqual(d['priority_scope'],'REQUEST_ORDER_ONLY_NOT_Q3_REJECTION_OR_PRICE_BOUND')

    def test_default_plan_and_resume_pass_selected_body_to_existing_helper(self):
        ctx=context();spec=dict(methods=['TRUNCATED_PF'],science_approved=False,
                              pf_completion=dict(mode='plan',entry='fixed_D_entry'))
        with tempfile.TemporaryDirectory()as d:
            with patch.object(S.PF,'complete_selected',return_value=dict(response=dict(items=[]),registry_file='registry'))as helper:
                _,_,first=S.prepare(Path(d),{},ctx,spec,Path(d)/'run',{})
                _,_,second=S.prepare(Path(d),{},ctx,spec,Path(d)/'run',{})
                sent=helper.call_args.args[1]['pf_completion']['selected']
                self.assertEqual(sent,[dict(configuration_body=ctx['candidates'][1]['configuration_body'])])
            self.assertEqual(first,second)
            self.assertEqual(first['new_scientific_calls'],0)
            changed=deepcopy(ctx);changed['candidates'][1]['key']['p']=1
            with self.assertRaisesRegex(ValueError,'identity changed'):
                S.prepare(Path(d),{},changed,spec,Path(d)/'run',{})

    def test_prepared_identity_and_artificial_route_cannot_be_substituted(self):
        ctx=context();spec=dict(methods=['TRUNCATED_PF'],science_approved=True,
            pf_completion=dict(mode='artificial',entry='fixture'))
        with tempfile.TemporaryDirectory()as d:
            with self.assertRaisesRegex(ValueError,'only artificial'):
                S.prepare(Path(d),{},ctx,spec,Path(d)/'mixed',{'TRUNCATED_PF':[dict(kind='pf_b_pf6')]})
            after=deepcopy(ctx);after['candidates'][1]['key']['formula']='4th';after['pf_evidence_registry']=dict(configurations={})
            with patch.object(S.PF,'complete_selected',return_value=dict(response=dict(items=[]),registry_file='registry')),patch.object(S,'_refresh',return_value=after):
                with self.assertRaisesRegex(ValueError,'another p/formula/width'):
                    S.prepare(Path(d),{},ctx,spec,Path(d)/'wrong',{'TRUNCATED_PF':[dict(kind='artificial')]})

    def test_saved_request_mismatch_holds_only_one_candidate_in_the_wave(self):
        ctx = dict(context(), pf_evidence_registry=dict(configurations={}))
        routes = {'TRUNCATED_PF': [dict(kind='artificial')]}
        spec = dict(methods=list(routes), science_approved=True,
                    pf_completion=dict(mode='complete', entry='fixed_D_entry', max_candidates=2))
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            saved = run/'pf_auto/candidates/missing/pf_auto/pf_completion/REQUEST.json'
            saved.parent.mkdir(parents=True)
            saved.write_text(json.dumps(dict(output_root='/previous-run', registry_file='/previous-registry')))
            before = saved.read_bytes()
            delivered = []

            def prepare(_root, _spec, _run, request, _entry, output, _runtime):
                delivered.extend(row['configuration_body']['id'] for row in request['selected'])
                output.write_text('{"items": []}')

            with patch('trottertracks.pf_distributed_preparation.complete_selected', side_effect=prepare), \
                 patch.object(S, '_refresh', side_effect=lambda root, problem, context, *a: context):
                _, _, wave = S.prepare_next(run, {}, ctx, spec, run, routes, object())
            self.assertEqual(delivered, ['saved7'])
            self.assertEqual(saved.read_bytes(), before)
            self.assertEqual(wave['records']['missing']['reason'], 'PF_COMPLETION_REQUEST_MISMATCH')
            self.assertIn('registry_file', wave['records']['missing']['differing_fields'])
            self.assertEqual(wave['status'], 'PREPARATION_WAVE_PROCESSED')


if __name__=='__main__':unittest.main()

def test_shared_pf_wave_ranking_is_stored_once_and_roundtrips(tmp_path):
    rows=[dict(configuration_id=str(i),evidence='data'*1000) for i in range(30)]
    selection=dict(candidate_count=len(rows),candidates=rows,selected=rows[0])
    shared=S.shared_selection_snapshot(tmp_path,selection)
    for row in rows:
        candidate=tmp_path/'pf_auto/candidates'/row['configuration_id']
        path=candidate/'pf_auto/SELECTION.json'
        S.write(path,S.selection_for_candidate(selection,row,shared,candidate))
        value=json.loads(path.read_text())
        assert 'candidates' not in value and value['selected']==row
        assert S.read_full_selection(path)['candidates']==rows
    inventory=list((tmp_path/'pf_auto/selection-inventory').glob('*.json'))
    assert len(inventory)==1
    assert S.shared_selection_snapshot(tmp_path,selection)==shared
