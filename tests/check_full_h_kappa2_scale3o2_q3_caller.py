"""Artificial observed-worker equivalence and fixed metadata, no real H4 Q3."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tests'))
from check_pf_shortlist_q3_caller import artificial,MOCK
spec=importlib.util.spec_from_file_location('_scale3o2_q3',ROOT/'scripts/run_full_h_kappa2_scale3o2_q3_fixed.py')
M=importlib.util.module_from_spec(spec);sys.modules[spec.name]=M;spec.loader.exec_module(M)


class CallerTests(unittest.TestCase):
    def test_fixed_vector_input_and_generic_count_bounds_without_risk(self):
        with patch.object(M.M.T,'evaluate',side_effect=AssertionError('No H4 risk'))as call:
            x=M.build_inputs()
        self.assertEqual(call.call_count,0)
        self.assertEqual(len(x['cases']),1);c=x['cases'][0]
        self.assertEqual([r['N_X']for r in c['allocation']['shots']],M.VECTOR)
        self.assertEqual([r['N_Y']for r in c['allocation']['shots']],M.VECTOR)
        self.assertEqual(M.VECTOR,[(3*r['N_X']+1)//2 for r in x['prior_allocations']['control']['shots']])
        self.assertEqual(x['finite_scope_preflight']['transition_pairs_upper'],353666241)
        self.assertNotIn('first_round_full_support_states',x['finite_scope_preflight'])
        self.assertEqual(x['options']['phase_radii'],[1.]*13)
        self.assertEqual(x['maximum_scientific_calls'],1)
        self.assertEqual(c['total_configuration_id'],'265b90dc4a9b81011736664205e23833c420d569e57c923ef45c39eb59489612')

    def test_actual_worker_route_observation_preserves_artificial_API(self):
        c,x,binding,view=artificial();before=deepcopy((c,x,view))
        with tempfile.TemporaryDirectory()as folder,patch.object(M.M,'metadata',return_value=(binding,view)),patch.object(M.M,'check_source'):
            root=Path(folder);plain=root/'plain';watched=root/'watched';plain.mkdir();watched.mkdir()
            M.M.worker(c,x,MOCK,plain)
            with M.observation(watched):M.M.worker(c,x,MOCK,watched)
            self.assertEqual(json.loads((plain/'API_RETURN.json').read_text()),json.loads((watched/'API_RETURN.json').read_text()))
            events=[json.loads(line)for line in(watched/'SEGMENTS.jsonl').read_text().splitlines()]
            self.assertEqual(events[0]['stage'],'DP_CALL_START');self.assertEqual(events[-1]['stage'],'DP_CALL_RETURN')
            self.assertTrue(any(e['stage']=='TERMINAL_POST_PROPAGATION_COMPLETE'for e in events))
            self.assertTrue((watched/'RESOURCE_FINAL.json').exists())
        self.assertEqual((c,x,view),before)

    def test_process_exit_recording_and_context_restoration(self):
        original=M.R.P.ProcessPoolExecutor
        with tempfile.TemporaryDirectory()as folder:
            with M.observation(folder):
                with M.ObservedExecutor(max_workers=2)as pool:self.assertEqual(list(pool.map(abs,[-1,-2])),[1,2])
            lines=[json.loads(line)for p in Path(folder).glob('PROCESS_EXITS_*.jsonl')for line in p.read_text().splitlines()]
            self.assertTrue(lines);self.assertTrue(all(c['exitcode']==0 and c['signal']is None for line in lines for c in line['children']))
        self.assertIs(M.R.P.ProcessPoolExecutor,original);self.assertIsNone(M.EXIT_DIRECTORY)

    def test_actual_supervisor_worker_observation_and_saved_receipt(self):
        c,x,binding,view=artificial();x['concurrent_configurations']=1
        with tempfile.TemporaryDirectory()as folder,patch.object(M.M,'metadata',return_value=(binding,view)),patch.object(M.M,'check_source'):
            root=Path(folder);(root/c['label']).mkdir()
            with M.observation(root):result=M.M.supervise([c],x,MOCK,root,poll_seconds=.01)
            self.assertEqual(result['errors'],[]);self.assertEqual(result['monitor_errors'],[])
            self.assertTrue(result['receipts'][0]['api_return_saved'])
            self.assertEqual(result['receipts'][0]['scientific_evaluation_calls'],1)
            events=[json.loads(line)for p in root.glob('PROCESS_EXITS_*.jsonl')for line in p.read_text().splitlines()]
            self.assertTrue(events);self.assertTrue(all(c['exitcode']==0 for e in events for c in e['children']))
            self.assertTrue((root/c['label']/'SEGMENTS.jsonl').exists())


if __name__=='__main__':unittest.main(verbosity=2)
