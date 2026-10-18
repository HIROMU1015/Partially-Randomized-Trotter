"""Artificial seed/schema tests. No real H4 loading, events, compile, or n design."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import h4_dp_contract as c
import h4_dp_science as science
import h4_p_seed_inventory as seed


def identity():
    return dict(geometry='0.70',H='1'*64,DF='2'*64,state='3'*64,input='4'*64,template='5'*64,
                source='a'*40,compiler='6'*64,environment='7'*64,wrapper_semantics='artificial')


class IntegerInventory(unittest.TestCase):
    def test_fast_ASCII_fingerprint_matches_original_exact_encoder(self):
        sys.path.insert(0,str(ROOT/'src'))
        from trottertracks.resource_applicability.h4_geometry import identity as original
        value=identity()
        self.assertEqual(seed.geometry_master(value,0),original.trajectory_seed(value,0,
            actual_inputs_frozen=True,signal_launch=True))
        for kind in ('order','component'):
            for index in (0,4):
                self.assertEqual(seed.geometry_child(2**64-1,7,3,index,kind),
                                 original.step_seed(2**64-1,7,3,index,kind))

    def test_non_ASCII_or_float_is_not_silently_reencoded(self):
        with self.assertRaisesRegex(ValueError,'INTEGER_ASCII'):
            seed.geometry_fingerprint('x',dict(value=.8))
        with self.assertRaisesRegex(ValueError,'INTEGER_ASCII'):
            seed.geometry_fingerprint('x',dict(value='日本'))

    def test_geometry_superset_covers_all_component_slots_without_sampling(self):
        case=dict(identity=identity(),q=2,r=1,K=2,template_id='artificial')
        values=list(seed.geometry_case_stream(case))
        self.assertEqual(len(values),32*(1+2*(2+2)))
        master=seed.geometry_master(identity(),0)
        self.assertIn(seed.geometry_child(master,1,0,2,'component'),values)
        self.assertIn(seed.geometry_child(master,1,0,0,'order'),values)

    def test_geometry_collision_check_detects_component_child(self):
        case=dict(identity=identity(),q=1,r=1,K=2,template_id='artificial')
        collision=seed.geometry_child(seed.geometry_master(identity(),0),0,0,2,'component')
        checked=seed._geometry_check(('0.70',[case],[collision]))
        self.assertEqual(checked['collisions'],[collision])

    def test_conditional_and_ordinary_hierarchies_are_both_included(self):
        tasks=[dict(seed=17,cell=dict(q=2))]
        self.assertEqual(seed.task_seed_values(tasks),set(c.seed_hierarchy(17,2)))
        values=seed.task_seed_values(tasks,True)
        self.assertIn(int(c.digest(['H6_DBA_ORDER',17])[:16],16),values)
        self.assertIn(int(c.digest(['H6_DBA_COMPONENT',17,1])[:16],16),values)

    def test_preparation_source_never_calls_scientific_operations(self):
        tree=ast.parse((ROOT/'scripts/prepare_h4_p_seal.py').read_text())
        called={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id
                for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))}
        self.assertTrue(called.isdisjoint({'load_context','steps','sample_events','transpile','normal_n_formula','analyze'}))


class FinalBindings(unittest.TestCase):
    def fixture(self,root):
        stats=root/'stats';exclusions=root/'old';receipt=root/'receipt'
        c.save(stats,dict(actual_calls=dict(numeric_design_pass=1),recommended=dict(n_fixed=dict(toy=2),
            proposed_P_calls=dict(trajectory=2,native_evolution=2,wrapper_compile=4,occurrence=2,RTE_event=4,
                                 control_probe_maximum=4,independent_event_oracle_maximum=1)),reference_directions=[
            dict(cell_id='toy',epsilon=.05,expected_direction=1,PR_saved_shots=dict(cosine=1,sine=1),
                 PF_cell_id='PF',PF_G_exact='10')]))
        c.save(exclusions,dict(values=[]))
        c.save(receipt,dict(status='COMPLETE_COLLISION_ZERO',all_registered_scopes_covered=True,
            geometry_distances=6,geometry_registered_candidates=1308,old_explicit_exclusions_sha256=c.sha(exclusions)))
        ep=dict(cell_id='toy',epsilon=.05,expected_direction=1,saved_shots=dict(cosine=1,sine=1),PF_id='PF',PF_G_exact='10')
        plan=dict(SOURCE='a'*40,n_fixed=dict(toy=2),evaluation_endpoints=[ep])
        return dict(stats_result_path=str(stats),stats_result_sha256=c.sha(stats),n_fixed=dict(toy=2),
            evaluation_endpoints=[ep],seed_plan=plan,seed_plan_digest=c.digest(plan),SOURCE='a'*40,
            seed_inventory_receipt=str(receipt),seed_inventory_receipt_sha256=c.sha(receipt),seed_exclusions=str(exclusions),
            calls=dict(trajectory=2,native_evolution=2,wrapper_compile=4,occurrence=2,RTE_event=4,
                       control_probe=4,independent_event_oracle=1))

    def test_fixed_stats_n_cannot_be_reduced(self):
        with tempfile.TemporaryDirectory() as directory:
            spec=self.fixture(Path(directory));c.validate_final_p_bindings(spec)
            spec['n_fixed']['toy']=1
            with self.assertRaisesRegex(ValueError,'FIXED_S_STATS_N'):c.validate_final_p_bindings(spec)

    def test_baseline_direction_cannot_be_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            spec=self.fixture(Path(directory));spec['evaluation_endpoints'][0]['expected_direction']=-1
            with self.assertRaisesRegex(ValueError,'REFERENCE_DIRECTION'):c.validate_final_p_bindings(spec)

    def test_incomplete_seed_inventory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            spec=self.fixture(Path(directory));path=Path(spec['seed_inventory_receipt']);row=c.read(path)
            row['all_registered_scopes_covered']=False;c.save(path,row);spec['seed_inventory_receipt_sha256']=c.sha(path)
            with self.assertRaisesRegex(ValueError,'COMPLETE_OLD_SEED_INVENTORY'):c.validate_final_p_bindings(spec)

    def test_seed_plan_n_cannot_diverge(self):
        with tempfile.TemporaryDirectory() as directory:
            spec=self.fixture(Path(directory));spec['seed_plan']['n_fixed']=dict(toy=3)
            spec['seed_plan_digest']=c.digest(spec['seed_plan'])
            with self.assertRaisesRegex(ValueError,'SEED_PLAN_BINDING'):c.validate_final_p_bindings(spec)

    def test_scientific_budget_cannot_be_reduced(self):
        with tempfile.TemporaryDirectory() as directory:
            spec=self.fixture(Path(directory));spec['calls']['wrapper_compile']=2
            with self.assertRaisesRegex(ValueError,'CALL_BUDGET'):c.validate_final_p_bindings(spec)


class ConfirmationSchema(unittest.TestCase):
    def test_master_step_seed_and_paired_cohort_recorded_with_mock_kernel(self):
        master=91;hierarchy=c.seed_hierarchy(master,2)
        requests=[SimpleNamespace(seed=x) for x in hierarchy[1:]]
        frozen=SimpleNamespace(steps=lambda *a:(requests,SimpleNamespace(exact_finite_distribution=1.,
            to_dict=lambda:dict(purpose='ARTIFICIAL'))),
            event_values=lambda r:[ [{'taylor_order':0}] for _ in r ],control=lambda *a:None)
        context=SimpleNamespace(preparations={2:SimpleNamespace(preparation_hash='p')},frozen=frozen,
            native=SimpleNamespace(partial_native_from_step_requests=lambda *a,**kw:object()),binding=dict(compiler='c'*64))
        task=dict(cell=dict(id='toy',prefix=2,q=2,r=4,R=8),seed=master,replica=0,saved_log_B=0.,
                  log_B_tolerance=1e-10,preparation_hash='p')
        def compile_fake(ctx,evolution,writer,budget,seed=None,event_digest=None):
            budget.take('wrapper_compile',2)
            return {axis:dict(RZ=10,seed=seed,event_digest=event_digest,compiler_hash='c'*64,fingerprint='f'*64)
                    for axis in c.AXES}
        with tempfile.TemporaryDirectory() as directory,patch.object(science,'compile_axes',side_effect=compile_fake):
            out=science.confirmation_task(context,task,directory)
            trace=c.read(Path(directory)/'trajectory.json')
            self.assertEqual(trace['master_seed'],master);self.assertEqual(trace['step_seeds'],hierarchy[1:])
            self.assertEqual(trace['cohort'],'INDEPENDENT_H4_CONFIRMATION')
            self.assertEqual(out['axes']['cosine']['event_digest'],out['axes']['sine']['event_digest'])
            self.assertEqual(out['calls']['wrapper_compile'],2)


if __name__=='__main__':unittest.main()
