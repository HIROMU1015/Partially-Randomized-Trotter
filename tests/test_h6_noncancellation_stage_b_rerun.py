"""Artificial full-output regression for the serialization-only correction."""
import ast
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests'),str(ROOT/'src/trotterlib')]
import run_h6_noncancellation_stage_b_rerun as runner
import run_h6_noncancellation_stage_b as stopped
from test_h6_noncancellation_stage_b import execute_toy


def bundle():
    _,rows,lineage,costs,replay,results,ranks,changes,audit=execute_toy()
    prior=deepcopy(results);prior[0]['condition_id']='SAVED_ARTIFICIAL_OTHER'
    combined,summary=runner.core.aggregate(results,prior)
    summary['SOURCE']='ARTIFICIAL'
    return {'joined_rows.json':dict(rows=rows),'patched120_lineage.json':dict(rows=lineage),
        'candidate_totals.json':dict(rows=costs),'selection_comparison30.json':dict(rows=results),
        'candidate_ranks.json':dict(rows=ranks),'rank_relation_changes.json':dict(rows=changes),
        'combined_selection100.json':dict(rows=combined),'summary.json':summary,
        'numeric_validation.json':audit,'used_price_evidence.json':dict(rows=[dict(label='人工,値\n次',nullable=None,flag=False)])}


class FullOutputTests(unittest.TestCase):
    def test_original_duplicate_source_failure_reproduced_artificially(self):
        with self.assertRaises(TypeError):dict(SOURCE='ARTIFICIAL',**{'SOURCE':'ARTIFICIAL'})

    def test_all_ten_JSON_and_five_CSV_written_and_read_independently(self):
        outputs=bundle();before=deepcopy(outputs)
        with tempfile.TemporaryDirectory(prefix='h6-stageB-serialization-test-') as directory:
            with patch.object(runner,'OUT',Path(directory)):
                names=runner.persist_outputs(outputs,'ARTIFICIAL')
                self.assertEqual(len(names),15)
                self.assertEqual(set(names),set(runner.JSON_OUTPUTS)|{n+'.csv' for n in runner.CSV_OUTPUTS})
                for name in runner.JSON_OUTPUTS:
                    saved=json.loads((Path(directory)/name).read_bytes())
                    self.assertEqual(saved,dict(outputs[name],SOURCE='ARTIFICIAL'))
                for stem in runner.CSV_OUTPUTS:
                    records=list(csv.DictReader(io.StringIO((Path(directory)/(stem+'.csv')).read_text())))
                    self.assertEqual(len(records),len(outputs[stem+'.json']['rows']))
                    for a,b in zip(records,outputs[stem+'.json']['rows']):
                        for key,value in b.items():
                            actual=json.loads(a[key]) if isinstance(value,(dict,list,bool,int,float)) or value is None else a[key]
                            self.assertEqual(actual,value)
                self.assertEqual(runner.read(Path(directory)/'summary.json')['SOURCE'],'ARTIFICIAL')
        self.assertEqual(outputs,before)

    def test_mismatching_existing_SOURCE_fails_before_any_output(self):
        outputs=bundle();outputs['summary.json']['SOURCE']='OTHER_SOURCE'
        with tempfile.TemporaryDirectory(prefix='h6-stageB-reject-test-') as directory:
            with patch.object(runner,'OUT',Path(directory)):
                with self.assertRaisesRegex(ValueError,'OUTPUT_SOURCE_MISMATCH:summary.json'):
                    runner.persist_outputs(outputs,'ARTIFICIAL')
                self.assertEqual(list(Path(directory).iterdir()),[])

    def test_null_existing_SOURCE_is_rejected(self):
        outputs=bundle();outputs['summary.json']['SOURCE']=None
        with self.assertRaisesRegex(ValueError,'OUTPUT_SOURCE_MISMATCH'):
            runner.output_payloads(outputs,'ARTIFICIAL')

    def test_complete_bundle_required(self):
        outputs=bundle();del outputs['numeric_validation.json']
        with self.assertRaisesRegex(ValueError,'COMPLETE_OUTPUT_BUNDLE'):
            runner.output_payloads(outputs,'ARTIFICIAL')

    def test_nonfinite_value_cannot_leave_partial_bundle(self):
        outputs=bundle();outputs['summary.json']['invalid']=float('nan')
        with tempfile.TemporaryDirectory(prefix='h6-stageB-nonfinite-test-') as directory:
            with patch.object(runner,'OUT',Path(directory)):
                with self.assertRaises(ValueError):runner.persist_outputs(outputs,'ARTIFICIAL')
                self.assertEqual(list(Path(directory).iterdir()),[])

    def test_empty_rank_changes_CSV_is_readable(self):
        outputs=bundle();outputs['rank_relation_changes.json']['rows']=[]
        payloads=runner.output_payloads(outputs,'ARTIFICIAL')
        self.assertEqual(list(csv.DictReader(io.StringIO(payloads['rank_relation_changes.csv']))),[])

    def test_serialization_does_not_call_scientific_functions(self):
        outputs=bundle()
        with patch.object(runner.core,'baseline',side_effect=AssertionError('science forbidden')):
            with patch.object(runner.core,'compare',side_effect=AssertionError('science forbidden')):
                runner.output_payloads(outputs,'ARTIFICIAL')

    def test_saved_completion_and_native_audit_output_forms(self):
        with tempfile.TemporaryDirectory(prefix='h6-stageB-receipt-test-') as directory:
            with patch.object(runner,'OUT',Path(directory)):
                names=runner.persist_outputs(bundle(),'ARTIFICIAL')
                hashes={n:hashlib.sha256((Path(directory)/n).read_bytes()).hexdigest() for n in names}
                runner.write('numeric_completion.json',dict(SOURCE='ARTIFICIAL',result_sha256=hashes))
                runner.write('grant_consumption_receipt.json',dict(SOURCE='ARTIFICIAL',status='CONSUMED_COMPLETE'))
                runner.write('one_shot.marker','ARTIFICIAL\n',exclusive=True)
                with self.assertRaises(FileExistsError):runner.write('one_shot.marker','ARTIFICIAL\n',exclusive=True)
                for name in ('native_exit_evidence.json','completion_audit.json','STOP.json'):
                    runner.write(name,dict(SOURCE='ARTIFICIAL',native_exit_code=0,reaped=True,artificial=True))
                    self.assertEqual(runner.read(Path(directory)/name)['SOURCE'],'ARTIFICIAL')
                for name,expected in runner.read(Path(directory)/'numeric_completion.json')['result_sha256'].items():
                    self.assertEqual(runner.sha(Path(directory)/name),expected)

    def test_scientific_input_list_metadata_and_core_calls_unchanged(self):
        self.assertEqual(runner.INPUTS,stopped.INPUTS)
        before=ast.parse((ROOT/'scripts/run_h6_noncancellation_stage_b.py').read_text())
        after=ast.parse((ROOT/'scripts/run_h6_noncancellation_stage_b_rerun.py').read_text())
        def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        self.assertEqual(ast.dump(function(before,'metadata')),ast.dump(function(after,'metadata')))
        def calls(tree):
            return [ast.dump(n) for n in ast.walk(function(tree,'run')) if isinstance(n,ast.Call)
                    and (ast.unparse(n.func).startswith('core.') or isinstance(n.func,ast.Name) and n.func.id=='stage')]
        self.assertEqual(calls(before),calls(after))
        old_contract=json.loads((ROOT/stopped.B/'track_a_h6_noncancellation_stage_b/2026-10-20/sealed_contract.json').read_bytes())
        for name,expected in old_contract['source_hashes'].items():
            self.assertEqual(runner.sha(ROOT/name),expected)
        self.assertNotEqual(runner.OUT,stopped.OUT)


if __name__=='__main__':unittest.main()
