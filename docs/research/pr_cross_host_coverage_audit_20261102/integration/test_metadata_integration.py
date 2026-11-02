"""Artificial identity/CSV tests; no numerical scientific workload."""
import unittest,copy,io,csv,json
from metadata_integration import normalize_selector,classify_attempt,same_result_key,condition_signature,PRIVATE
from integrate_catalog import group_records,pointer_value

class IdentityTests(unittest.TestCase):
    def rec(self,rid,blob='a'*40,selector='CSV_DATA_ROW=1',path='artifacts/fixture/result.csv'):
        return dict(environment='LOCAL',record_id=rid,result=dict(blob=blob,path=path,selector=selector,state='VERIFIED_GIT_BLOB'),condition={'hamiltonian.rank':'12','method.q':'1'},source_commit='UNKNOWN',run_id='UNKNOWN')
    def test_csv_selector_aliases(self):
        self.assertEqual({normalize_selector(s) for s in ['CSV_DATA_ROW=7','CSV:data_row=7','CSV data row 7']},{'CSV_DATA_ROW=7'})
    def test_same_source_result_is_copy(self):
        g,c=group_records([self.rec('one'),self.rec('two')]);self.assertEqual(len(g),1);self.assertEqual(g[0][2],'VERIFIED_SAVED_RESULT_IDENTITY')
    def test_equal_values_do_not_merge_different_results(self):
        self.assertEqual(len(group_records([self.rec('one'),self.rec('two',blob='b'*40)])[0]),2)
    def test_different_rows_and_rounds_stay_separate(self):
        self.assertEqual(len(group_records([self.rec('one'),self.rec('two',selector='CSV_DATA_ROW=2')])[0]),2)
    def test_different_paths_require_more_than_equal_blob(self):
        self.assertEqual(len(group_records([self.rec('one'),self.rec('two',path='artifacts/other/result.csv')])[0]),2)
    def test_scientific_condition_conflict_is_not_merged(self):
        a,b=self.rec('one'),self.rec('two');b['condition']['hamiltonian.rank']='10';g,c=group_records([a,b]);self.assertEqual(len(g),2);self.assertEqual(len(c),1)
    def test_component_source_roles_need_explicit_binding(self):
        a,b=self.rec('one'),self.rec('two');a['source_commit']='c'*40;b['source_commit']='d'*40
        self.assertEqual(len(group_records([a,b])[0]),2)
        roles={a['result']['path']:{a['source_commit']:['provider'],b['source_commit']:['native']}}
        self.assertEqual(len(group_records([a,b],roles)[0]),1)
    def test_different_attempts_not_silently_copied(self):
        a,b=self.rec('one'),self.rec('two');a['run_id']='run01';b['run_id']='run02';self.assertEqual(len(group_records([a,b])[0]),2)
    def test_same_seed_repeat_not_independent(self):
        self.assertEqual(classify_attempt(same_conditions=True,distinct_attempt=True,same_seed=True,disjoint_documented=True,fixed_nominee=True),'RERUN_OF')
    def test_different_seed_alone_not_independent(self):
        self.assertEqual(classify_attempt(same_conditions=True,distinct_attempt=True),'RERUN_OF')
    def test_documented_fixed_disjoint_confirmation(self):
        self.assertEqual(classify_attempt(same_conditions=True,distinct_attempt=True,disjoint_documented=True,fixed_nominee=True),'INDEPENDENT_CONFIRMATION_OF')
    def test_multiline_csv_and_null_tokens(self):
        original=['record','line1\nline2, "quoted"',json.dumps({'scientific_null':'null','missing':'UNKNOWN','zero':'0'})];s=io.StringIO(newline='');csv.writer(s,lineterminator='\r\n').writerow(original);s.seek(0);self.assertEqual(next(csv.reader(s)),original)
    def test_json_pointer_and_empty_root(self):
        d={'a/b':[{'x~y':'null'}]};self.assertEqual(pointer_value(d,'/a~1b/0/x~0y')[0],'null');self.assertEqual(normalize_selector(''),'FILE')
    def test_private_path_guard(self):
        self.assertTrue(PRIVATE.search('/'+'home'+'/example/private'));self.assertFalse(PRIVATE.search('LOCAL-PRIVATE-EVIDENCE-0001'))
if __name__=='__main__':unittest.main()
