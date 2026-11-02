"""Artificial receipt/manifest mutations only. No H2 evaluation or QPY building."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h2_saved_cost_static_join_b2 as j


def fixture():
    guard=dict(policy=j.previous.GUARD,unchanged_gate_stream=True,state_fit=False,
        raw_validation_threshold_changed=False,passes=[dict(policy=j.previous.GUARD,
        unchanged_gate_stream=True,input_or_output_state_fit=False,postcompile_phase_correction=False)])
    check=dict(check='compiled_vector_cosine',PASS=True,phase_alignment=False,strictly_below=1e-7)
    candidate=dict(system='H2',method='FULL',prefix=3,q=4,formula='2nd')
    archives=[]
    def archive(name,char):
        rec=dict(path='cosine_'+name+'.qpy.gz',archive_sha256=char*64,compressed_bytes=10,
                 original_sha256='f'*64,original_bytes=20)
        archives.append(dict(candidate='H2_FULL_3_4_2nd',locator='${PF}/'+rec['path'],
            archive=dict(sha256=rec['archive_sha256'],bytes=10),raw=dict(sha256='f'*64,bytes=20),
            QPY_static=dict(qpy_version=13,nq=5,nc=1,circuit_object_created=False,scientific_libraries_imported=False)))
        return rec
    saved=dict(saved_candidate='H2_FULL_3_4_2nd',candidate_fields=candidate,axis='cosine',
        compiler_settings=deepcopy(j.previous.SETTINGS),compiler_hash=j.previous.COMPILER,
        runtime_id=j.previous.PY312,cohort='COMMON_PF_DETERMINISTIC_20261022',RZ=1655,unit=j.UNIT,
        preparation_hash=j.PREP,H='1'*64,state='2'*64,scalar=0.,guard=guard,correctness=check,
        wrapper=archive('wrapper','3'),compiled=archive('compiled','4'))
    recovered=dict(compiler_hash=j.previous.COMPILER,saved_RZ_per_axis=1655,
        original_plan_sha256=j.PLAN,SOURCE=j.previous.PF_SOURCE,preparation_sha256=j.PREP,
        H='1'*64,state='2'*64,phase_guard_receipt_SHA256=j.r.digest(guard),phase_guard_QPY_metadata_identical=True,
        correctness=dict(receipt=deepcopy(check),scope='SINGLE_SAVED_INPUT_STATE',all_state_or_operator_proof=False),
        wrapper_axis_wire_check=True)
    packet=dict(candidate=deepcopy(candidate),prices=dict(cosine=dict(compiler=j.previous.COMPILER,
        metrics=dict(rz_count=1655),phase_guard=deepcopy(guard))),SOURCE=j.previous.PF_SOURCE,
        preparation_hash=j.PREP,hamiltonian='1'*64,state='2'*64,scalar=0.,checks=[deepcopy(check)])
    plan=dict(rows=[deepcopy(candidate)],private_original_plan_sha256=j.PLAN,
        compiler=j.previous.COMPILER,binding=dict(hamiltonian='1'*64,state='2'*64))
    return saved,deepcopy(saved),recovered,packet,archives,plan


def closure_fixture():
    p=[dict(role='SOURCE:artificial_'+str(i),sha256='1'*64,bytes=i) for i in range(365)]
    c=[dict(index=i,logical_path='${SCIENCE_SOURCE}/artificial_'+str(i),historical_actual_load_proven=False,**e) for i,e in enumerate(p)]
    return c,p


class StaticJoinTests(unittest.TestCase):
    def test_inherited_observer_never_becomes_GPU02_direct(self):
        result=j.join_full(*fixture())
        self.assertEqual(result['original_byte_observer'],'GPUA100-gleap-01')
        self.assertFalse(result['original_bytes_rehashed_GPU02'])
        self.assertEqual(result['detailed_gate_parameter_phase_stream_binding'],'UNRESOLVED')

    def test_no_execution_even_all_conditions_verified(self):
        p=j.policy_preflight([dict(condition='fixture',status='VERIFIED')])
        for k in ('policy2a_adopted','alternative_adopted','science_authorized','execution_ready','strict_contract_changed'):
            self.assertIs(p[k],False)
        self.assertEqual(p['COMPARE'],'BLOCKED');self.assertIsNone(p['grant'])
        self.assertEqual(set(p['actual_calls'].values()),{0})

    def test_mismatch_not_ignored(self):
        p=j.policy_preflight([dict(condition='provider',status='MISMATCH')])
        self.assertEqual(p['conditions'][0]['status'],'MISMATCH')
        self.assertFalse(p['evidence_complete_for_unconditional_common_native_reuse'])

    def test_unknown_status_denied(self):
        with self.assertRaises(j.r.EvidenceError):j.policy_preflight([dict(condition='provider',status='SAFE_BY_NAME')])

    def test_duplicate_conditions_denied(self):
        with self.assertRaises(j.r.EvidenceError):j.policy_preflight([dict(condition='x',status='VERIFIED')]*2)

    def test_scientific_execute_denied(self):
        with self.assertRaises(j.r.EvidenceError):j.forbid_execution('H2_REAL')

    def test_closure_projection_does_not_certify_historical_load(self):
        result=j.join_runtime_projection(*closure_fixture())
        self.assertFalse(result['historical_actual_load_complete'])
        self.assertFalse(result['actual_absolute_path_map_recovered_GPU02'])

    def test_identical_declared_duplicate_paths_allowed(self):
        c,p=closure_fixture();p[1]=deepcopy(p[0]);c[1]=dict(c[0],index=1)
        self.assertEqual(j.join_runtime_projection(c,p)['closure_entries'],365)

    def test_integrated_axis_identity_index(self):
        rows=[];keys=set()
        for i in range(67):
            for a in ('cosine','sine'):
                keys.add((str(i),a))
                rows.append(dict(series='PR',candidate=j.previous.PR_CANDIDATES[i>=33],task=str(i),axis=a,
                    cohort='EXPLORATION',saved_object_ref=dict(commit=j.B,path=j.B_DIR+'saved_PR_cost_objects_134.json')))
        for c in j.FULL:
            for a in ('cosine','sine'):
                rows.append(dict(series='FULL',candidate=c,task=None,axis=a,
                    cohort='COMMON_PF_DETERMINISTIC_20261022',
                    saved_object_ref=dict(commit=j.B,path=j.B_DIR+'saved_common_PF_cost_objects_4.json')))
        self.assertEqual(j.validate_identity_index(rows,keys),dict(PR=134,Full=4))
        for field,value in [('candidate','OLD_FULL'),('cohort','CONFIRMATION'),('axis','other')]:
            bad=deepcopy(rows);bad[-1][field]=value
            with self.assertRaises(j.r.EvidenceError):j.validate_identity_index(bad,keys)
        bad=deepcopy(rows);bad[0]['saved_object_ref']['commit']='0'*40
        with self.assertRaises(j.r.EvidenceError):j.validate_identity_index(bad,keys)
        with self.assertRaises(j.r.EvidenceError):j.validate_identity_index(rows[:-1],keys)
        bad=deepcopy(rows);bad[-1]=deepcopy(bad[-2])
        with self.assertRaises(j.r.EvidenceError):j.validate_identity_index(bad,keys)

    def test_no_science_import(self):
        self.assertFalse(any(n.split('.')[0] in ('numpy','qiskit','scipy','symengine','trotterlib','trottertracks') for n in sys.modules))


def full_mutation(label,fn):
    def test(self):
        values=fixture();fn(values)
        with self.assertRaises((j.r.EvidenceError,KeyError,StopIteration)):j.join_full(*values)
    setattr(StaticJoinTests,'test_full_reject_'+label,test)


mutations={
 'other_candidate':lambda v:v[0].update(saved_candidate='H2_OLD'),
 'old_RZ':lambda v:v[0].update(RZ=1367),
 'different_axis':lambda v:v[0].update(axis='sine'),
 'cohort':lambda v:v[0].update(cohort='CONFIRMATION'),
 'runtime':lambda v:v[0].update(runtime_id='3'*64),
 'unit':lambda v:v[0].update(unit='total_T_count'),
 'seed':lambda v:v[0]['compiler_settings'].update(seed_transpiler=18),
 'guard':lambda v:v[3]['prices']['cosine']['phase_guard'].update(state_fit=True),
 'plan_SHA':lambda v:v[5].update(private_original_plan_sha256='0'*64),
 'result_SOURCE':lambda v:v[3].update(SOURCE='0'*40),
 'plan_candidate':lambda v:v[5].update(rows=[]),
 'missing_archive':lambda v:v[4].clear(),
 'duplicate_archive':lambda v:v[4].append(deepcopy(v[4][0])),
 'archive_SHA':lambda v:v[4][0]['archive'].update(sha256='0'*64),
 'raw_SHA':lambda v:v[4][1]['raw'].update(sha256='0'*64),
 'raw_size':lambda v:v[4][1]['raw'].update(bytes=99),
 'wire':lambda v:v[4][0]['QPY_static'].update(nq=6),
 'axis_terminal':lambda v:v[2].update(wrapper_axis_wire_check=False),
 'QPY_guard':lambda v:v[2].update(phase_guard_QPY_metadata_identical=False),
 'preparation':lambda v:v[2].update(preparation_sha256='0'*64),
 'scalar':lambda v:v[3].update(scalar=1.),
 'phase_alignment':lambda v:v[2]['correctness']['receipt'].update(phase_alignment=True),
 'operator_promotion':lambda v:v[2]['correctness'].update(all_state_or_operator_proof=True),
 'correctness_scope':lambda v:v[2]['correctness'].update(scope='ALL_STATE_OPERATOR'),
 'missing_correctness':lambda v:v[3].update(checks=[])}
for label,fn in mutations.items():full_mutation(label,fn)


def closure_mutation(label,fn):
    def test(self):
        c,p=closure_fixture();fn(c)
        with self.assertRaises(j.r.EvidenceError):j.join_runtime_projection(c,p)
    setattr(StaticJoinTests,'test_closure_reject_'+label,test)


for label,fn in {
 'role':lambda c:c[0].update(role='pinned_provider'),
 'SHA':lambda c:c[0].update(sha256='2'*64),
 'size':lambda c:c[0].update(bytes=99),
 'path':lambda c:c[0].update(logical_path='/private/undisclosed'),
 'duplicate_path':lambda c:c[0].update(logical_path=c[1]['logical_path']),
 'index':lambda c:c[0].update(index=1),
 'actual_load_promotion':lambda c:c[0].update(historical_actual_load_proven=True),
 'missing_entry':lambda c:c.pop()}.items():closure_mutation(label,fn)


if __name__=='__main__':unittest.main()
