"""Synthetic receipts and mock authority only. No saved H2 u/N/G evaluation."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import importlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h2_archived_exploratory as c


def costs_fixture():
    pr=[];full=[]
    for candidate,n in zip(c.policy.CANDIDATES[2:],(33,34)):
        for i in range(n):
            for a in ('cosine','sine'):
                pr.append(dict(candidate=candidate.id,task='ARTIFICIAL_'+candidate.id+'_'+str(i),axis=a,
                    RZ=10+i,cohort='EXPLORATION',runtime_id='e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f',
                    unit='compiled_RZ_per_full_Hadamard_wrapper_axis'))
    for cid,(saved,rz) in c.FULL.items():
        for a in ('cosine','sine'):
            full.append(dict(requested_candidate=cid,saved_candidate=saved,axis=a,RZ=rz,
                cohort='COMMON_PF_DETERMINISTIC_20261022',runtime_id=pr[0]['runtime_id'],
                unit='compiled_RZ_per_full_Hadamard_wrapper_axis'))
    return pr,full


def adaptive(candidate):
    obs=lambda value,unit,scope:c.policy.Observation(value,unit,scope,'ARTIFICIAL_RECEIPT')
    return c.policy.AdaptiveInputs(candidate.id,'ARTIFICIAL_ONLY','ARTIFICIAL_REFERENCE',
        obs(1e-16,'CORRECTED_SIGNAL','H2_REFERENCE_COMBINED_DIAGNOSTIC'),
        (obs(0.,'CORRECTED_SIGNAL','PRIMARY_VS_INDEPENDENT_SIGNAL'),),
        (obs(0.,'CORRECTED_STATE_VECTOR','PRIMARY_VS_INDEPENDENT_VECTOR'),),
        obs(0.,'CORRECTED_SIGNAL','B_RAW_MINUS_CORRECTED_SIGNAL'),
        obs(0.,'LOG_NORMALIZATION','TWO_PATH_LOG_B_DIFFERENCE'),
        dict(deterministic=2,tail=1),dict(deterministic=2,tail=1),
        obs(1.,'CORRECTED_STATE_VECTOR','BOTH_PATHS_ALL_REQUIRED_STAGES'),
        'BOTH_PATHS_ALL_REQUIRED_STAGES','ARTIFICIAL_TRACE',c.policy.NATIVE_POLICY,'f'*64)


class NumericalFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cap=tempfile.TemporaryDirectory(prefix='h2-exploratory-artificial-provider-')
        cls.root=Path(cls.cap.name)
        for path,(_,raw) in zip(c.PROVIDER_PATHS,c.seal.git_batch(c.PROVIDER,c.PROVIDER_PATHS)):
            p=cls.root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        sys.path.insert(0,str(cls.root/'src'))

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.root/'src'))
        for name in list(sys.modules):
            if name=='trottertracks' or name.startswith('trottertracks.'):del sys.modules[name]
        cls.cap.cleanup()

    def run_fixture(self,ineligible=False):
        from h2_archived_exploratory_backend import Backend
        with tempfile.TemporaryDirectory(prefix='h2-exploratory-artificial-run-') as name:
            root=Path(name);s=c.Session.__new__(c.Session)
            s.cfg=dict(budget=c.counts(),domain='ARTIFICIAL_ONLY');s.root=root
            s.used=dict.fromkeys(c.KINDS,0);s.sequence=0;s.active=None
            for folder in ('ledger','results','delegated_receipts'):(root/folder).mkdir()
            pr,full=costs_fixture();costs=c.validate_costs(pr,full)
            diagnostics={x.id:dict(inputs=adaptive(x),primary=dict(log_B=0.),
                bias=dict(real=.001 if ineligible else .00001,imag=.00002)) for x in c.policy.CANDIDATES}
            data=dict(diagnostics=diagnostics,costs=costs,reference_id='ARTIFICIAL_REFERENCE')
            s.take('bootstrap');backend=Backend(s)
            try:
                with patch.object(c,'saved_inputs',return_value=data):result=c.pipeline(s,backend)
            finally:backend.close()
            with patch.object(c.old.Session,'verify'),patch.object(c.old.Session,'loaded_closure'):
                c.audit_end(s)
            self.assertEqual(s.used,c.counts())
            self.assertEqual(len(list((root/'ledger').glob('*.json'))),sum(c.counts().values()))
            return result

    def test_reused_numeric_pipeline_exact_internal_calls(self):
        result=self.run_fixture()
        self.assertEqual(len(result['methods']),4)
        self.assertFalse(result['science_GO'])
        for r in result['methods']:
            self.assertTrue(r['eligible']);self.assertEqual(sum(r['G_axis'].values()),r['G_total'])
            self.assertFalse(r['population_mean_certified']);self.assertIsNone(r['nominal_interval'])
            self.assertEqual(len(r['saved_cost_sample_tasks']),33 if r['candidate']==c.policy.CANDIDATES[2].id else 34 if r['candidate']==c.policy.CANDIDATES[3].id else 1)

    def test_ineligible_C_only_not_zero_or_forced_ranking(self):
        result=self.run_fixture(ineligible=True)
        self.assertEqual(result['delegated_comparison']['status'],'INCOMPLETE_OR_INELIGIBLE_NO_C_DECISION')
        for r in result['methods']:
            self.assertIsNone(r['G_total']);self.assertIsNone(r['G_axis'])
            self.assertEqual(r['status'],'C_ONLY_N_G_NOT_ELIGIBLE')
            self.assertTrue(r['compiled_RZ_statistics'])

    def test_pairing_is_retained(self):
        module=importlib.import_module('trottertracks.resource_applicability.h6_matched_accounting_v1')
        signal=dict(cell=dict(id='ARTIFICIAL',method='PR'),precision_rows={str(.0005):{'1.0':dict(
            cost_acquisition_eligible=True,N_total=4,axes={'real':{'shots':2},'imag':{'shots':2}})}})
        rows=[dict(replica=i,axis=a,control='symmetric_directional',metrics={'RZ':v})
              for i,pair in enumerate(((1,5),(2,4),(3,3))) for a,v in zip(('cosine','sine'),pair)]
        result=module.resource_row(signal,rows,.0005)
        self.assertEqual(result['G_standard_error'],0.)
        self.assertGreater(result['axes']['cosine']['standard_error'],0.)


class ControlTests(unittest.TestCase):
    def test_fixed_cost_pair_shape(self):
        out=c.validate_costs(*costs_fixture());self.assertEqual([len(out[x.id]) for x in c.policy.CANDIDATES],[1,1,33,34])

    def test_no_quantum_or_diagnostic_calls(self):
        caps=c.counts()
        for name in ('setup','prepare','sampling','circuit_build','lowering','compile','transpile','simulate','signal','reference_recompute','primitive_evaluation'):
            self.assertEqual(caps[name],0)

    def test_budget_overrun_and_forbidden_kind(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'ledger').mkdir()
            s=c.Session.__new__(c.Session);s.root=root;s.cfg={'budget':c.counts()};s.used=dict.fromkeys(c.KINDS,0);s.sequence=0
            s.take('bootstrap')
            with self.assertRaises(c.Stop):s.take('bootstrap')
            with self.assertRaises(c.Stop):s.take('compile')
            with self.assertRaises(c.Stop):s.take('extra_science')
            self.assertEqual(s.used['compile'],0)

    def approval(self):return dict(issuer='USER_EXPLICIT',execution_authorized=True,contract_sha256=c.digest(self.cfg),accepted_limitations=list(c.LIMITS))

    def grant(self,a):return dict(id=c.SERIES+':ARTIFICIAL_123456',purpose=c.SERIES,state='UNUSED',contract_sha256=c.digest(self.cfg),approval_sha256=c.digest(a),run_root=self.cfg['run_root'],budget=c.counts(),retry=False)

    def setUp(self):self.cfg=dict(run_root='ARTIFICIAL',origin_run_root='ARTIFICIAL_ORIGIN')

    def test_unapproved_execution_denied_before_preflight(self):
        with patch.object(c,'preflight') as check:
            with self.assertRaises(c.Stop):c.authorize(self.cfg,dict(issuer='USER_EXPLICIT',execution_authorized=False),{})
            check.assert_not_called()

    def test_valid_synthetic_authority_shape(self):
        a=self.approval();g=self.grant(a)
        with patch.object(c,'preflight'),patch.object(c,'read',return_value=dict(grant={'id':'OLD_DIAGNOSTIC'})):
            c.authorize(self.cfg,a,g)

    def test_reject_old_or_consumed_grant(self):
        a=self.approval()
        with patch.object(c,'preflight'),patch.object(c,'read',return_value=dict(grant={'id':'OLD_DIAGNOSTIC'})):
            for field,value in [('id','OLD_DIAGNOSTIC'),('state','CONSUMED'),('retry',True),('purpose','STRICT_COMPARE'),('contract_sha256','0'*64)]:
                g=self.grant(a);g[field]=value
                with self.assertRaises(c.Stop):c.authorize(self.cfg,a,g)

    def test_limitations_cannot_be_dropped(self):
        a=self.approval();a['accepted_limitations']=[]
        with self.assertRaises(c.Stop):c.authorize(self.cfg,a,{})

    def test_new_root_collision_denied(self):
        a=self.approval();g=self.grant(a)
        with patch.object(c,'authorize'),tempfile.TemporaryDirectory() as name:
            self.cfg['run_root']=name
            with self.assertRaises(FileExistsError):c.Session(self.cfg,a,g)

    def test_end_audit_rejects_disk_ledger_drift(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'ledger').mkdir();s=c.Session.__new__(c.Session)
            s.root=root;s.cfg={'budget':dict.fromkeys(c.KINDS,0)};s.cfg['budget']['bootstrap']=1
            s.used=deepcopy(s.cfg['budget']);s.sequence=1
            path=root/'ledger/000001.json'
            row=dict(kind='bootstrap',amount=1,nonrefundable=True,contract_sha256=c.digest(s.cfg))
            path.write_text(json.dumps(row))
            with patch.object(c.old.Session,'verify'),patch.object(c.old.Session,'loaded_closure'):
                c.audit_end(s)
                row['kind']='accounting';path.write_text(json.dumps(row))
                with self.assertRaises(c.Stop):c.audit_end(s)

    def test_no_scientific_package_imports(self):
        self.assertFalse(any(k.split('.')[0] in ('numpy','qiskit','scipy','symengine','trotterlib') for k in sys.modules))


class SealFixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='h2-exploratory-artificial-seal-')
        self.addCleanup(self.tmp.cleanup);self.assets=Path(self.tmp.name)
        self.addCleanup(patch.stopall);patch.object(c,'ASSETS',self.assets).start()
        patch.object(c,'saved_inputs',return_value={'ARTIFICIAL_ONLY':True}).start()
        patch.object(c.seal,'git_batch',side_effect=lambda commit,paths:[('ARTIFICIAL_BLOB',b'# ARTIFICIAL SOURCE\n') for p in paths]).start()
        files=[];sources=[]
        for role,paths in [('runner',c.NEW+c.REUSED),('provider',c.PROVIDER_PATHS)]:
            for rel in paths:
                p=self.assets/role/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'# ARTIFICIAL SOURCE\n')
                files.append(dict(path=str(p),sha256=c.sha(p),role='SOURCE:'+role))
                sources.append(dict(root=role,path=rel,local_path=str(p),sha256=c.sha(p)))
        manifest=self.assets/'input_manifest_PRIVATE.json';manifest.write_text('{}')
        files.append(dict(path=str(manifest),sha256=c.sha(manifest),role='INPUT:ARTIFICIAL_ONLY'))
        self.cfg=dict(schema=c.SERIES,source_commit='1'*40,physical=deepcopy(c.policy.PHYSICAL),
            candidates=[asdict(x) for x in c.policy.CANDIDATES],budget=c.counts(),limitations=list(c.LIMITS),
            files=files,source_files=sources,manifest=str(manifest),origin_contract='ARTIFICIAL',
            origin_run_root='ARTIFICIAL_ORIGIN',runtime={},run_root=str(self.assets/'EXPLORATORY_run01'),
            retry=False,mandatory_STOP=True)

    def test_synthetic_seal_and_source_binding(self):
        self.assertEqual(c.verify(self.cfg,runtime=False),{'ARTIFICIAL_ONLY':True})

    def test_input_SHA_drift_rejected(self):
        Path(self.cfg['manifest']).write_text('{"drift":true}')
        with self.assertRaises(c.Stop):c.verify(self.cfg,runtime=False)

    def test_runtime_file_SHA_drift_rejected(self):
        p=self.assets/'ARTIFICIAL_RUNTIME';p.write_text('sealed')
        self.cfg['files'].append(dict(path=str(p),sha256=c.sha(p),role='runtime:ARTIFICIAL_ONLY'))
        p.write_text('changed')
        with self.assertRaises(c.Stop):c.verify(self.cfg,runtime=False)

    def test_fixed_SOURCE_rejected_even_after_local_rehash(self):
        row=self.cfg['source_files'][0];p=Path(row['local_path']);p.write_text('# substituted')
        row['sha256']=c.sha(p);self.cfg['files'][0]['sha256']=c.sha(p)
        with self.assertRaises(c.Stop):c.verify(self.cfg,runtime=False)

    def test_budget_cannot_add_compile(self):
        self.cfg['budget']['compile']=1
        with self.assertRaises(c.Stop):c.verify(self.cfg,runtime=False)

    def test_physical_conditions_cannot_change(self):
        self.cfg['physical']['T']=1.
        with self.assertRaises(c.Stop):c.verify(self.cfg,runtime=False)

    def test_inherited_limits_cannot_disappear(self):
        self.cfg['limitations'].pop()
        with self.assertRaises(c.Stop):c.verify(self.cfg,runtime=False)


def mutation(label,fn):
    def test(self):
        p,f=costs_fixture();fn(p,f)
        with self.assertRaises(c.Stop):c.validate_costs(p,f)
    setattr(ControlTests,'test_cost_reject_'+label,test)


for label,fn in {
 'missing_axis':lambda p,f:p.pop(),
 'duplicate_axis':lambda p,f:p.__setitem__(0,deepcopy(p[1])),
 'wrong_candidate':lambda p,f:p[0].update(candidate='UNAPPROVED'),
 'wrong_cohort':lambda p,f:p[0].update(cohort='CONFIRMATION'),
 'wrong_runtime':lambda p,f:p[0].update(runtime_id='0'*64),
 'wrong_units':lambda p,f:p[0].update(unit='T_count'),
 'zero_missing_cost':lambda p,f:p[0].update(RZ=0),
 'boolean_cost':lambda p,f:p[0].update(RZ=True),
 'old_Full_1367':lambda p,f:f[0].update(RZ=1367),
 'old_Full_2033':lambda p,f:f[-1].update(RZ=2033),
 'wrong_Full_alias':lambda p,f:f[0].update(requested_candidate='OTHER'),
 'wrong_Full_axis':lambda p,f:f[0].update(axis='other')}.items():mutation(label,fn)


if __name__=='__main__':unittest.main()
