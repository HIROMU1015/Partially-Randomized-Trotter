"""Synthetic metadata and pure gates; no science fixtures or production launch."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

HELPER=None


class ObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h=HELPER;cls.r=HELPER.resources;cls.Stop=HELPER.identity.Stop

    def fixture(self):
        root='/fixture/cgroup'
        psi='some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n'
        files={'/proc/self/cgroup':'0::/a/b\n',
            '/proc/self/mountinfo':'1 2 0:30 / '+root+' rw - cgroup2 cgroup2 rw\n',
            '/proc/meminfo':'MemAvailable: '+str(80*2**20)+' kB\n',
            '/proc/self/status':'Cpus_allowed_list:\t0-5\n',
            '/proc/pressure/memory':psi,root+'/cgroup.controllers':'cpu memory pids\n'}
        for path in (root+'/a/b',root+'/a'):
            files.update({path+'/memory.max':'max\n',path+'/memory.current':'0\n',
                path+'/memory.events':'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n',path+'/memory.pressure':psi})
        return files

    def observe(self,files,namespace=None,read_calls=None):
        namespace=namespace or 'cgroup:[%d]'%self.r.INITIAL_CGROUP_NS_INO
        def read(path,*a,**kw):
            name=str(path)
            if read_calls is not None:read_calls.append(name)
            if name not in files:raise FileNotFoundError(name)
            value=files[name]
            if isinstance(value,BaseException):raise value
            return value
        with patch.object(Path,'read_text',read),patch.object(self.r.os,'readlink',return_value=namespace),\
             patch.object(self.r.time,'monotonic',return_value=10):
            return self.r.observe_memory()

    def test_proven_v2_root_missing_control_interfaces_is_normal(self):
        files=self.fixture();calls=[];obs=self.observe(files,read_calls=calls)
        self.assertEqual(obs['available'],80*2**30)
        self.assertEqual([e['is_root'] for e in obs['hierarchy']],[False,False,True])
        self.assertEqual(obs['cgroup_namespace'],'cgroup:[4026531835]')
        for name in ('memory.max','memory.current','memory.events','memory.pressure'):
            self.assertNotIn('/fixture/cgroup/'+name,calls)
        self.assertIn('/fixture/cgroup/a/memory.events',calls)
        self.assertIn('/fixture/cgroup/a/b/memory.pressure',calls)
        self.assertIn('/proc/pressure/memory',calls)

    def test_leaf_finite_remaining_stricter_than_host(self):
        files=self.fixture();files['/fixture/cgroup/a/b/memory.max']=str(64*2**30)
        files['/fixture/cgroup/a/b/memory.current']=str(8*2**30)
        self.assertEqual(self.observe(files)['available'],56*2**30)

    def test_strictest_ancestor_is_kept(self):
        files=self.fixture()
        files.update({'/fixture/cgroup/a/b/memory.max':str(70*2**30),'/fixture/cgroup/a/b/memory.current':str(2*2**30),
                      '/fixture/cgroup/a/memory.max':str(60*2**30),'/fixture/cgroup/a/memory.current':str(20*2**30)})
        self.assertEqual(self.observe(files)['available'],40*2**30)

    def test_unlimited_and_finite_ancestors_mixed(self):
        files=self.fixture();files['/fixture/cgroup/a/memory.max']=str(64*2**30)
        files['/fixture/cgroup/a/memory.current']=str(4*2**30)
        obs=self.observe(files)
        self.assertEqual(obs['available'],60*2**30)
        self.assertIsNone(obs['hierarchy'][0]['maximum'])

    def test_host_is_still_minimum_after_all_interfaces_read(self):
        files=self.fixture();files['/fixture/cgroup/a/memory.max']=str(200*2**30)
        calls=[];obs=self.observe(files,read_calls=calls)
        self.assertEqual(obs['available'],80*2**30)
        self.assertEqual(len(obs['oom_events']),2)
        self.assertIn('/fixture/cgroup/a/memory.current',calls)

    def test_private_or_unknown_namespace_cannot_fake_root_slash(self):
        for namespace in ('cgroup:[4026532183]','cgroup:[0]','unknown'):
            with self.assertRaisesRegex(self.Stop,'initial cgroup namespace'):
                self.observe(self.fixture(),namespace)

    def test_namespace_link_unreadable_is_not_guessed(self):
        with patch.object(self.r.os,'readlink',side_effect=PermissionError('namespace hidden')):
            with self.assertRaises(PermissionError):
                self.r.cgroup_directories('0::/a/b','1 2 0:30 / /fixture/cgroup rw - cgroup2 cgroup2 rw')

    def test_nonroot_mount_root_hides_upstream_limits(self):
        files=self.fixture();files['/proc/self/mountinfo']='1 2 0:30 /a /fixture/cgroup rw - cgroup2 cgroup2 rw\n'
        with self.assertRaisesRegex(self.Stop,'hidden upstream'):self.observe(files)

    def test_shadow_mount_and_ambiguous_memory_mount_rejected(self):
        for line in ('2 1 0:31 / /fixture/cgroup/a rw - tmpfs tmpfs rw\n',
                     '2 1 0:30 / /duplicate rw - cgroup2 cgroup2 rw\n'):
            files=self.fixture();files['/proc/self/mountinfo']+=line
            with self.assertRaises(self.Stop):self.observe(files)

    def test_escaped_dotdot_relative_membership_and_mount_rejected(self):
        for value in ('0::/../a\n','0::relative\n','0::/a/./b\n','0::/a\\040b\n'):
            files=self.fixture();files['/proc/self/cgroup']=value
            with self.assertRaises(self.Stop):self.observe(files)
        files=self.fixture();files['/proc/self/mountinfo']='1 2 0:30 / /fixture/cgroup\\040name rw - cgroup2 cgroup2 rw\n'
        with self.assertRaises(self.Stop):self.observe(files)

    def test_missing_controller_or_membership_never_host_only_fallback(self):
        files=self.fixture();files['/fixture/cgroup/cgroup.controllers']='cpu pids\n'
        with self.assertRaises(self.Stop):self.observe(files)
        files=self.fixture();files['/proc/self/cgroup']=''
        with self.assertRaises(self.Stop):self.observe(files)

    def test_membership_or_mount_visibility_changes_stop(self):
        files=self.fixture();original=Path.read_text;seen=0
        def read(path,*a,**kw):
            nonlocal seen
            if str(path)=='/proc/self/cgroup':
                seen+=1
                return files[str(path)] if seen==1 else '0::/different\n'
            return files[str(path)]
        with patch.object(Path,'read_text',read),patch.object(self.r.os,'readlink',return_value='cgroup:[4026531835]'):
            with self.assertRaisesRegex(self.Stop,'visibility changed'):self.r.observe_memory()

    def test_host_and_nonroot_pressure_both_preserved(self):
        for name in ('/proc/pressure/memory','/fixture/cgroup/a/memory.pressure','/fixture/cgroup/a/b/memory.pressure'):
            files=self.fixture();files[name]=files[name].replace('full avg10=0.00','full avg10=0.10')
            obs=self.observe(files);self.assertEqual(obs['psi_full_avg10'],.1)
            with self.assertRaisesRegex(self.Stop,'PSI pressure'):
                self.r.Monitor(1,obs,clock=lambda:10)

    def test_nonroot_oom_ancestor_delta_stop(self):
        baseline=self.observe(self.fixture());monitor=self.r.Monitor(1,baseline,clock=lambda:10)
        for path in ('/fixture/cgroup/a/memory.events','/fixture/cgroup/a/b/memory.events'):
            files=self.fixture();files[path]='oom 1\noom_kill 2\n'
            observed=self.observe(files)
            with self.assertRaisesRegex(self.Stop,'OOM'):monitor.check(observed,[0])

    def test_memory_admission_boundaries_freshness_headroom_RSS(self):
        self.assertEqual(self.r.admission(72*2**30,10,6,range(6),range(6),now=10),6)
        self.assertEqual(self.r.admission(72*2**30-1,10,6,range(6),range(6),now=10),5)
        self.assertEqual(self.r.admission(32*2**30,10,6,[0],[0],now=10),1)
        for available,at in ((32*2**30-1,10),(80*2**30,4)):
            with self.assertRaises(self.Stop):self.r.admission(available,at,6,range(6),range(6),now=10)
        obs=self.observe(self.fixture());monitor=self.r.Monitor(1,obs,clock=lambda:10)
        with self.assertRaises(self.Stop):monitor.check({**obs,'available':16*2**30-1},[0])
        with self.assertRaises(self.Stop):monitor.check(obs,[8*2**30+1])

    def test_empty_CPU_permission_and_subset_mismatch_rejected(self):
        with self.assertRaises(self.Stop):self.r.admission(80*2**30,10,6,[],range(6),now=10)
        self.h.identity.require({0,1}<={0,1,2},'process can use unpermitted CPU')
        with self.assertRaises(self.Stop):self.h.identity.require({0,1,2}<={0,1},'process can use unpermitted CPU')
        old=self.h.blob(self.h.OLD_SOURCE,'src/trottertracks/resource_applicability/h4_geometry/execution.py')
        self.assertEqual(old,(self.h.ROOT/'src/trottertracks/resource_applicability/h4_geometry/execution.py').read_bytes())

    def test_v1_legacy_root_interfaces_and_ancestors_retained(self):
        files=self.fixture();files['/proc/self/cgroup']='2:memory:/a/b\n'
        files['/proc/self/mountinfo']='1 2 0:30 / /fixture/cgroup rw - cgroup memory rw,memory\n'
        for path in ('/fixture/cgroup','/fixture/cgroup/a','/fixture/cgroup/a/b'):
            files[path+'/memory.limit_in_bytes']=str(2**63-4096)
            files[path+'/memory.usage_in_bytes']='0\n';files[path+'/memory.failcnt']='0\n'
        files['/fixture/cgroup/a/memory.limit_in_bytes']=str(64*2**30)
        self.assertEqual(self.observe(files)['available'],64*2**30)

    def test_source_invariants_identity_and_old_bundle_audit(self):
        _contract,audit=self.h.source_audit()
        self.assertEqual(len(audit['new_source_hashes']),17)
        self.assertEqual(len(audit['unchanged_previous_source_hashes']),15)
        self.assertEqual(set(audit['changed_previous_sources']),self.h.CHANGED)
        self.assertIn('Monitor',audit['invariant_resource_AST_nodes'])
        self.assertIn('admission',audit['invariant_resource_AST_nodes'])
        self.assertEqual(audit['dependency_count'],45)
        self.assertEqual(audit['additional_transpile'],0)


def missing_file_test(name):
    def test(self):
        files=self.fixture();del files['/fixture/cgroup/a/'+name]
        with self.assertRaises(FileNotFoundError):self.observe(files)
    return test
for name in ('memory.max','memory.current','memory.events','memory.pressure'):
    setattr(ObserverTests,'test_nonroot_missing_'+name.replace('.','_'),missing_file_test(name))


def bad_file_test(name,value):
    def test(self):
        files=self.fixture();files['/fixture/cgroup/a/'+name]=value
        with self.assertRaises((HELPER.identity.Stop,PermissionError)):self.observe(files)
    return test
for label,name,value in (
    ('permission','memory.max',PermissionError('denied')),
    ('max_negative','memory.max','-1'),('max_nan','memory.max','NaN'),('max_float','memory.max','1.5'),
    ('current_negative','memory.current','-2'),('current_bad','memory.current','bad'),
    ('oom_missing','memory.events','oom 0\n'),('oom_negative','memory.events','oom -1\noom_kill 0\n'),
    ('oom_duplicate','memory.events','oom 0\noom 1\noom_kill 0\n'),
    ('pressure_nan','memory.pressure','full avg10=nan avg60=0 avg300=0 total=0\n'),
    ('pressure_missing','memory.pressure','some avg10=0 avg60=0 avg300=0 total=0\n')):
    setattr(ObserverTests,'test_nonroot_reject_'+label,bad_file_test(name,value))


class BindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h=HELPER;cls.g=HELPER.gates;cls.i=HELPER.identity
        directory=HELPER.ROOT/HELPER.AUTH_BUNDLE
        cls.saved=tuple(json.loads((directory/n).read_bytes()) for n in ('input_generation_plan_v2.json','authorization_draft_v2.json','stage_review_v2.json'))
        cls.old=tuple(json.loads(HELPER.blob(HELPER.BASE,HELPER.OLD_AUTH+'/'+n)) for n in ('input_generation_plan_v1.json','authorization_draft_v1.json','stage_review_v1.json'))

    def simulated(self):
        p,a,r=copy.deepcopy(self.saved);a['allowed_cpus']=[0];r['approved']=True
        self.bind(p,a,r);return p,a,r

    def bind(self,p,a,r):
        a['plan_fingerprint']=self.i.fingerprint('h4-execution-plan-v1',p)
        r['plan_fingerprint']=a['plan_fingerprint'];r['authorization_digest']=self.i.fingerprint('h4-authorization-v1',a)

    def permit(self,documents):return self.g.authorize('input_generation',*documents,explicit_launch=True)

    def test_saved_review_false_and_CPU_empty_remain_rejected(self):
        self.g.structural_gate(*self.saved)
        self.assertIs(self.saved[2]['approved'],False);self.assertEqual(self.saved[1]['allowed_cpus'],[])
        with self.assertRaisesRegex(self.i.Stop,'review/authorization'):self.permit(self.saved)
        p,a,r=copy.deepcopy(self.saved);r['approved']=True
        with self.assertRaisesRegex(self.i.Stop,'explicit CPU permission'):self.permit((p,a,r))

    def test_simulated_metadata_checkout_new_SOURCE_only(self):
        permit=self.permit(self.simulated());contract,options=self.g.checkout_gate(permit)
        self.assertEqual(permit.source_root,str(self.h.ROOT));self.assertEqual(options['num_processes'],1)
        self.assertEqual(contract['templates'],self.saved[0]['templates'])
        audit=json.loads((self.h.ROOT/self.g.SOURCE_AUDIT).read_bytes())
        self.assertEqual(audit['source_commit'],self.saved[0]['source_commit'])
        self.assertEqual({**audit['new_source_hashes'],**audit['namespace_parent_hashes']},self.saved[0]['source_hashes'])

    def test_new_old_PLAN_AUTH_REVIEW_mixing_rejected(self):
        new=self.simulated();old=copy.deepcopy(self.old)
        old[1]['allowed_cpus']=[0];old[2]['approved']=True;self.bind(*old)
        for documents in ((new[0],old[1],new[2]),(new[0],new[1],old[2]),(old[0],new[1],new[2])):
            with self.assertRaises(self.i.Stop):self.permit(documents)

    def test_old_source_or_hash_or_audit_binding_rejected_even_rebound(self):
        for key in ('source_commit','source_hashes','source_audit_sha256','source_root'):
            documents=self.simulated();documents[0][key]=copy.deepcopy(self.old[0][key]);self.bind(*documents)
            with self.assertRaises(self.i.Stop):self.g.checkout_gate(self.permit(documents))

    def test_scope_resource_permission_conditions_not_changed(self):
        p,a,r=self.saved;oldp,olda,oldr=self.old
        changes={'source_commit','source_hashes','source_root','source_audit_sha256'}
        self.assertEqual({k:v for k,v in p.items() if k not in changes},{k:v for k,v in oldp.items() if k not in changes})
        self.assertEqual({k:v for k,v in a.items() if k!='plan_fingerprint'},{k:v for k,v in olda.items() if k!='plan_fingerprint'})
        self.assertEqual(p['requested_workers'],6);self.assertIsNone(p['inputs']);self.assertIsNone(p['generation_freeze_digest'])
        self.assertEqual(len(p['templates']),218)

    def test_no_explicit_launch_or_wrong_stage_or_permission(self):
        with self.assertRaises(self.i.Stop):self.g.authorize('input_generation',*self.simulated(),explicit_launch=False)
        with self.assertRaises(self.i.Stop):self.g.authorize('signal_compile',*self.simulated(),explicit_launch=True)
        documents=self.simulated();documents[1]['permission']='signal_compile';self.bind(*documents)
        with self.assertRaises(self.i.Stop):self.permit(documents)

    def test_real_plan_source_blobs_unchanged_after_SOURCE_commit(self):
        plan=self.saved[0]
        for path,value in plan['source_hashes'].items():
            data=(self.h.ROOT/path).read_bytes()
            self.assertEqual(self.i.sha(data),value)
            self.assertEqual(data,self.h.blob(plan['source_commit'],path))
        self.assertIs(self.saved[2]['approved'],False)
