"""Common-entry -> real accept_saved/C input; no scientific arithmetic.

Fixture commits live ONLY in TemporaryDirectory Git repositories, using the
project object store read-only as an alternate. Nothing is indexed as a real mean.
"""
import ast
from contextlib import contextmanager
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from trottertracks import rpe_pauli_signal_intake as A
from trottertracks import rpe_c_interfaces as C
from trottertracks import rpe_total_binding as T
from trottertracks import rpe_total_risk as Q
from check_rpe_pauli_integration import forbid_science


def blob(commit, path):
    return subprocess.check_output(['git', 'show', commit+':'+path], cwd=ROOT)


def packet_function():
    nodes = [n for n in ast.parse(blob(A.ENTRY_SOURCE, A.ENTRY_MODULE)).body
             if isinstance(n, ast.FunctionDef) and n.name == 'packet']
    assert len(nodes) == 1
    scope = dict(deepcopy=deepcopy, MODULE=A.ENTRY_MODULE, HELPER=A.ENTRY_DICTIONARY_HELPER,
                 p1=NS(B=A.p.SavedArtifacts(ROOT).read(A.HANDOFF)['packets'][0]['B_source']),
                 LOCAL=A.p.SavedArtifacts(ROOT).read(A.LD3_HANDOFF)['packets'][0]['rounds'][0]['status'])
    exec(compile(ast.Module(body=nodes, type_ignores=[]), A.ENTRY_SOURCE+':'+A.ENTRY_MODULE, 'exec'), scope)
    return scope['packet']


@contextmanager
def metadata_only():
    with forbid_science(), patch.object(Q, 'evaluate', side_effect=AssertionError('No Q3')), \
            patch.object(T, 'evaluate', side_effect=AssertionError('No Q3 binding')):
        yield


class Repository:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory(prefix='pauli-intake-fixture-')
        self.root = Path(self.temp.name)
        self.git('init', '-q')
        common = subprocess.check_output(['git','rev-parse','--git-common-dir'],cwd=ROOT,text=True).strip()
        objects = (ROOT/common/'objects').resolve()
        (self.root/'.git/objects/info/alternates').write_text(str(objects)+'\n')
        for name in A.ENTRY_FILES:
            path=self.root/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob(A.ENTRY_SOURCE,name))
        self.commit()

    def git(self, *args):
        return subprocess.check_output(['git',*args],cwd=self.root,stderr=subprocess.PIPE,text=True).strip()

    def write(self, path, obj):
        dst=self.root/path;dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_text(json.dumps(obj,sort_keys=True,allow_nan=False)+'\n')

    def commit(self):
        self.git('add','.')
        self.git('-c','user.name=Synthetic Fixture','-c','user.email=fixture@example.invalid',
                 'commit','-qm','Temporary artificial intake fixture; not scientific evidence')
        return self.git('rev-parse','HEAD')

    def save(self, value, name='MEAN_HANDOFF.json'):
        self.write('fixtures/'+name,value)
        return dict(commit=self.commit(),path='fixtures/'+name)

    def close(self):
        self.temp.cleanup()


class EntryIntake(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet=staticmethod(packet_function())
        cls.store=A.p.SavedArtifacts(ROOT);cls.saved={}
        for ref in (A.HANDOFF,A.LD3_HANDOFF,A.LD5_HANDOFF):
            handoff=cls.store.read(ref)
            for i,pk in enumerate(handoff['packets']):
                cls.saved[pk['prefix']]=(ref,i,pk)

    def accept(self, repo, ref):
        with metadata_only():
            return A.accept_saved(repo.root,integration=C.interfaces(str(repo.root)).pipeline,handoff_source=ref)

    def test_saved_four_packets_match_existing_route(self):
        repo=Repository()
        try:
            for prefix,(ref,i,pk) in self.saved.items():
                config=pk['configuration']
                rows={r['m']:(deepcopy(r['mean_record']),deepcopy(r['mean_source']),True) for r in pk['rounds']}
                handoff=self.packet(config,rows,configuration_source=dict(ref,selector=['packets',i,'configuration']))
                before=deepcopy(handoff)
                entry=self.accept(repo,repo.save(handoff,'LD%d.json'%prefix))
                with metadata_only():
                    old=A.accept_saved(ROOT,integration=C.interfaces(str(ROOT)).pipeline,handoff_source=ref)['results'][i]
                actual=entry['results'][0];a=actual['signal_packet'];b=old['signal_packet']
                self.assertEqual(actual['configuration'],old['configuration'])
                for key in ('target','means','dictionary_identity','implementation_sources','target_view_source'):
                    self.assertEqual(a[key],b[key],(prefix,key))
                self.assertEqual(actual['signal_input']['means'],old['signal_input']['means'])
                self.assertEqual(actual['C_plan_price_bridge'],'SOURCE_BOUND_PAULI_PHYSICAL_BRIDGE_METADATA_ONLY')
                self.assertEqual(entry['science_calls'],0)
                self.assertEqual(handoff,before)
                self.assertEqual(len(a['means']),13)
                self.assertTrue(all(m['qualification']=='UNQUALIFIED' for m in a['means']))
                for key in ('sufficient_N','global_RMSE_bound','G','winner'):
                    self.assertIsNone(entry[key]);self.assertIsNone(a[key])
                # A's local-JSON, identical-reuse assemble mode has no separate
                # configuration locator. The packet still pins the whole config.
                inline=deepcopy(handoff);inline['configuration_source']=None
                accepted=self.accept(repo,repo.save(inline,'inline_LD%d.json'%prefix))
                self.assertEqual(accepted['results'][0]['configuration'],config)
                self.assertEqual(accepted['results'][0]['signal_packet']['means'],a['means'])
        finally:repo.close()

    def fixture(self, repo, edit):
        _,_,pk=self.saved[3];spec=deepcopy(pk['configuration']['spec'])
        helpers=A._ld3_metadata_helpers(str(repo.root))
        if edit=='kappa':
            f=A.B.a.exact_scalar(spec['kappa']['value'])*Fraction(3,2)
            spec['kappa']['value']=A.B.a.exact_value(f)
        elif edit in ('delta','exact_delta'):
            spec['delta_pf']=A.B.a.exact_value(Fraction(.2) if edit=='delta' else Fraction(1,5))
        elif edit=='prefix':
            dictionary=self.store.read(pk['dictionary_identity']['source']);dictionary['prefix']=4
            dictionary['collector_metadata']['artificial_fixture']='SYNTHETIC_NOT_A_MEAN'
            path='fixtures/dictionary/LD4_DICTIONARY.json';repo.write(path,dictionary)
            repo.write('fixtures/dictionary/result.json',dict(artificial_fixture='SYNTHETIC_METADATA_NOT_P0_EVIDENCE'))
            source=dict(commit=repo.commit(),path=path)
            raw=json.dumps(dictionary,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            import hashlib
            descriptor=helpers.dictionary_identity(dict(payload=dictionary,
                dictionary_sha256=hashlib.sha256(raw).hexdigest(),source=source))
            spec['L_D']=spec['split']['L_D']=4
            spec['split']['tail_dictionary']=spec['law']['tail_dictionary']=descriptor
            spec['split']['lambda_source']=dict(source,selector=['lambda_P'])
        config=A.B.generate(spec)
        self.assertNotEqual(config['id'],pk['configuration']['id'])
        api=NS(validate=A.B._validate,exact_scalar=A.B.a.exact_scalar)
        for m in range(13):
            rec=deepcopy(pk['rounds'][m]['mean_record']);job=helpers.job(config,m,rec['source'],api)
            for key in ('m','B_configuration_id','B_candidate_id','B_identity','B_law_binding','physical_context'):
                rec[key]=deepcopy(job[key])
            rec['candidate']=deepcopy(job['physical_context']['physical_candidate'])
            for key in ('signed_weights','allocation','event_count'):rec[key]=deepcopy(config['rounds'][m]['structure'][key])
            # Explicit artificial observations, never a physical signal evaluation.
            rec['signal'].update(raw=dict(real=.25,imag=-.5),corrected=dict(real=.25,imag=-.5),reference=dict(real=.25,imag=-.5))
            rec['artificial_fixture']='SYNTHETIC_NOT_A_MEAN'
            repo.write('fixtures/%s/round%d.json'%(edit,m),dict(record=rec))
        repo.write('fixtures/%s/CONFIGURATION.json'%edit,config);raw_commit=repo.commit()
        rows={m:(json.loads((repo.root/('fixtures/%s/round%d.json'%(edit,m))).read_text())['record'],
                 dict(commit=raw_commit,path='fixtures/%s/round%d.json'%(edit,m),selector=['record'],
                      scientific_source=pk['rounds'][m]['mean_record']['source']['commit']),False) for m in range(13)}
        handoff=self.packet(config,rows,configuration_source=dict(commit=raw_commit,path='fixtures/%s/CONFIGURATION.json'%edit),
                            raw_result_source=dict(commit=raw_commit,path='fixtures/%s/round0.json'%edit))
        handoff['artificial_fixture']='SYNTHETIC_NOT_A_MEAN_NOT_FOR_SAVED_CATALOG'
        return handoff,repo.save(handoff,edit+'.json')

    def test_changed_kappa_delta_and_unlisted_LD_use_normal_intake(self):
        repo=Repository()
        try:
            for edit in ('kappa','delta','prefix'):
                handoff,ref=self.fixture(repo,edit);before=deepcopy(handoff)
                out=self.accept(repo,ref);result=out['results'][0];view=result['signal_packet']
                self.assertEqual(result['configuration'],handoff['packets'][0]['configuration'])
                self.assertEqual(view['source'],ref)
                self.assertEqual(out['assembly_implementation_source'],dict(commit=A.ENTRY_SOURCE,path=A.ENTRY_MODULE))
                self.assertEqual([m['raw'] for m in view['means']],[dict(real=.25,imag=-.5)]*13)
                self.assertFalse(result['readiness']['statistics_executed'])
                self.assertEqual(result['C_plan_price_bridge'],'SOURCE_BOUND_PAULI_PHYSICAL_BRIDGE_METADATA_ONLY')
                self.assertEqual(handoff,before)
        finally:repo.close()

    def test_unknown_or_changed_assembler_and_helper_rejected(self):
        repo=Repository()
        try:
            h,ref=self.fixture(repo,'kappa')
            bad=deepcopy(h);bad['B_intake_dictionary_identity']['commit']='0'*40
            with self.assertRaisesRegex(ValueError,'dictionary helper'):self.accept(repo,repo.save(bad))
            bad=deepcopy(h);bad['assembly']='other.py'
            with self.assertRaisesRegex(ValueError,'entry assembly'):self.accept(repo,repo.save(bad))
            for name in A.ENTRY_FILES:
                path=repo.root/name;before=path.read_bytes();path.write_bytes(before+b'\n# changed\n')
                changed=repo.save(h)
                with self.assertRaisesRegex(ValueError,'entry code differs'):self.accept(repo,changed)
                path.write_bytes(before);repo.commit()
            with self.assertRaisesRegex(ValueError,'assembly SOURCE'):
                self.accept(repo,dict(A.LD3_HANDOFF,commit='0'*40))
        finally:repo.close()

    def test_non_binary64_exact_step_still_obeys_C_price_bridge_guard(self):
        repo=Repository()
        try:
            _,ref=self.fixture(repo,'exact_delta')
            # Signal intake itself is valid; do not weaken C's separate exact
            # physical-step comparison to force a price join for this grid.
            pipeline=C.interfaces(str(repo.root)).pipeline
            with metadata_only():
                out=A.accept_saved(repo.root,integration=NS(signal_input=pipeline.signal_input),handoff_source=ref)
            self.assertEqual(out['results'][0]['C_plan_price_bridge'],'UNKNOWN_PAULI_PHYSICAL_ID_ADAPTER_REQUIRED')
            with self.assertRaisesRegex(ValueError,'Pauli physical recipe differs'):
                self.accept(repo,ref)
        finally:repo.close()

    def test_configuration_carrier_law_and_dictionary_mismatches_rejected(self):
        repo=Repository()
        try:
            h,ref=self.fixture(repo,'delta')
            variants=[]
            bad=deepcopy(h);bad['configuration_source']['selector']=['missing'];variants.append(bad)
            bad=deepcopy(h);bad['packets'][0]['rounds'][0]['candidate_id']='OTHER';variants.append(bad)
            bad=deepcopy(h);bad['packets'][0]['rounds'][0]['law_binding']={};variants.append(bad)
            bad=deepcopy(h);bad['packets'][0]['rounds'][0]['mean_source']['scientific_source']='OTHER';variants.append(bad)
            bad=deepcopy(h);bad['packets'][0]['rounds'][0]['mean_record']['signal']['raw']=[0.,0.];variants.append(bad)
            bad=deepcopy(h);bad['dictionary_source']=dict(A.LD5_DICTIONARY);variants.append(bad)
            bad=deepcopy(h);bad['packets'][0]['rounds'][0]['N_X']=3;variants.append(bad)
            for bad in variants:
                with self.assertRaises((ValueError,KeyError)):self.accept(repo,repo.save(bad))
        finally:repo.close()


if __name__=='__main__':unittest.main(verbosity=2)
