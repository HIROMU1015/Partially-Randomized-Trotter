"""Invented scalar, literal IR and 2-qubit controls; no registered provider."""
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import importlib.util

import numpy as np
from trottertracks.algorithm_codesign.rte_development.key_mechanisms import (
    inventory_terms,linear_price,one_key_boundary,symmetric_root_split)
from trottertracks.algorithm_codesign.rte_development.boundary import (
    gate,inverse,simplify,cancellation_depth,native_price,aggregate_pairs)
from trottertracks.algorithm_codesign.rte_development.boundary_launch import validate_binding,verify_launch
from trottertracks.algorithm_codesign.rte_development.launch import consume_marker,Guard
from trottertracks.algorithm_codesign.rte_development.evidence import EvidenceWriter,verify_chunks
from trottertracks.algorithm_codesign.g9_native import simplify as old_pass

ROOT=Path(__file__).resolve().parents[3]
CP=ROOT/'artifacts/track_b_d2_boundary_source/2026-10-13/v1/contract_v1.json'
CACHE={'1/9':{'T_count':7,'one_qubit_count':19},'2/13':{'T_count':11,'one_qubit_count':29}}


def matrix(g):
    """Two-qubit invented matrices. W phase is retained, never quotiented out."""
    name=g[0];I=np.eye(2,dtype=complex)
    one={'H':np.array([[1,1],[1,-1]])/np.sqrt(2),'X':np.array([[0,1],[1,0]]),
         'Y':np.array([[0,-1j],[1j,0]]),'Z':np.diag([1,-1]),
         'T':np.diag([1,np.exp(1j*np.pi/4)]),'t':np.diag([1,np.exp(-1j*np.pi/4)]),
         'S':np.diag([1,1j]),'s':np.diag([1,-1j])}
    if name in ('W','w'):return np.exp((1 if name=='W' else -1)*1j*np.pi/4)*np.eye(4)
    if name in ('CX','CZ'):
        out=np.zeros((4,4),complex)
        for i in range(4):
            bits=[i//2,i%2];j=i
            if name=='CX' and bits[g[1]]:j=i^(2 if g[2]==0 else 1)
            out[j,i]=(-1 if name=='CZ' and bits[0]*bits[1] else 1)
        return out
    if name=='R':
        t=g[3]*np.arctan(float(F(g[2])))
        u=np.diag([np.exp(-1j*t/2),np.exp(1j*t/2)])
    else:u=one[name]
    return np.kron(u,I) if g[1]==0 else np.kron(I,u)


def circuit(gs):
    out=np.eye(4,dtype=complex)
    for g in gs:out=matrix(g)@out
    return out


class Prices(unittest.TestCase):
    def test_literal_separate(self):
        l,m=inventory_terms([['T',0],['t',1],['R',0,'1/9',1],['R',0,'1/9',-1]])
        self.assertEqual((l,m['1/9']),(2,2));self.assertEqual(linear_price(l,m,{'1/9':7}),16)
    def test_missing_price_is_not_free(self):
        with self.assertRaises(KeyError):linear_price(0,{'1/9':1},{})
    def test_root_prob_price_identity(self):
        v=symmetric_root_split(F(2,7),F(3,11),14,18)
        self.assertEqual(v['total'],F(3,11)*18-F(2,7)*14)
    def test_single_key_tie(self):
        v=one_key_boundary(-7,2,10)
        self.assertEqual(v['tie_price'],F(27,2));self.assertEqual(v['strict_sign_flip_integer'],14)
    def test_zero_coefficient(self):self.assertIsNone(one_key_boundary(-1,0,7)['tie_price'])
    def test_unreachable_negative_price(self):
        v=one_key_boundary(-20,-1,5);self.assertFalse(v['nonnegative_tie']);self.assertIsNone(v['strict_sign_flip_integer'])
    def test_decreasing_price_can_flip(self):self.assertEqual(one_key_boundary(-2,-1,7)['strict_sign_flip_integer'],0)
    def test_exact_integer_tie_is_not_strict_flip(self):self.assertEqual(one_key_boundary(-4,2,8)['strict_sign_flip_integer'],11)


class Boundary(unittest.TestCase):
    def test_signed_R_inverse(self):self.assertEqual(inverse(('R',0,'1/9',1)),('R',0,'1/9',-1))
    def test_all_inverse_matrices(self):
        gates=[('H',0),('X',1),('Y',0),('Z',1),('T',1),('t',0),('S',1),('s',0),('W',),('w',),('CX',0,1),('CZ',0,1),('R',0,'2/13',1)]
        for g in gates:self.assertLess(np.linalg.norm(matrix(inverse(g))@matrix(g)-np.eye(4)),1e-13)
    def test_strict_phase_survives(self):
        gs=[('Z',0),('H',1),('H',1),('W',)]
        self.assertLess(np.linalg.norm(circuit(gs)-circuit(simplify(gs))),1e-13)
        self.assertGreater(np.linalg.norm(circuit(gs)-np.eye(4)),1)
    def test_cascade(self):
        a=[('X',0),('T',1),('H',0)];b=[('H',0),('t',1),('X',0)]
        self.assertEqual(cancellation_depth(a,b),3);self.assertEqual(simplify(a+b),[])
    def test_no_commutation(self):
        a=[('T',0),('H',1)];b=[('t',0)]
        self.assertEqual(cancellation_depth(a,b),0);self.assertEqual(len(simplify(a+b)),3)
    def test_same_key_different_wire_not_cancelled(self):self.assertEqual(cancellation_depth([('R',0,'1/9',1)],[('R',1,'1/9',-1)]),0)
    def test_same_angle_same_sign_not_fused(self):self.assertEqual(len(simplify([('R',0,'1/9',1)]*2)),2)
    def test_same_rule_as_frozen_pass(self):
        alphabet=[('H',0),('Z',1),('T',0),('t',0),('CX',0,1),('R',0,'1/9',1),('R',0,'1/9',-1)]
        for seq in product(alphabet,repeat=3):self.assertEqual(simplify(seq),old_pass(list(seq)))
    def test_pair_algebra_matches_direct_stack(self):
        fixtures=[[('X',0),('T',1),('H',0)],[('H',0),('t',1),('Z',0)],[('R',1,'1/9',1)],[('R',1,'1/9',-1)]]
        for a,b in product(fixtures,repeat=2):
            k=cancellation_depth(a,b);v=a[:-k]+b[k:] if k else a+b
            self.assertEqual(v,simplify(a+b));self.assertLess(np.linalg.norm(circuit(a+b)-circuit(v)),1e-13)
    def test_actual_sequence_phase_adjoint(self):
        # Invented sequence is phaseful T W. Its negative factor is its exact adjoint.
        a=[('T',0),('W',)];b=[('w',),('t',0)]
        self.assertLess(np.linalg.norm(circuit(a+b)-np.eye(4)),1e-13)
        self.assertEqual(simplify(a+b),[])
    def test_unsupported_gate(self):
        with self.assertRaises(ValueError):gate(('arbitrary',0))
    def test_bad_wire(self):
        with self.assertRaises(ValueError):gate(('CX',0,0))
    def test_missing_key(self):
        with self.assertRaises(KeyError):native_price([('R',0,'3/17',1)],CACHE)
    def bindings(self):return [dict(native_ir=[('R',0,'1/9',1)],event={'proposal':'1/6'}),dict(native_ir=[('R',0,'1/9',-1)],event={'proposal':'1/3'})]
    def test_integer_aggregation_against_direct(self):
        bs=self.bindings();v=aggregate_pairs(bs,CACHE)
        direct={k:F(0) for k in ('T','CX','1Q')}
        for a,b in product(bs,repeat=2):
            c=native_price(simplify(a['native_ir']+b['native_ir']),CACHE)
            for k in c:direct[k]+=F(a['event']['proposal'])*F(b['event']['proposal'])*c[k]
        self.assertEqual(direct,v['after_per_whole_trial']);self.assertEqual(v['positive_pairs'],4)
    def test_zero_fill_keeps_unconditional_mass(self):
        v=aggregate_pairs(self.bindings(),CACHE);self.assertEqual(v['acceptance_squared'],F(1,4))
        self.assertEqual(sum(v['depth_probability_unconditional'].values()),F(1,4))
    def test_input_not_reduced_rejected(self):
        with self.assertRaises(ArithmeticError):aggregate_pairs([dict(native_ir=[('H',0),('H',0)],event={'proposal':'1'})],CACHE)
    def test_pair_cap(self):
        with self.assertRaises(RuntimeError):aggregate_pairs(self.bindings(),CACHE,max_pairs=3)
    def test_denominator_guard(self):
        with self.assertRaises(RuntimeError):aggregate_pairs(self.bindings(),CACHE,denominator_bits=2)
    def test_midstream_guard_failure(self):
        with self.assertRaises(RuntimeError):aggregate_pairs(self.bindings(),CACHE,check=lambda:(_ for _ in ()).throw(RuntimeError('fake cap')))


class Protection(unittest.TestCase):
    def auth(self):return dict(status='APPROVED_FOR_ONE_D2_BOUNDARY_RUN',science_execution_authorized=True,source_commit='a'*40,contract_sha256='h',runs=1,retries=0,mandatory_STOP=True,explicit_execution_instruction='off-domain mock instruction')
    def bind(self,a=None,**kw):
        v=dict(contract_sha='h',source='a'*40,head='b'*40,parents=['a'*40],changed=['auth'],dirty=False,allowed=['auth','receipt']);v.update(kw)
        return validate_binding(a or self.auth(),**v)
    def test_direct_child_valid(self):self.bind()
    def test_pending_rejected(self):
        with self.assertRaises(PermissionError):self.bind(dict(self.auth(),science_execution_authorized=False))
    def test_wrong_contract(self):
        with self.assertRaises(PermissionError):self.bind(contract_sha='wrong')
    def test_wrong_parent(self):
        with self.assertRaises(PermissionError):self.bind(parents=['c'*40])
    def test_source_change_child(self):
        with self.assertRaises(PermissionError):self.bind(changed=['auth','code'])
    def test_dirty(self):
        with self.assertRaises(PermissionError):self.bind(dirty=True)
    def test_auth_file_must_change(self):
        with self.assertRaises(PermissionError):self.bind(changed=['receipt'])
    def test_no_retry(self):
        with self.assertRaises(PermissionError):self.bind(dict(self.auth(),retries=1))
    def test_pending_before_git_or_registered_IO(self):
        with patch('trottertracks.algorithm_codesign.rte_development.boundary_launch.git',side_effect=AssertionError('must reject before git')):
            with self.assertRaises(PermissionError):verify_launch(ROOT,CP,'a'*40)
    def test_marker_exclusive_and_old_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'old_marker').write_text('retain')
            consume_marker(p/'new',{'mock':True})
            with self.assertRaises(FileExistsError):consume_marker(p/'new',{'mock':True})
            self.assertEqual((p/'old_marker').read_text(),'retain')
    def test_saved_namespace_reuse(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'new';d.mkdir();(d/'result').write_text('keep')
            with self.assertRaises(FileExistsError):consume_marker(d,{'mock':True})
    def test_UTF8_chunk_reconstruction(self):
        with tempfile.TemporaryDirectory() as t:
            v=EvidenceWriter(t,100000,chunk_bytes=51).write('a.json',{'é🙂':'測試'*200})
            self.assertTrue(verify_chunks(t,v))
    def test_output_cap_failure_keeps_partial(self):
        with tempfile.TemporaryDirectory() as t:
            w=EvidenceWriter(t,32770)
            with self.assertRaises(RuntimeError):w.write('a.json',{'a':'z'*100})
            self.assertFalse((Path(t)/'a.json').exists());self.assertTrue((Path(t)/'a.json.partial').exists())
    def test_guard_RSS_failure(self):
        c=json.loads(CP.read_text())
        with patch.object(Guard,'process_stats',return_value=(0,513*1048576)):
            with self.assertRaises(RuntimeError):Guard(c['caps']).check()
    def test_guard_CPU_failure(self):
        c=json.loads(CP.read_text())
        with patch.object(Guard,'process_stats',return_value=(901,1)):
            with self.assertRaises(RuntimeError):Guard(c['caps']).check()
    def test_no_registered_constructor_or_compiler_imports(self):
        for rel in ('src/trottertracks/algorithm_codesign/rte_development/boundary.py','scripts/tracks/algorithm_codesign/run_d2_saved_boundary_one_shot.py'):
            s=(ROOT/rel).read_text()
            for forbidden in ('import numpy','native_ir(', 'make_generator(', 'transpile(', 'synthesize(', 'Q_matrices(', '.sample('):self.assertNotIn(forbidden,s)


class RunnerMock(unittest.TestCase):
    def run_mock(self,fail=False):
        spec=importlib.util.spec_from_file_location('d2_runner_mock',ROOT/'scripts/tracks/algorithm_codesign/run_d2_saved_boundary_one_shot.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        from trottertracks.algorithm_codesign.rte_development.evidence import digest
        from trottertracks.algorithm_codesign.rte_development import boundary
        import sys
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);c=json.loads(CP.read_text());m.ROOT=p;m.CONTRACT=p/'contract.json'
            c['result_directory']='new';c['saved_result_directory']='saved'
            c['runtime_executable_sha256']=digest(Path(sys.executable).resolve())
            c['caps']['positive_pair_evaluations']=6
            for s in c['registered_rows']:s['positive_pair_bound']=1
            (p/'saved/events').mkdir(parents=True);(p/'saved/rows').mkdir()
            (p/'contract.json').write_text(json.dumps(c))
            (p/'auth.json').write_text('{}');c['authorization_path']='auth.json'
            (p/'manifest.json').write_text('{}');c['source_manifest']='manifest.json'
            (p/'saved/synthesis_cache.json').write_text('{}')
            for s in c['registered_rows']:
                ep=p/f'saved/events/event_{s["row_index"]}.json'
                ep.write_text(json.dumps([dict(native_ir=[['Z',0]],event={'proposal':'1/2'})]))
                v=dict(steps=2,degree=5,arm=s['arm'],events=dict(path=str(ep.relative_to(p/'saved')),sha256=digest(ep)),
                       per_whole_trial_native={'T':'0','CX':'0','1Q':'1/2'},
                       reference_law_diagnostic_only={'acceptance':'1/2'},budget={'N_per_axis':7},
                       expected_native_intercept={'T':'0','CX':'0','1Q':'7'},prep_readout_slope='7/2',
                       accepted_tail_native={},hard_native={},workspace_beyond_system=1)
                (p/'saved'/s['row_path']).write_text(json.dumps(v))
            actual=boundary.aggregate_pairs;calls=[0]
            def limited(*a,**k):
                calls[0]+=1
                if fail and calls[0]==2:raise RuntimeError('off-domain synthetic interruption')
                return actual(*a,**k)
            with patch.object(m,'verify_launch',return_value=(c,{},'b'*40)),patch.object(m,'check_protected',return_value=True),patch('trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch.verify_runtime',return_value={'mock':True}),patch.object(boundary,'aggregate_pairs',side_effect=limited):
                code=m.execute('a'*40)
            terminal=json.loads((p/'new/terminal.json').read_text())
            self.assertTrue((p/'new/one_shot_consumed.json').exists());self.assertTrue((p/'new/STOP.json').exists())
            self.assertTrue(terminal['mandatory_STOP']);self.assertEqual(terminal['retries'],0)
            return code,terminal,calls[0]
    def test_synthetic_full_run_only(self):
        code,v,calls=self.run_mock();self.assertEqual(code,0);self.assertEqual(calls,6)
        self.assertEqual(v['status'],'D2_BOUNDARY_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW')
    def test_failure_prefix_cannot_classify(self):
        code,v,calls=self.run_mock(fail=True);self.assertEqual(code,1);self.assertEqual(calls,2)
        self.assertEqual(v['status'],'D2_TECHNICAL_INCONCLUSIVE');self.assertFalse(v['prefix_scientific_use'])


if __name__=='__main__':unittest.main()
