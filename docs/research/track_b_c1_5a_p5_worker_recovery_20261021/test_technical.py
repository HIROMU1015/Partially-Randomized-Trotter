"""P5R artificial tests. Never opens/instantiates H4 p/model/basis."""
from fractions import Fraction as F
import ast
import inspect
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from common import HERE,ROOT,OLD,read,require,new_marker,digest
from supervisor import supervise
from bridge import connect
d=connect()
from selector import Prefix,IndexedP5
from trottertracks.algorithm_codesign.g9_p5 import P5Closed
from trottertracks.algorithm_codesign.return_aggregation import dyadic_index

class Fixed:
    def __init__(self,values):self.values=iter(values);self.draws=[]
    def bits(self,n):
        v=next(self.values,0);self.draws.append((n,v));return v

class Tests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def task(self,mode,identifier):
        return dict(id=identifier,cmd=[sys.executable,'-B',str(HERE/'synthetic_worker.py'),mode,identifier,str(self.path)],env=os.environ)

    def test_01_multi_exec_start_exit_unique_ids(self):
        out=supervise([self.task('normal',str(i)) for i in range(4)],self.path/'supervisor',4)
        self.assertTrue(out['complete']);self.assertEqual(len(set(x['worker_pid'] for x in out['exits'])),4)
        self.assertEqual(sorted(out['completed']),['0','1','2','3'])
        self.assertTrue(all(x['exit_code']==0 and x['signal'] is None for x in out['exits']))
    def test_02_exit_code7_preserves_partial_and_fails_closed(self):
        out=supervise([self.task('exit7','bad'),self.task('normal','later')],self.path/'sup',1)
        self.assertFalse(out['complete']);self.assertEqual(out['first_failure']['exit_code'],7)
        self.assertTrue((self.path/'bad.partial.jsonl').exists())
        self.assertFalse((self.path/'later.complete.json').exists())
        self.assertIsNone(out['first_failure']['signal'])
    def test_03_signal_is_recorded(self):
        out=supervise([self.task('signal','sig')],self.path/'sup',1)
        self.assertEqual(out['first_failure']['signal'],15);self.assertEqual(out['first_failure']['exit_code'],-15)
    def test_04_marker_rejects_second_run_after_failure(self):
        tasks=[self.task('exit7','bad')];supervise(tasks,self.path/'sup',1)
        with self.assertRaises(FileExistsError):supervise(tasks,self.path/'sup',1)
    def test_05_prefix_all_bit_values_and_zero_mass_nodes(self):
        law=(F(1,4),F(0),F(1,8),F(5,8));selector=Prefix(law,12)
        for b in range(1<<12):self.assertEqual(selector.index(b),dyadic_index(law,12,b))
        for value in (-1,1<<12,True):
            with self.assertRaises(ValueError):selector.index(value)
    def test_06_sample_body_only_changes_draw(self):
        def body(fn):
            node=ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
            node.body=node.body[1:]
            return ast.dump(node,include_attributes=False)
        self.assertEqual(body(P5Closed.sample),body(IndexedP5.sample))
    def test_07_P5_event_proposal_weight_exact_same(self):
        for p in ((F(1,3),F(2,3)),(F(1,5),F(3,10),F(1,2))):
            old=P5Closed(p,F(1,2));new=IndexedP5.attach(old)
            for trial in range(64):
                a,b=d.Bits('closed_P5_full',trial,2),d.Bits('closed_P5_full',trial,2)
                ea,eb=old.sample(a),new.sample(b)
                self.assertEqual(ea,eb);self.assertEqual(a.draws,b.draws)
                self.assertEqual(eb['proposal']*eb['weight'],eb['coefficient'])
    def test_08_group_boundaries_all_modes(self):
        g=P5Closed((F(1,5),F(3,10),F(1,2)),F(1,2));new=IndexedP5.attach(g);start=0
        for q in g.group_law:
            end=start+int(q*(1<<g.H))
            for bit in (start,end-1):
                a,b=Fixed([bit]),Fixed([bit]);self.assertEqual(g.sample(a),new.sample(b));self.assertEqual(a.draws,b.draws)
            start=end
    def test_09_frozen_bitstream_under_different_schedule(self):
        ids=[(t,k) for t in range(8) for k in range(4)]
        def draw(t,k):
            bits=d.Bits('closed_P5_full',t,k);return [bits.bits(2048) for _ in range(4)]
        expected={(t,k):draw(t,k) for t,k in ids}
        self.assertEqual(expected,{(t,k):draw(t,k) for t,k in reversed(ids)})
    def test_10_complete_plan_zero_and_failure_distinct(self):
        from summarize import summarize
        with self.assertRaises(FileNotFoundError):summarize()
        out=supervise([self.task('exception','bad')],self.path/'sup',1)
        self.assertFalse(out['complete']);self.assertNotEqual(out['first_failure']['exit_code'],0)
        self.assertEqual(out['completed'],[])
    def test_11_old_source_result_marker_unmodified(self):
        import subprocess,hashlib
        base='b022f2ea421649015a4c21ea3c904fe619d0c198'
        for name in ('run.py','diagnostic.py','authorization_receipt.json','attempt_marker.json',
                     'result/closed_P5_full/terminal.json','result/ordinary/summary.json','result/full_return/summary.json'):
            path=OLD/name;expected=subprocess.check_output(['git','show',base+':'+str(path.relative_to(ROOT))],cwd=ROOT)
            self.assertEqual(path.read_bytes(),expected)
    def test_12_no_worker_pool_or_sampler_transfer(self):
        root=(HERE/'run.py').read_text();job=(HERE/'job.py').read_text()
        self.assertNotIn('ProcessPoolExecutor',root);self.assertNotIn('mp.get_context',root)
        self.assertIn("sampler_instances=1",job)
        provider_body=ast.get_source_segment(job,next(n for n in ast.parse(job).body if isinstance(n,ast.FunctionDef) and n.name=='provider'))
        self.assertNotIn('P5Closed(',provider_body);self.assertNotIn('Sampler(',provider_body)

    def test_13_split_Evidence_operator_exact_old_glue(self):
        import numpy as np
        import math
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import RYGate
        from qiskit.quantum_info import Operator
        from trotterlib.df_rte_tail import DFBasisRegistry
        from support import prior_adapter
        from job import path_record,decode_path
        reg=DFBasisRegistry();labels=[]
        for i in range(3):
            entry=reg.register(((RYGate(.1*(i+1)),(i%2,)),),num_system_qubits=2)
            labels.append(dict(label=i,component_id='artificial:'+str(i),sign=(-1 if i==1 else 1),
                support=[i%2],basis_id=entry.basis_id,basis_hash=entry.basis_hash))
        adapter=prior_adapter().DFAdapter(labels,(F(1,5),F(3,10),F(1,2)),reg,2,'ARTIFICIAL')
        model=dict(adapter=adapter,p=adapter.p,x=F(1,2),scalar=F(-3,11),labels_id=adapter.labels_id)
        precision=dict(root_bits=256,probability_bits=160,eta=F(1,10**12),rho=F(1,10**12),
            angle_tau=F(1,10**12),angle_root_bits=256,angle_terms=64)
        old=d.Sampler('closed_P5_full',model,'ARTIFICIAL',self.path,precision=precision)
        new=d.Sampler('closed_P5_full',model,'ARTIFICIAL',self.path,precision=precision)
        new.g=IndexedP5.attach(new.g)
        for trial in range(8):
            expected=d.generate_path(old,trial,d.Timers())
            actual=path_record(new,trial,d.Timers(),'ARTIFICIAL')
            self.assertEqual(expected['exact'],actual['exact'])
            record,events,angles=decode_path(self.path,actual,model,{}, {})
            circuit=d.build_tail(adapter,model['scalar'],events,angles)
            self.assertEqual(d.logical_resources(circuit,events),expected['logical'])
            qs=[]
            for label in adapter.rows:
                basis=QuantumCircuit(2)
                for gate,qubits in reg.definition(label['basis_id']).runtime_operations:basis.append(gate,qubits)
                b=Operator(basis).data
                diagonal=np.diag([label['sign']*(-1)**sum((k>>i)&1 for i in label['support']) for k in range(4)])
                qs.append(b@diagonal@b.conj().T)
            unitary=np.eye(4,dtype=complex)
            for event in events:
                word=np.eye(4,dtype=complex)
                for label in event['raw_word']:word=word@qs[label]
                angle=math.atan(float(event['ratio']))
                step=(-1j)**len(event['raw_word'])*(math.cos(angle)*np.eye(4)-1j*math.sin(angle)*qs[event['child']])@word
                unitary=step@unitary
            unitary*=np.exp(-1j*float(F(4,5)*model['scalar']))
            controlled=np.eye(8,dtype=complex);controlled[4:,4:]=unitary
            self.assertLess(float(np.linalg.norm(Operator(circuit).data-controlled,'fro')),1e-10)

    def test_14_launch_failure_saved_and_no_retries(self):
        task=dict(id='missing',cmd=['/nonexistent/P5R-artificial-executable'])
        out=supervise([task],self.path/'sup',1)
        self.assertFalse(out['complete']);self.assertEqual(out['not_completed'],['missing'])
        self.assertIn('supervisor_exception',out['first_failure'])

if __name__=='__main__':unittest.main(verbosity=2)
