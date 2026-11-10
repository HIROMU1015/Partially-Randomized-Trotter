"""Unbound artificial templates: no H4 inputs, numeric angles or grid search.

Before execution fix exactly Bprime/Reference x product absent/present, their
registered post-unroll gate sets, and one free parameter per gate argument.
Inspect before final cancellation to qualify SX-only size accounting. Four
artificial compiles, no operator/simulation/scientific circuit or pricing call.
"""
import json
from pathlib import Path
import unittest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import XXPlusYYGate
from trottertracks.track_c_compiler_a import compile_reference, PROFILE


class SymbolicContext(unittest.TestCase):
    def test_all_four_Q_contexts_before_X_cancellation(self):
        results=[]
        for arm in ('Bprime','Reference_B1'):
            for product in (False,True):
                q=QuantumCircuit(3)
                names=['h','p','rz','rzz','xx_plus_yy']
                if arm=='Bprime':names+=['cx','crz']+(['cz'] if product else [])
                else:names+=['cswap']+(['z'] if product else [])
                for name in names:
                    theta=Parameter('theta_'+name)
                    if name=='p':q.p(-theta,0)
                    elif name=='rz':q.rz(theta,0)
                    elif name=='rzz':q.rzz(theta,0,1)
                    elif name=='crz':q.crz(theta,0,1)
                    elif name=='xx_plus_yy':q.append(XXPlusYYGate(theta,Parameter('beta')),[0,1])
                    elif name=='cswap':q.cswap(0,1,2)
                    elif name in ('cx','cz'):getattr(q,name)(0,1)
                    else:getattr(q,name)(0)
                    q.barrier()
                trace={}
                def cb(**kw):
                    name=kw['pass_'].name()
                    if name=='UnrollCustomDefinitions':trace['post_unroll']=sorted(set(kw['dag'].count_ops())-{'barrier'})
                    if name=='BasisTranslator':trace['before_last_ops']={k:int(v) for k,v in kw['dag'].count_ops().items()}
                native=compile_reference(q,callback=cb)
                self.assertEqual(set(trace['post_unroll']),set(names))
                self.assertEqual(trace['before_last_ops'].get('x',0),0)
                self.assertEqual(native.count_ops().get('x',0),0)
                self.assertEqual(set(native.parameters),set(q.parameters))
                results.append(dict(arm=arm,product=product,all_parameters_unbound=True,**trace))
        root=Path(__file__).resolve().parents[1]
        out=root/'artifacts/pr_reference_feedback_q3cd/2026-11-10/run01/SYMBOLIC_CONTEXT.json'
        with out.open('x') as stream:
            json.dump(dict(status='ARTIFICIAL_SYMBOLIC_TEMPLATE_PASS',rows=results,profile=PROFILE,
                artificial_compiles=4,scientific_compiles=0,
                proof_scope='global family maps contain no X before cancellation; numeric binding does not change gate names'),stream,indent=2)
            stream.write('\n')


if __name__=='__main__':unittest.main()
