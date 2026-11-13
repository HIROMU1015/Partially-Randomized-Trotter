"""Artificial finite alphabets/replica descriptor only, no molecular inputs."""
from fractions import Fraction
from dataclasses import dataclass
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from trottertracks import h4_structural_kernel as k
from trottertracks.h4_df10_price_v1 import Accounting
from trottertracks import h4_pr_physical_closure as c


@dataclass(frozen=True)
class ArtificialBasisOp:
    name: str
    qubits: tuple
    parameters: tuple
    matrix_sha256: object=None


def artificial_model():
    def ops(angle):
        return [dict(name='xx_plus_yy',qubits=[q,q+1],parameters=[str(angle),'0.0']) for q in range(0,8,2)]
    registry=[ops(.1+.03*b) for b in range(9)]
    supports=[(q,) for q in range(8)]+list(itertools.combinations(range(8),2))
    components=[];old=[]
    for b in range(9):
        for j,s in enumerate(supports):
            comp=NS(component_id='ARTIFICIAL_'+str(b)+'_'+str(j),basis_id='basis'+str(b),basis_hash='hash'+str(b),
                probability=1/324,coefficient_abs=1.,coefficient_sign=(-1)**j,diagonal_pauli_support=s,
                basis_change_operations=tuple(ArtificialBasisOp(name=o['name'],qubits=tuple(o['qubits']),parameters=tuple(o['parameters'])) for o in registry[b]))
            components.append(comp);old.append(dict(id=comp.component_id,coefficient_abs=1.,probability_hex=comp.probability.hex(),support=list(s),sign=comp.coefficient_sign))
    prep=NS(preparation_hash='ARTIFICIAL',rte_preparation=NS(symbolic_tail=NS(components=components)))
    diag0=NS(rzz=[],rz=[(q,.02+.01*q) for q in range(8)],global_phase=.17)
    diag1=NS(rzz=[(q,q+1,.12) for q in range(0,8,2)],rz=[],global_phase=.04)
    context=NS(Accounting=Accounting,blocks=[[dict(basis=ops(.43),diagonal=diag0),dict(basis=ops(.71),diagonal=diag1)]],scalar=.3,ctx=NS(prep=prep))
    saved=dict(preparation=dict(hash='ARTIFICIAL',components=old,
        basis_registry=[dict(id='basis'+str(b),hash='hash'+str(b),operations=registry[b]) for b in range(9)]),
        laws=[dict(R_outer=R,q_PF=q,m=m,candidate_id='ARTIFICIAL_'+str(m),rotation_angles={'0':.13/R,'2':.05/R},
            order_probabilities=[.9,.1],order_weights=[9.,1.]) for R,q,m in ((2,1,0),(128,64,6))])
    return k.Model(ROOT,context,saved)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.model=artificial_model()

    def test_interval_kernel_contains_exact_correlated_enumeration(self):
        states=(0,1)
        def step(s,j):return j,int(s!=j)
        band=k.discrete(states,[.1,.2,.7],[1.,2.,7.],lambda s,j:step(s,int(j==2)))
        terminal=np.array([0.,1.]);value,lo,hi=band.repeat(4).expected(0,terminal)
        exact=Fraction()
        for path in itertools.product((0,1),repeat=4):
            state=0;cost=0;weight=Fraction(1)
            for end in path:weight*=Fraction(7 if end else 3,10);cost+=int(state!=end);state=end
            exact+=weight*(cost+state)
        self.assertLessEqual(Fraction.from_float(lo),exact);self.assertGreaterEqual(Fraction.from_float(hi),exact)
        self.assertLessEqual(lo,value);self.assertLessEqual(value,hi)

    def test_nominal_probabilities_are_not_repaired(self):
        probs=[.1,.2,.7]
        band=k.discrete((0,1),probs,[1.,2.,7.],lambda s,j:(int(j==2),j))
        self.assertEqual(band.central.transition[0,0],probs[0]+probs[1])
        self.assertEqual(band.central.reward[0],sum(p*j for j,p in enumerate(probs)))

    def test_phase_and_basis_ast_parts_reuse_original_source(self):
        step,close=k.source_tail_functions(ROOT);T=c.tracer(Accounting);t=T()
        ops=[dict(name='p',qubits=[0],parameters=['.2'])]
        app=dict(application_index=0,is_identity=False,coefficient_sign=-1,role='product',basis_id='ART',basis_hash='ART',basis_change_operations=ops,diagonal_pauli_support=[0])
        active,phase=step(t,None,0.,app,0,0.)
        self.assertEqual(phase,math.pi);close(t,active,phase)
        self.assertIsNone(t.pending_basis)

    def test_UNKNOWN_K0_cannot_call_K1_or_write_prices(self):
        fake=Mock(return_value=(None,None,None,dict(status='UNKNOWN',K1_started=False)));price=Mock()
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'fresh';k.execute_batch(ROOT,'ARTIFICIAL',out,k0=fake,k1=price)
            price.assert_not_called();self.assertFalse((out/'K1.json').exists())

    def test_FAIL_is_saved_and_K1_not_called(self):
        fake=Mock(side_effect=ValueError('EXACT_LOCAL_REWARD_OR_Z_PARITY'));price=Mock()
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'fresh'
            with self.assertRaises(ValueError):k.execute_batch(ROOT,'ARTIFICIAL',out,k0=fake,k1=price)
            self.assertEqual(json.loads((out/'K0.json').read_text())['status'],'FAIL');price.assert_not_called()

    def test_artificial_descriptor_complete_K0_K1_path(self):
        model=self.model;results={}
        for q in range(9):
            wire=k.Wire(model,q);results[q]=[wire.build(law) for law in model.laws]
        values=k.run_K1(results,model.laws)
        self.assertEqual(len(values),4)
        for v in values:
            self.assertEqual(v['quantity'],'FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT')
            self.assertIsNone(v['mean_compiled_C']);self.assertFalse(v['compiled_price_qualified'])
            self.assertLessEqual(v['numerical_enclosure'][0],v['mean_S']);self.assertLessEqual(v['mean_S'],v['numerical_enclosure'][1])

    def test_original_rte_metadata_on_artificial_components_no_sampling(self):
        source='c31ef424175c02379eafebe8fc6c7548c1d9aabe'
        rte=types.ModuleType('_artificial_original_rte');sys.modules[rte.__name__]=rte
        exec(compile(c.blob(ROOT,source,'src/trotterlib/rte.py'),source+':rte.py','exec'),rte.__dict__)
        original=[]
        for comp in self.model.components:
            original.append(rte.RTEComponent(component_id=comp.component_id,probability=comp.probability,
                coefficient_abs=comp.coefficient_abs,coefficient_sign=comp.coefficient_sign,is_identity=False,
                basis_id=comp.basis_id,basis_hash=comp.basis_hash,diagonal_pauli_support=comp.diagonal_pauli_support,
                basis_change_operations=tuple(rte.BasisChangeOperation(**asdict(op)) for op in comp.basis_change_operations)))
        laws=[]
        for R in (2,128):
            tau=.23/R;dist=rte.finite_rte_distribution(tau,2)
            laws.append(dict(tau=tau,order_weights=dist.unnormalized_order_weights,order_probabilities=dist.order_probabilities,
                event_normalization=dist.exact_finite_distribution,rotation_angles={str(o):math.atan(tau/(o+1)) for o in (0,2)}))
        model=NS(components=original,nominal=self.model.nominal,laws=laws,registry=self.model.registry,components_meta=self.model.components_meta)
        result=k.law_metadata_parity(NS(rte=rte),model)
        self.assertEqual(result['metadata_records_checked'],3240);self.assertFalse(result['trajectory_sampling'])


if __name__=='__main__':unittest.main(verbosity=2)
