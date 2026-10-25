"""Small-system Arb oracle, raw-ray, measurement and matrix-free checks."""
import importlib,json,sys,types,unittest
from pathlib import Path
from fractions import Fraction as Q
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=types.ModuleType('hx_vector_test');p.__path__=[str(ROOT/'src/trotterlib')];sys.modules[p.__name__]=p
V=importlib.import_module(p.__name__+'.pauli_vector_enclosure')
M=importlib.import_module(p.__name__+'.h4_pauli_pr_mean')
P=importlib.import_module(p.__name__+'.h4_arb_pr0_preparation')
C=V.C
FIX=ROOT/'artifacts/pauli_h2_h6_comparison/2026-10-25/artificial/pr_fixtures.json'
class Checks(unittest.TestCase):
 def test_signed_permutations_equal_existing_C(self):
  import itertools
  for n in [2,3]:
   a=np.array([complex(j/16,(3-j)/32) for j in range(1<<n)])
   for label in map(''.join,itertools.product('IXYZ',repeat=n)):
    for sign in [-1,1]:
     got=V.permute(a,V.mapping(label,sign));old=P._pauli_action([C.acb(float(z.real),float(z.imag)) for z in a],P._pauli_map(label,sign),C)
     self.assertTrue(all(C.complex_ball([float(z.real),float(z.imag)])==w for z,w in zip(got,old)))
 def test_effective_mean_contains_existing_Arb_oracle(self):
  for f in json.loads(FIX.read_text()):
   law=f['law'];cfg=law['config'];raw=f['raw']
   model=dict(state=raw,n_qubits=law['n_system'],hamiltonian=dict(terms=[(x['label'],x['coefficient']) for x in law['retained']],constant=0.))
   ref=dict(f['reference'],method='ARTIFICIAL_REFERENCE')
   got=V.evaluate(model,ref,cfg,law);oracle=M.artificial_mean(law,raw,f['reference'],96)
   for key in ['real','imag']:
    a=got['physical_corrected_mean'][key];b=oracle['physical_corrected_mean'][key]
    self.assertLessEqual(Q(a['lower']),Q(b['lower']));self.assertGreaterEqual(Q(a['upper']),Q(b['upper']))
   self.assertLess(got['state_l2_error_upper'],1e-8)
 def test_raw_ray_and_B_once(self):
  f=json.loads(FIX.read_text())[1];law=f['law'];cfg=law['config'];raw=f['raw']
  model=dict(state=raw,n_qubits=2,hamiltonian=dict(terms=[(x['label'],x['coefficient']) for x in law['retained']],constant=0.))
  ref=dict(f['reference'],method='ARTIFICIAL_REFERENCE')
  a=V.evaluate(model,ref,cfg,law);model['state']=[[2*x,2*y] for x,y in raw];b=V.evaluate(model,ref,cfg,law)
  self.assertEqual(Q(b['raw_norm_exact']),4*Q(a['raw_norm_exact']))
  self.assertEqual(a['B_applied_count'],1)
  for key in ['real','imag']:
   self.assertLessEqual(max(Q(a['physical_corrected_mean'][key]['lower']),Q(b['physical_corrected_mean'][key]['lower'])),min(Q(a['physical_corrected_mean'][key]['upper']),Q(b['physical_corrected_mean'][key]['upper'])))
 def test_Duhamel_reference_no_exact_eigenstate_assumption(self):
  model=dict(state=[[1.,0.],[.125,0.],[0.,0.],[0.,0.]],energy=.3,n_qubits=2,Hamiltonian_hash='toy',hamiltonian=dict(constant=.2,terms=[('IZ',.1),('IX',.07)]))
  got=V.reference_enclosure(model)
  with C.ctx.workprec(192):
   h=C.identity(4)*C.real(.2)+C.pauli('IZ')*C.real(.1)+C.pauli('IX')*C.real(.07)
   a=C.acb_mat([[C.complex_ball(z)] for z in model['state']]);z=C.signal(a,(-h*C.acb(0,C.real(.8))).exp()*a)/C.signal(a,a).real
   for key,ball in [('real',z.real),('imag',z.imag)]:
    bounds=got['physical_reference'][key];self.assertLessEqual(Q(bounds['lower']),C.dyadic(ball.lower()));self.assertGreaterEqual(Q(bounds['upper']),C.dyadic(ball.upper()))
  self.assertFalse(got['exact_eigenstate_assumed'])
if __name__=='__main__':unittest.main()
