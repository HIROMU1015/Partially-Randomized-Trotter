"""One isolated ARTIFICIAL symbolic readout; no H4 or native price query."""
import hashlib,json
from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from trottertracks.track_c_compiler_a import compile_reference,PROFILE
EVIDENCE={}


class SymbolicReadoutTest(unittest.TestCase):
    def test_unbound_axis_parameter_stays_one_signed_RZ(self):
        phi=Parameter('q3b_artificial_feedback')
        q=QuantumCircuit(1,1);q.p(-phi,0);q.h(0);q.measure(0,0)
        native=compile_reference(q)
        dependent=[i for i in native.data if any(phi in getattr(p,'parameters',()) for p in i.operation.params)]
        self.assertEqual(len(dependent),1)
        self.assertEqual(dependent[0].operation.name,'rz')
        self.assertEqual(dependent[0].operation.params[0],-phi)
        self.assertEqual(native.global_phase.gradient(phi),-.5)
        self.assertEqual(native.data[-1].operation.name,'measure')
        EVIDENCE.update(status='ARTIFICIAL_SYMBOLIC_LOCAL_PASS',profile=PROFILE,
            native_operations=[dict(name=i.operation.name,params=list(map(str,i.operation.params))) for i in native.data],
            common_global_phase=str(native.global_phase),parameter_unbound=True,
            same_signed_axis_parameter=True,artificial_compiles=1,scientific_compiles=0,
            scientific_native_price_acquisition=False,production_all_contexts_qualified=False,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SymbolicReadoutTest))
    if result.wasSuccessful():
        path=ROOT/'artifacts/pr_reference_feedback_q3b/2026-11-10/run01/POST_B2_TECHNICAL.json'
        assert not path.exists()
        path.write_text(json.dumps(EVIDENCE,indent=2,sort_keys=True)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
