"""Check only the supplied two-qubit legacy exports; no real H4 data loader."""
from fractions import Fraction as Q
import importlib.util
import json
from pathlib import Path
import sys
from flint import acb,ctx

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'tests'))
from test_h4_ball_evidence import MemoryBudget,ReferenceContraction,e

def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n')

def main():
    records=[];oracle=ReferenceContraction()
    for name in ('nonvacuum','complex'):
        directory=HERE/('artificial_legacy_'+name)
        for rep in ('pauli_jw','df_native'):
            model=json.loads((directory/(rep+'.json')).read_text())
            assert model['artificial_only'] is True and model['n_qubits']==2
            with ctx.workprec(256):
                result=e.EnclosureEngine(model,MemoryBudget()).tier(96)
                if rep=='pauli_jw':expected=oracle.pauli_reference(model['state'])
                else:
                    expected=sum((abs(e.complex_ball(z))**2*e.phase(-e.real(energy)*e.real(.8))
                                  for z,energy in zip(model['state'],(Q(1,8),Q(1,2),Q(1,2),Q(7,8)),strict=True)),acb(0))
                oracle.assert_encloses(result['reference'],expected)
            assert result['status']=='TARGET_MET' and len(result['axes'])==20
            assert not result['formal_certificate_accepted']
            assert all(a['old_N'] is None and a['old_G'] is None for a in result['axes'])
            save(directory/(rep+'_artificial_tier.json'),result)
            records.append(dict(state=name,representation=rep,status='PASS',axes=20,
                                reference_checked_against_analytic_scalar_formula=True))
    record=dict(status='PASS_ARTIFICIAL_LEGACY_PRODUCTION_CONNECTION',models=records,
                checks=len(records),artificial_only=True,real_H4_calls=0,
                chemistry_sampling_compilation_calls=0)
    save(HERE/'legacy_connection.json',record);print(json.dumps(record))

if __name__=='__main__':main()
