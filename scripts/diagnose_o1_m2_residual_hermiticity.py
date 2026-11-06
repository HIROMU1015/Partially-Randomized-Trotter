"""Same-input operator-only diagnosis; do not change the fixed mapper or tests.

Separates the saved residual's anti-Hermitian rounding part from the existing
real-Pauli projection. No new tolerance, repair, fit, state, circuit or dynamics.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def diagnose(core_path,input_path):
    import numpy as np
    from openfermion import InteractionOperator
    import validate_o1_m2_h4_operator as reference
    from trottertracks.o1_m2_residual_adapter import load_saved_pair
    from trottertracks.o1_m2_cutoff_free_pauli import map_integrals,_unit_jw
    from trottertracks.c1_stage1 import spin_lift_unthresholded,sector_indices
    pair=load_saved_pair(core_path,input_path)
    car=reference.CARReference(4).integrals(pair.residual)
    mapped=reference.pauli_matrix(map_integrals(pair.residual,model_id='diagnostic_R'))
    one,two=spin_lift_unthresholded(pair.residual.h,pair.residual.g)
    raw=InteractionOperator(0.,one,two/2)
    grouped={}
    for term in raw:
        coefficient=raw[term]
        if coefficient==0:
            continue
        for word,weight in _unit_jw(term):
            label=['I']*8
            for q,axis in word:
                label[7-q]=axis
            grouped.setdefault(''.join(label),[]).append(coefficient*weight)
    complex_coefficients={label:complex(math.fsum(complex(z).real for z in values),
                                      math.fsum(complex(z).imag for z in values))
                          for label,values in grouped.items()}
    packet=SimpleNamespace(num_qubits=8,constant=complex_coefficients.pop('I'*8,0.),
                           terms=tuple(complex_coefficients.items()))
    unprojected=reference.pauli_matrix(packet)
    hermitian=(car+car.conj().T)/2
    anti=(car-car.conj().T)/2
    sector=sector_indices(4,2,2)
    records={}
    for name,delta in (('real_mapper_minus_raw_CAR',mapped-car),
                       ('real_mapper_minus_Hermitian_part',mapped-hermitian),
                       ('complex_JW_minus_raw_CAR',unprojected-car),
                       ('raw_anti_Hermitian_part',anti)):
        records[name]={}
        for space,matrix in (('full256',delta),('sector36',delta[np.ix_(sector,sector)])):
            records[name][space]=dict(max_entry=float(np.max(abs(matrix))),
                                     frobenius=float(np.linalg.norm(matrix)))
    return dict(scope='SAME_H4_RESIDUAL_OPERATOR_DIAGNOSIS_ONLY',
        fixed_source_commit='cee130ad8c24868a88ad09778f664aa01ae50154',
        diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tolerance_changed=False,qualification_changed=False,mapper_changed=False,
        complex_Pauli_imaginary_coefficients=dict(
            count=sum(z.imag!=0 for z in complex_coefficients.values()),
            maximum=max(abs(z.imag) for z in complex_coefficients.values()),
            l1=math.fsum(abs(z.imag) for z in complex_coefficients.values())),
        input_tensor_asymmetries={role:dict(
            h_max=float(np.max(abs(p.h-p.h.T))),
            g_adjoint_max=float(np.max(abs(p.g-p.g.transpose(1,0,3,2)))))
            for role,p in (('physical',pair.physical),('core',pair.core),('residual',pair.residual))},
        records=records,
        interpretation='Numerical diagnosis only. Real-coefficient output removes the anti-Hermitian part of the supplied binary64 residual. This neither repairs the fixed input nor changes the original NOT_QUALIFIED result.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core',required=True,type=Path)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh diagnostic path required')
    result=diagnose(args.core,args.input)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')
