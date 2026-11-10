"""Evaluate review (3.6), not new raw signals or a different PF/law."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def evaluate(private_dir, operands_path):
    import numpy as np
    from scipy.linalg import eigh, eig, expm, solve
    from trotterlib import b2_p1_signal_validation as p1
    from trotterlib.calpha_macrostep_bounds import macrostep_diagnostics

    prior_path = ROOT/'artifacts/calpha_h4_step1/2026-11-10/run02_sector_corrected/result.json'
    prior = json.loads(prior_path.read_text())
    if hashlib.sha256(prior_path.read_bytes()).hexdigest() != '8ff3e10d6fb663714b68aa0e9b2affa0816bc04d92c509eede7dca8791fbfffb':
        raise ValueError('RUN02_CHANGED')
    data = p1.load_private_inputs(private_dir)
    ix = np.ix_(data['indices'],data['indices'])
    hermitian = lambda x: (x+x.conj().T)/2
    old = prior['operator_quantities']
    heads = np.array([hermitian(x)[ix] for x in [data['one_body'],*data['fragments'][:3]]])
    tail = hermitian(sum(data['fragments'][3:],np.zeros((256,256),complex))-old['identity']*np.eye(256))[ix]
    vals, vecs = eigh(tail)
    theta, delta = .005, .02
    rt = (vecs*(np.arctan(theta*vals)/theta))@vecs.conj().T
    rates = (math.log1p(theta**2*old['lambda_R']**2)-np.log1p(theta**2*vals**2))/(2*theta)
    damping = (vecs*rates)@vecs.conj().T
    _, f, generator = macrostep_diagnostics(heads,rt,damping,delta)
    exact = expm(delta*generator)
    defect = f-exact
    values, basis = eig(generator)
    coefficients = solve(basis,data['sector_state'])
    weights = (defect@basis)*coefficients
    projection_residual = float(np.linalg.norm(basis@coefficients-data['sector_state']))
    eigen_residual = float(np.linalg.norm(generator@basis-basis*values,2))
    rows = []
    for m in (0,17):
        q = 1<<m
        times = delta*np.arange(q)
        # All columns are defect vectors, never cosine/sine/RPE signals.
        propagation = np.exp(values[:,None]*times[None,:])
        vector_sum = float(np.sum(np.linalg.norm(weights@propagation,axis=0)))
        geometric = np.sum(np.exp(values.real[:,None]*times[None,:]),axis=1)
        triangle = float(np.dot(np.linalg.norm(weights,axis=0),geometric))
        rows.append(dict(m=m,q=q,review_3_6_state_defect_sum=vector_sum,
                         eigenmode_triangle_diagnostic=triangle))
    # Private operands are solely for numerical enclosure, not public evidence.
    np.savez(operands_path, heads=heads,tail=tail,h_reference=hermitian(data['physical'])[ix],
             psi=data['sector_state'],energy=np.array(data['energy']),
             eigenvalues=values,eigenvectors=basis,coefficients=coefficients,
             F=f,E=exact,G=generator,Rtheta=rt,Gamma=damping,
             lambda_R=np.array(old['lambda_R']))
    return dict(rounds=rows, eigenbasis_condition=float(np.linalg.cond(basis)),
        eigen_residual_norm=eigen_residual,projection_residual_norm=projection_residual,
        operands_SHA256=hashlib.sha256(Path(operands_path).read_bytes()).hexdigest(),
        qualification='binary64_diagnostic_of_review_3_6_NOT_CERTIFIED',
        new_raw_signals=0,new_damped_signals=0,ground_state_solves=0,
        private_operands_not_public=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-dir',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--operands',required=True)
    a = p.parse_args()
    output = Path(a.output)
    output.mkdir(parents=True,exist_ok=False)
    if Path(a.operands).exists():
        raise ValueError('NEW_PRIVATE_OPERAND_PATH_REQUIRED')
    started=time.monotonic()
    source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result=dict(started_UTC=datetime.now(timezone.utc).isoformat(),SOURCE_SHA256=source,
                review_equation='3.6',status='STARTED')
    try:
        result.update(evaluate(a.private_dir,a.operands))
        result['status']='STATE_DEFECT_DIAGNOSTIC_COMPLETE'
    except Exception as exc:
        result.update(status='TECHNICAL_EXCEPTION',reason=str(exc))
        raise
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        result['SOURCE_unchanged']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==source
        (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(result['status'])
