"""Real ffsim prepare/paired-fit worker on invented H4-shaped arrays, nit=0."""
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import scipy.optimize
import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_pr_aware_df_pilot as m


@pytest.mark.parametrize('role',['baseline','proposed'])
def test_real_workers_mock_optimizer(tmp_path,role):
    rng=np.random.default_rng(3020);a=rng.normal(size=(8,4,4))*.01
    J=a+a.transpose(0,2,1);U=np.array([np.eye(4)]*8);U[0,0,0]=-1;U[1,:2,:2]=-np.eye(2)
    g=m.old.reconstruct(J,U)+1e-4*np.einsum('pq,rs->pqrs',np.eye(4),np.eye(4))
    np.savez(tmp_path/'input.npz',one_mo=np.eye(4),eri_chemist=g,nuclear_repulsion=np.array(.5),mo_coeff=np.eye(4))
    np.savez(tmp_path/'export.npz',diag_coulomb_mats=J,orbital_rotations=U)
    c=dict(initial_export='export.npz')
    blocked=AssertionError('NO_OPTIMIZER_DURING_PREPARE')
    with patch.object(scipy.optimize,'minimize',side_effect=blocked):m.prepare_worker(tmp_path,tmp_path/'prepare',c,tmp_path/'input.npz')
    m.io.save(tmp_path/'initial_gate.json',dict(status='PASS'))
    def minimize(fun,x,**kwargs):
        value,grad=fun(x)
        return scipy.optimize.OptimizeResult(x=x.copy(),jac=grad,fun=value,
            success=True,status=0,message='ARTIFICIAL_ZERO_ITERATIONS',nit=0,nfev=1,njev=1)
    with patch.object(scipy.optimize,'minimize',side_effect=minimize) as mock:
        m.fit_worker(tmp_path,tmp_path/role,c,role)
    mock.assert_called_once()
    initial=m.io.read(tmp_path/role/'initial_diagnostic.json');final=m.io.read(tmp_path/role/'final_diagnostic.json')
    assert initial['combined_objective']==final['combined_objective']
    assert initial['combined_objective']==pytest.approx(1. if role=='baseline' else 1.1,abs=1e-8)
    assert m.io.read(tmp_path/role/'optimizer.json')['nit']==0
    assert m.io.read(tmp_path/role/'STOP.json')['scientific_units']==5
    assert m.io.read(tmp_path/'prepare/STOP.json')['scientific_units']==6
    assert final['CA_achieved'] is final['total_cost'] is final['attenuation'] is None
