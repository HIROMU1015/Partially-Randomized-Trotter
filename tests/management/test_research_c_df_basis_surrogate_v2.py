"""Negative/signed-zero orthogonal inputs; no saved molecular arrays."""
from pathlib import Path
import sys,hashlib
from copy import deepcopy
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_df_basis_surrogate_v2 as v2
from trottertracks import c_df_basis_reanalysis_v2 as run


@pytest.mark.parametrize('case',['reflection','pi','rotation','random_SO','random_O'])
def test_negative_multiframe_basis_keys(case):
    from trottertracks import c_pr_aware_df_stage_b as b
    from trottertracks.c1_general_k_adapter import _matrix,_spin_lift
    rng=np.random.default_rng(29149);U=np.repeat(np.eye(4)[None],8,axis=0)
    for i in range(8):
        if case=='reflection':U[i]=np.diag([-1.,1.,1.,1.])
        elif case=='pi':U[i]=np.diag([-1.,-1.,1.,1.])
        elif case=='rotation':
            a=.13*(i+1);U[i,:2,:2]=[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]
        else:
            u=np.linalg.qr(rng.normal(size=(4,4)))[0]
            if case=='random_SO' and np.linalg.det(u)<0:u[:,0]*=-1
            U[i]=u
    J=np.repeat(np.diag([-1.,2.,3.,4.])[None],8,axis=0)
    arm=b.prepare_arm(np.eye(4),np.zeros((4,)*4),.31,J,U,role='proposed',source_identity='ARTIFICIAL',input_identity='ARTIFICIAL',ld=2)
    distances=v2.geometry(arm.plan.atoms,U)
    assert len(distances)==36
    for i in range(2,8):
        key=hashlib.sha256(repr(_spin_lift(_matrix(U[i],'invented'))).encode()).hexdigest()
        assert all(a.basis_key==key for a in arm.plan.atoms if a.frame_id.endswith(':F'+str(i)))
    from trottertracks import c_df_basis_surrogate as old
    with pytest.raises(ValueError,match='EXACT_SAVED_BASIS_KEY'):old.geometry(arm.plan.atoms,U)
    U[2,:,0]*=-1
    with pytest.raises(ValueError,match='EXACT_SAVED_BASIS_KEY'):v2.geometry(arm.plan.atoms,U)


def test_existing_engine_bindings_restored_and_statistics_unchanged():
    from trottertracks import c_df_basis_reanalysis as old,c_df_basis_surrogate as f
    before=old.features,old.SCOPE,old.FRESH
    with pytest.raises(RuntimeError):
        with run.bindings():
            assert old.features is v2 and old.FRESH.endswith('run02')
            assert v2.describe_models is f.describe_models and v2.trajectory_features is f.trajectory_features
            raise RuntimeError('ARTIFICIAL')
    assert (old.features,old.SCOPE,old.FRESH)==before


def test_repair_does_not_change_registered_science(tmp_path):
    from trottertracks import c_df_basis_reanalysis as old
    c={k:{} for k in ('cohorts','models','descriptive_plan','features','numerical_checks','call_limits','packages')}
    c.update(folds='fixed',DF_clusters=2,samples_per_cohort=32,event_count=128,maximum_logical_units=148)
    old.save(tmp_path/'old.json',c)
    c=dict(c,implementation=v2.VERSION,technical_relaunch=1,previous_contract='old.json',previous_contract_sha256=old.sha(tmp_path/'old.json'))
    old.save(tmp_path/'new.json',c);assert run.check_revision(tmp_path,tmp_path/'new.json')==c
    c['models']={'changed':['x']};(tmp_path/'bad.json').write_text(__import__('json').dumps(c))
    with pytest.raises(ValueError,match='CHANGED_SCIENCE:models'):run.check_revision(tmp_path,tmp_path/'bad.json')
