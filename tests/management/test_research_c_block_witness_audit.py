from copy import deepcopy
import pytest
from trottertracks import c_block_witness_audit as audit,c_cx_structural_blocks as b
from test_research_c_relative_pauli import g,r


def fixture():
    a=r(2,[g('z',[0]),g('cx',[0,1]),g('z',[0])]);c=r(2,[g('cx',[0,1])])
    saved,plan=b.make_plan(a,c);cert=b.evaluate(saved,plan,parallel=False)
    return saved,plan,cert,a,c


def test_audit_reads_saved_endpoints_and_never_calls_numeric_evaluator(monkeypatch):
    values=fixture()
    from trottertracks import c_pr_aware_df_fixed_certificate as fixed
    monkeypatch.setattr(fixed,'_local',lambda *args:pytest.fail('NO_NEW_NUMERIC_ARITHMETIC'))
    out=audit.audit(*values)
    assert out['exact_multiplicities_match'] and out['new_H4_operator_bound'] is None
    assert out['new_H4_Arb_evaluations']==0 and out['exact_translation_total_matches']


@pytest.mark.parametrize('fault',['multiplicity','phase','record','plan'])
def test_saved_corruption_is_FAIL(fault):
    saved,plan,cert,a,c=fixture()
    if fault=='multiplicity':cert['block_multiplicities'][next(iter(cert['block_multiplicities']))]+=1
    if fault=='phase':cert['block_witnesses'][next(iter(cert['block_witnesses']))]['delta_pi_exact']='100'
    if fault=='record':a['phase_hex']=.125.hex()
    if fault=='plan':plan['factors']=[]
    with pytest.raises(ValueError):audit.audit(saved,plan,cert,a,c)
