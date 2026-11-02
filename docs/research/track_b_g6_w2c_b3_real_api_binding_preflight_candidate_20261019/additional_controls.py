"""Postprocessing controls; fixed runtime SOURCE is unchanged."""
from pathlib import Path
import json,sys,tempfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.dont_write_bytecode=True

def main():
    from trottertracks.algorithm_codesign.g6_w2c_b3_api_binding_v1 import static as S,ports as P,authority as A
    P.install_guard()
    c=json.loads((HERE/'technical_contract_candidate.json').read_text())
    S.preflight(c,ROOT)
    root=Path(tempfile.mkdtemp(prefix='b3-bundle-artificial-API-extra-control-'))
    au=A.ArtificialAuthority(c,root);b=au.expected;a=A.approval(b);a['binding_hash']='0'*64
    controls=[]
    try:au.claim(b,a)
    except S.Rejected:controls.append(dict(name='fake_approval_only_mismatch',PASS=au.snapshot()['claim'] is None))
    try:S.production_entry()
    except S.Rejected:controls.append(dict(name='production_without_any_grant',PASS=True))
    try:S.real_bits(P.ArtificialBits('ordinary','X'))
    except S.Rejected:controls.append(dict(name='fake_bits_object_to_real_port',PASS=True))
    try:S.real_authority(au)
    except S.Rejected:controls.append(dict(name='fake_authority_object_to_real_port',PASS=True))
    assert len(controls)==4 and all(x['PASS'] for x in controls)
    report=dict(schema='b3_API_additional_postprocessing_controls_v1',controls=controls,
                forbidden_imports=P.forbidden_imports(),not_added_to_unit_73=True,not_independent_review=True,
                fixed_SOURCE='78c571005fe766c61b32dc867e006531099ff608')
    (HERE/'acceptance/additional_controls.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
if __name__=='__main__':main()
