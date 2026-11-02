"""Offline metadata verification only. No scientific imports or execution."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import re

PROFILE = '622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12'
SCIENCE = '745a674546f2cb6258526fbfde1d9a00601689f8'
ORIGIN = '1549b6f891a15ff48379996c34b344118ba2606c'


def check(profile, circuit, axes, changes):
    assert profile['canonical_profile_sha256'] == PROFILE, 'PROFILE_DIGEST'
    assert profile['original_contract']['sha256'] == '2c1e12d5e543a2038d2815a5a8a134a2b241b6aafba21514fd6502d6f5f16fad', 'HISTORICAL_CONTRACT_SHA'
    assert profile['original_contract']['sha256'] == profile['historical_public_contract_witness']['contract_sha256'], 'PUBLIC_WITNESS'
    assert profile['runtime_file_count'] == 34273, 'RUNTIME_COUNT'
    assert profile['old_compiler_output_bit_identity_verified'] is False, 'BIT_IDENTITY_LIMIT'
    assert profile['receiver_can_recompute_original_canonical_digest_from_redacted_candidate_alone'] is False, 'REDACTION_LIMIT'
    assert profile['raw_profile_in_share_candidate'] is False, 'PRIVATE_PROFILE'
    assert len(profile['scope_bindings']) == 2, 'TWO_CONDITIONS'
    for row in profile['scope_bindings']:
        assert row['axes'] == 96 and row['paired_trajectories'] == 48, 'N48'
        assert row['runtime_map_exactly_matches_original_profile'] is True, 'MAP_MATCH'
        assert row['profile_canonical_digest'] == PROFILE, 'SCOPE_PROFILE'
    assert circuit['status'] == '未取得', 'CIRCUIT_STATUS'
    assert circuit['saved_pretranspile_PR_axes_acquired'] == circuit['saved_posttranspile_PR_axes_acquired'] == 0, 'NO_SAVED_PR_CIRCUITS'
    assert circuit['fingerprint_semantics']['original_circuit_fingerprint_rederived'] is False, 'NO_FINGERPRINT_REDERIVATION'
    assert circuit['reason_classification']['deleted'].startswith('UNKNOWN'), 'DELETION_UNKNOWN'
    assert len(axes) == len({(r['scope'],r['replica'],r['axis']) for r in axes}) == 192, 'AXIS_COVERAGE'
    reused = 0
    for row in axes:
        expected_reuse = row['scope'] == 'H8_CONFIRM_P8_Q2' and int(row['replica']) == 0
        assert row['reused_complete_trajectory'] == str(expected_reuse), 'REUSE_LINEAGE'
        assert row['measurement_SOURCE'] == (ORIGIN if expected_reuse else SCIENCE), 'MEASUREMENT_SOURCE'
        assert row['profile_canonical_digest'] == PROFILE, 'AXIS_PROFILE'
        assert row['saved_pretranspile_circuit'] == row['saved_posttranspile_circuit'] == 'NOT_ACQUIRED', 'AXIS_CIRCUIT_GAP'
        assert row['fingerprint_rederived_from_saved_circuit'] == 'False', 'AXIS_FINGERPRINT_GAP'
        assert len(row['pretranspile_wrapper_fingerprint']) == 64, 'WRAPPER_FINGERPRINT'
        reused += expected_reuse
    assert reused == 2, 'REUSED_TWO_AXES'
    assert changes['old_packet_files_modified'] is False, 'OLD_PACKET_PRESERVED'
    assert changes['scientific_review_started'] is changes['final_evidence_freeze'] is False, 'REVIEW_BOUNDARY'
    assert [r['new_status'] for r in changes['changes']] == ['原本で確認済み（原契約内JSON）','未取得','確認不能','確認不能'], 'JUSTIFIED_STATUS_CHANGES'


def privacy(data):
    text = data.decode('utf-8')
    for path in ('/'+'home/', '/'+'tmp/', 'file'+':/'+'/'):
        assert path not in text, 'PRIVATE_PATH'
    for pattern in (r'gh[pousr]_[A-Za-z0-9]{20,}', r'github_pat_[A-Za-z0-9_]{20,}',
                    r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', r'AKIA[A-Z0-9]{16}'):
        assert not re.search(pattern,text), 'CREDENTIAL_PATTERN'


def verify(directory, mutations=True):
    directory=Path(directory)
    manifest=json.loads((directory/'supplement_manifest.json').read_text())
    assert manifest['external_sharing_authorized'] is False
    assert manifest['push_or_external_send_performed'] is False
    assert set(manifest['files']) == {p.name for p in directory.iterdir() if p.is_file() and p.name!='supplement_manifest.json'}
    for name,row in manifest['files'].items():
        data=(directory/name).read_bytes()
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'], 'FILE_SHA'
        privacy(data)
    privacy((directory/'supplement_manifest.json').read_bytes())
    profile=json.loads((directory/'01_original_profile_receipt.json').read_text())
    circuit=json.loads((directory/'02_saved_PR_circuit_search.json').read_text())
    axes=list(csv.DictReader(io.StringIO((directory/'03_axis_coverage.csv').read_text())))
    changes=json.loads((directory/'05_missing_status_changes.json').read_text())
    check(profile,circuit,axes,changes)
    source=json.loads((directory/'04_source_and_provenance.json').read_text())
    for row in source['source_excerpts']:
        assert hashlib.sha256(row['excerpt'].encode()).hexdigest()==row['excerpt_sha256'], 'EXCERPT_SHA'
    checks=[]
    if mutations:
        for name,mutation in [
            ('profile digest altered',lambda p,c,a,s:p.update(canonical_profile_sha256='0'*64)),
            ('axis removed',lambda p,c,a,s:a.pop()),
            ('reused measurement SOURCE replaced',lambda p,c,a,s:next(x for x in a if x['reused_complete_trajectory']=='True').update(measurement_SOURCE=SCIENCE)),
            ('missing circuit promoted',lambda p,c,a,s:c.update(saved_posttranspile_PR_axes_acquired=192)),
            ('old compiler bit identity asserted',lambda p,c,a,s:p.update(old_compiler_output_bit_identity_verified=True)),
            ('private profile independent verification overstated',lambda p,c,a,s:p.update(receiver_can_recompute_original_canonical_digest_from_redacted_candidate_alone=True)),
        ]:
            vals=copy.deepcopy((profile,circuit,axes,changes));mutation(*vals)
            try:check(*vals)
            except AssertionError:checks.append(dict(name=name,status='PASS_REJECTED'))
            else:raise AssertionError('MUTATION_ACCEPTED:'+name)
        try:privacy(('/'+'home/'+'private/example').encode())
        except AssertionError:checks.append(dict(name='internal path inserted',status='PASS_REJECTED'))
        else:raise AssertionError('PRIVATE_PATH_ACCEPTED')
    return dict(status='PASS_METADATA_AND_REDACTION_STRUCTURE_ONLY',axes=192,reused_axes=2,
                original_profile_digest_matched=True,saved_PR_circuit_axes=0,
                files=len(manifest['files'])+1,mutation_checks=checks,
                original_profile_digest_recomputed_by_this_public_checker=False,
                new_scientific_calls=0,scientific_review_started=False,final_evidence_freeze=False)


if __name__=='__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent),ensure_ascii=False,indent=2))
