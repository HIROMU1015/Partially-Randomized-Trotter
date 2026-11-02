"""Execution descriptions, never execution authority or a scientific certificate.

Callers admit/claim through the unchanged gate/store before any science. These
pure formatters also accept saved provenance in artificial metadata tests; their
return values are not permits and cannot dispatch a resolver or spend a claim.
"""
from hashlib import sha256
from .guard import require
from ..g6_w2c_b3_live_preflight_v1 import spec as S

SCHEMA='b3_execution_metadata_v1'


def context(m,*,live=None):
    require(type(m) is dict,'METADATA_MANIFEST')
    inferred=m.get('artificial_only') is False
    if live is None:live=inferred
    require(type(live) is bool and live is inferred,'METADATA_EXECUTION_MODE')
    require(m.get('SCIENCE_RUN_AUTHORIZED') is live and m.get('artificial_only') is (not live),
        'METADATA_APPROVAL_MODE')
    if live:
        a=m.get('approval_record');anchor=m.get('activation_record')
        require(m.get('approval_status')=='EXPLICIT_USER_EXECUTION_APPROVED'
            and m.get('scientific_adoption')=='USER_ADOPTED_512_CORRECTNESS'
            and all(m.get(k) is True for k in ('content_adopted','run_ready','contract_grants_execution')),
            'METADATA_USER_ADOPTION')
        require(type(a) is dict and a.get('approver')=='USER' and a.get('explicit_execution_approval') is True
            and type(a.get('approval_text')) is str and bool(a['approval_text'].strip())
            and a.get('approval_text_sha256')==sha256(a['approval_text'].encode()).hexdigest()
            and a.get('manifest_binding')==S.digest({k:v for k,v in m.items() if k!='approval_record'})
            and a.get('measured_SOURCE')==m.get('measured_SOURCE')
            and type(anchor) is dict and anchor.get('purpose')=='USER_APPROVED_ONE_SHOT_512_CORRECTNESS',
            'METADATA_USER_APPROVAL_PROVENANCE')
    else:
        require(m.get('global_scope','').startswith('ARTIFICIAL_ONLY/RELEASE/') and m.get('real_grant') is None,
            'METADATA_MOCK_PROVENANCE')
    return dict(schema=SCHEMA,execution_type='USER_APPROVED_LIVE' if live else 'ARTIFICIAL',
        artificial_only=not live,approval_mode='USER_SIMPLE_ONE_SHOT' if live else 'MOCK_APPROVAL',
        external_authority_used=False,user_approved_live_execution=live,
        manifest_binding=S.digest(m),approval_binding=S.digest(m['approval_record']) if live else None,
        description_is_execution_permission=False)


def observation(report,m):
    meta=context(m);n=report['states']['COMPLETED'];live=not meta['artificial_only']
    failed_trial=any((p.get('failure') or {}).get('trial_failure_provenance') for p in report.get('pairs',[]))
    return dict(status=('FIXED_MANIFEST_TRIALS_COMPLETED' if n==m['planned'] else
        'PARTIAL_TRIAL_PREFIX' if n else 'NO_COMPLETED_TRIALS') if live else 'ARTIFICIAL_ONLY',
        completed_trials=n,planned_trials=m['planned'],scope='FIXED_MANIFEST_ONLY',
        source_binding=m['source_release_candidate'],manifest_binding=S.digest(m),
        general_compatibility_guarantee='NOT_ESTABLISHED',unverified_inputs='OUTSIDE_FIXED_MANIFEST',
        real_F01_failure_path='RECORDED_TRIAL_FAILURE' if live and failed_trial else 'NOT_OBSERVED',
        general_failure_handling_guarantee='NOT_ESTABLISHED')


def stamp_report(report,m):
    report.update(artificial_only=context(m)['artificial_only'],execution_metadata=context(m),
        live_API_compatibility='NOT_ESTABLISHED',live_API_observation=observation(report,m),
        science_correctness_certified=False)
    return report


def validate_bundle(report,m,documents):
    expected=context(m)
    require(report['artificial_only'] is expected['artificial_only'] and
        report['execution_metadata']==expected,'REPORT_EXECUTION_METADATA')
    require(report['live_API_compatibility']=='NOT_ESTABLISHED' and report['science_correctness_certified'] is False
        and report['live_API_observation']==observation(report,m),'REPORT_FINITE_OBSERVATION')
    for name,value in documents.items():
        require(value['execution_metadata']==expected,'DOCUMENT_EXECUTION_METADATA:'+name)
        if 'artificial_only' in value:require(value['artificial_only'] is expected['artificial_only'],'DOCUMENT_MODE:'+name)
        if 'production_permission' in value:require(value['production_permission'] is False,'NO_FURTHER_PERMISSION:'+name)
        if 'claim' in value and 'execution_state' in value:
            spent=bool(value.get('claim') and value['claim'][4]==1) or value['intent_exists'] or value['run_intent_exists']
            require(value['execution_metadata_verified'] is True and value['external_authority_used'] is False and value['real_authority'] is False
                and value['real_authority_semantics']=='LEGACY_ALIAS_FOR_EXTERNAL_AUTHORITY_USED'
                and value['claim_spent'] is spent and value['run_status']==value['execution_state'],
                'STORE_MODE_CLAIM_STATUS')
    return True
