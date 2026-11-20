"""Checks shared by the normal executor and completion-only collection."""


class MissingPreparationArtifacts(ValueError):
    """The completion can be checked again after the declared blobs arrive."""


def validate_preparation_result(row, receipt, outcome, artifact_cache):
    if outcome.get('request_id', outcome.get('task_id')) != row['job_id'] or outcome.get('request_sha256') != row['request_sha256']:
        raise ValueError('Preparation result identity differs from its request')
    preparation = row['request']['preparation']
    if outcome.get('stage') != preparation['stage'] or outcome.get('source_manifest') != preparation['source']:
        raise ValueError('Preparation result stage/SOURCE differs')
    expected_profile = ((row.get('eligibility') or {}).get('execution_profiles') or {}).get(row['host_id'])
    if expected_profile is not None:
        from .rpe_preparation_worker import preparation_profile_core
        from .rpe_q3_coordinator import digest
        if outcome.get('execution_profile_sha256') != digest(preparation_profile_core(expected_profile)):
            raise ValueError('Preparation result execution profile differs')
    if outcome.get('status') not in ('PREPARATION_COMPLETE', 'PREPARATION_FAILED'):
        raise ValueError('Preparation outcome is not a supported terminal result')
    if outcome['status'] == 'PREPARATION_COMPLETE' and (receipt.get('returncode') != 0 or receipt.get('signal') is not None):
        raise ValueError('Preparation result without a clean adapter exit')
    artifacts = outcome.get('artifacts', {})
    if not isinstance(artifacts, dict): raise ValueError('Preparation artifact manifest required')
    if artifact_cache.missing(artifacts.values()):
        raise MissingPreparationArtifacts('Preparation result artifacts were not durably transferred')
    # Resolve path safety before acceptance. Large payloads remain in CAS,
    # outside the small immutable completion bundle, including on replay.
    from .rpe_preparation_artifacts import artifact_path
    for name in artifacts: artifact_path(name)
