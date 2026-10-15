"""Bind current-profile preparation bytes separately from legacy evidence.

Never replaces an expected digest at runtime. No scientific imports or builds.
Prefix preparation evidence is retained from the pilot. New q variants are
static metadata expectations; actual preparation must match at approved launch.
"""
from copy import deepcopy
import json
from pathlib import Path

POLICY='STATIC_Q_LIFT_CURRENT_PROFILE_BLUEPRINT_RETAIN_LEGACY_V1'
CASES=tuple((p,q) for p in (5,10,15) for q in (1,2,4,8))


class PreparationBindingError(ValueError):
    def __init__(self,reason,diagnostic):
        super().__init__(reason);self.preparation_diagnostic=diagnostic


def differences(expected,actual):
    return {key:dict(expected=expected.get(key),actual=actual.get(key))
        for key in sorted(set(expected)|set(actual)) if expected.get(key)!=actual.get(key)}


def validate_evidence(contract,g,root):
    config=contract['preparation_identity']
    if config.get('policy')!=POLICY:raise g.Stop('FIXED_PREPARATION_PROFILE_POLICY_REQUIRED')
    ref=config['reference'];name=ref['path'];path=Path(root)/name
    if Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(Path(root).resolve()):
        raise g.Stop('PREPARATION_REFERENCE_PATH')
    raw=g.read_bytes(path)
    if (len(raw)!=ref['bytes'] or g.sha(raw)!=ref['sha256'] or
            contract['source_hashes'].get(name)!=ref['sha256']):
        raise g.Stop('PREPARATION_REFERENCE_BYTES_SHA')
    value=g.d0.strict_json(raw)
    if (value.get('schema')!='h6_full_family_static_q_preparation_identity_v1' or
            value.get('profile_digest')!=g.d0.digest(contract['profile']) or
            value.get('input_receipts_digest')!=g.d0.digest([r for r in contract['input_receipts'] if r['logical_name'] in contract['scientific_input_names']]) or
            value.get('provider_SOURCE')!=contract['native_provider']['SOURCE'] or
            value.get('provider_closure_digest')!=g.d0.digest(contract['native_provider']['source_hashes']) or
            value.get('input_array_generation') is not False or value.get('native_build_compile_performed') is not False):
        raise g.Stop('PREPARATION_REFERENCE_PROFILE_INPUT_PROVIDER')
    rows=value.get('cases',[])
    if [(r['prefix'],r['q']) for r in rows]!=list(CASES):raise g.Stop('PREPARATION_REFERENCE_CASES')
    for row in rows:
        key=f"{row['prefix']}/{row['q']}";current=row['current_blueprint'];legacy=row['legacy_blueprint']
        binding={k:v for k,v in contract['binding'].items() if k not in ('b0_fingerprints','fragment_order')}
        if (g.d0.digest(current)!=row['current_digest'] or g.d0.digest(legacy)!=row['legacy_digest'] or
                row['current_digest']!=contract['binding']['b0_fingerprints'][key] or
                current.get('binding')!=binding or current.get('prefix')!=row['prefix'] or
                current.get('q')!=row['q'] or current.get('T')!=.8 or current.get('formula')!='2nd' or
                {k:v for k,v in current.items() if k!='preparation_hash'}!=
                {k:v for k,v in legacy.items() if k!='preparation_hash'}):
            raise g.Stop('PREPARATION_REFERENCE_IDENTITY_OR_SCIENTIFIC_FIELDS')
    return value


def checked_prepared(grant,prepared,*,prefix,q,bridge):
    g=bridge.g;grant.require(bridge.SCOPE)
    evidence=validate_evidence(grant.contract,g,grant.contract['runtime_root'])
    found=[r for r in evidence['cases'] if (r['prefix'],r['q'])==(prefix,q)]
    if len(found)!=1:raise g.Stop('PREPARATION_REFERENCE_CASE_UNIQUE')
    row=found[0];actual=bridge.blueprint(prepared,bridge.binding_for(grant,prefix,q),prefix=prefix,q=q)
    digest=g.d0.digest(actual);field_diff=differences(row['current_blueprint'],actual)
    audit=dict(prefix=prefix,q=q,policy=POLICY,current_expected_digest=row['current_digest'],
        actual_digest=digest,legacy_digest=row['legacy_digest'],current_blueprint=actual,
        field_differences=field_diff,legacy_field_differences=differences(row['legacy_blueprint'],actual),
        legacy_bit_identity_claimed=False,physics_or_compiled_equivalence_claimed=False,
        reference_sha256=grant.contract['preparation_identity']['reference']['sha256'])
    if digest!=row['current_digest'] or field_diff:
        raise PreparationBindingError('CURRENT_PROFILE_B0_BLUEPRINT_MISMATCH',audit)
    return prepared,audit
