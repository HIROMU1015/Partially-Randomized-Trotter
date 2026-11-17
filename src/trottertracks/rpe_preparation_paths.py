"""Resolve operational paths in original preparation inputs without editing them.

Numerical identities, Git locators, SOURCE and frozen input bytes remain intact.
Only explicitly named runtime/output paths are resolved by their caller.
"""
import json
import os


def resolve(value):
    if not isinstance(value, str):
        return value
    mapping = json.loads(os.environ.get('RPE_PREPARATION_PATH_MAP', '{}'))
    for old, new in sorted(mapping.items(), key=lambda pair: len(pair[0]), reverse=True):
        if value == old or value.startswith(old.rstrip('/') + '/'):
            return new.rstrip('/') + value[len(old.rstrip('/')):]
    return value


def runtime_request(request, fields):
    result = dict(request)
    for field in fields:
        if field in result:
            result[field] = resolve(result[field])
    if 'interpreter' in fields and os.environ.get('RPE_PREPARATION_PYTHON'):
        result['interpreter'] = os.environ['RPE_PREPARATION_PYTHON']
    if 'quote_interpreter' in fields and os.environ.get('RPE_PREPARATION_PYTHON'):
        result['quote_interpreter'] = os.environ['RPE_PREPARATION_PYTHON']
    return result
