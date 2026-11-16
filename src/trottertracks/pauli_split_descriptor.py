"""On-demand descriptor bridge to the existing Pauli split stage-1 runner.

Inventory and metadata functions read JSON only. ``derive_descriptor`` is the
numerical leaf: callers must keep it out of an unapproved real-system request.
It reuses the function that produced the historical context, without circuit
preparation or an alternative coefficient convention.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from . import df_head_pauli_tail_ld3 as L

DERIVATION_SOURCE = dict(
    commit='7f4c9a171a1d4caeaeaee6c443217edec761f4c9',
    path='scripts/h46_new2_core.py', function='matrix_only_tail_coefficients')


def fragment_count(receipt):
    """Fragment count from saved snapshot-array metadata, without array loads."""
    try:
        lam = receipt['arrays']['lambdas']['shape']
        gs = receipt['arrays']['g_matrices']['shape']
    except (KeyError, TypeError) as error:
        raise ValueError('MISSING_INPUT: receipt.arrays.lambdas.shape and '
                         'receipt.arrays.g_matrices.shape') from error
    if (not isinstance(lam, (list, tuple)) or len(lam) != 1
            or type(lam[0]) is not int or lam[0] < 2
            or not isinstance(gs, (list, tuple)) or len(gs) != 3
            or any(type(n) is not int or n < 1 for n in gs)
            or gs[0] != lam[0] or gs[1] != gs[2]
            or ('rank' in receipt and receipt['rank'] != lam[0])):
        raise ValueError('SNAPSHOT_FRAGMENT_SHAPES_DISAGREE')
    return lam[0]


def _prefix(prefix, count):
    if type(prefix) is not int or not 1 <= prefix < count:
        raise ValueError('PREFIX_OUTSIDE_ONE_BODY_HEAD_AND_NONEMPTY_TAIL')


def _descriptor(value):
    if not isinstance(value, dict) or set(value) != {'scalar', 'lambda_R'}:
        raise ValueError('MISSING_INPUT: descriptor.scalar and descriptor.lambda_R')
    if (any(isinstance(value[k], bool) or not isinstance(value[k], (int, float))
            or not math.isfinite(value[k]) for k in value) or value['lambda_R'] < 0):
        raise ValueError('INVALID_DESCRIPTOR_COEFFICIENTS')
    return deepcopy(value)


def saved_descriptor(context, prefix, source):
    """A saved descriptor or a precise missing-input report; never derives one."""
    value = context.get('prefixes', {}).get(str(prefix))
    if value is None:
        return dict(status='DESCRIPTOR_DERIVATION_REQUIRED', descriptor=None,
                    source=None, missing_inputs=['descriptor for L_D=%s' % prefix],
                    derivation_source=deepcopy(DERIVATION_SOURCE))
    return dict(status='SAVED_DESCRIPTOR', descriptor=_descriptor(value),
                source=dict(deepcopy(source), selector=['prefixes', str(prefix)]),
                missing_inputs=[])


def derive_descriptor(root, ham, prefix):
    """Numerical leaf: exact historical matrix-only helper, no circuit builder.

    Same fragment order, signed weights, descending-absolute eigenvalue order,
    I/Z/ZZ extraction and ``math.fsum`` as the saved context producer. This is
    not invoked by inventory or metadata preparation.
    """
    _prefix(prefix, ham.n_blocks)
    fixed = L.p0.fixed_function(root, DERIVATION_SOURCE['commit'],
        DERIVATION_SOURCE['path'], DERIVATION_SOURCE['function'], dict(math=math))
    scalar, lam = fixed(ham, prefix)
    return _descriptor(dict(scalar=scalar, lambda_R=lam))


def metadata_with_descriptor(root, prefix, descriptor, source):
    """Fixed P0 metadata expanded only by this requested split and provenance.

    Existing saved descriptors win, with exact equality required if the caller
    supplies one. No other split's coefficient is substituted. LD2 remains the
    original stage-1 control, and the saved context bytes are never modified.
    """
    base = L.p0.metadata(root)
    context_spec = base['context_source']
    raw = Path(context_spec['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != context_spec['sha256']:
        raise ValueError('SAVED_CONTEXT_BYTES_CHANGED')
    context = json.loads(raw)
    try:
        receipt = json.loads(Path(base['inputs']['H4_tol_receipt.json']['path']).read_bytes())
    except (KeyError, FileNotFoundError) as error:
        raise ValueError('MISSING_INPUT: saved H4_tol_receipt.json fragment shapes') from error
    _prefix(prefix, fragment_count(receipt))
    supplied = _descriptor(descriptor)
    saved = context.get('prefixes', {}).get(str(prefix))
    if saved is not None and saved != supplied:
        raise ValueError('SUPPLIED_DESCRIPTOR_DIFFERS_FROM_SAVED_CONTEXT')
    if saved is None and (not isinstance(source, dict) or not source):
        raise ValueError('MISSING_INPUT: descriptor SOURCE and input provenance')
    prefixes = tuple(dict.fromkeys((2, prefix)))
    tails, sources = {}, {}
    for p in prefixes:
        old = context.get('prefixes', {}).get(str(p))
        if old is not None:
            tails[str(p)] = _descriptor(old)
            sources[str(p)] = dict(index=deepcopy(L.p0.INDEX),
                context_sha256=context_spec['sha256'], selector=['prefixes', str(p)])
        elif p == prefix:
            tails[str(p)] = deepcopy(supplied)
            sources[str(p)] = deepcopy(source)
        else:
            raise ValueError('MISSING_INPUT: saved LD2 control descriptor')
    if tails['2'] != base['saved_tail_descriptors']['2']:
        raise ValueError('SAVED_LD2_CONTROL_DESCRIPTOR_CHANGED')
    base.update(saved_tail_descriptors=tails, tail_prefixes=list(prefixes),
                tail_descriptor_sources=sources)
    return base


@contextmanager
def stage1_metadata(root, prefix, descriptor, source):
    """Supply one missing descriptor to unchanged ``stage1.execute_p0``.

    This temporary binding is process-local and restored on failure. Callers
    should use one stage-1 process per split, not concurrent threads that patch
    this shared module. It does not authorize or itself execute the P0 leaf.
    """
    prepared = metadata_with_descriptor(root, prefix, descriptor, source)
    previous = L.p0_metadata
    def supplied_metadata(caller_root, prefixes):
        if Path(caller_root).resolve() != Path(root).resolve():
            raise ValueError('STAGE1_DESCRIPTOR_ROOT_DIFFERS')
        if not set(prefixes) <= set(prepared['tail_prefixes']):
            raise ValueError('STAGE1_REQUESTS_ANOTHER_SPLIT_DESCRIPTOR')
        out = deepcopy(prepared)
        out['tail_prefixes'] = list(prefixes)
        out['saved_tail_descriptors'] = {str(p): out['saved_tail_descriptors'][str(p)] for p in prefixes}
        out['tail_descriptor_sources'] = {str(p): out['tail_descriptor_sources'][str(p)] for p in prefixes}
        return out
    L.p0_metadata = supplied_metadata
    try:
        yield prepared
    finally:
        L.p0_metadata = previous
