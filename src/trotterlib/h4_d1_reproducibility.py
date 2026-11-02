"""Design-stage classification of supplied evidence, with no quantum execution.

This module does not load H4, build circuits, normalize a historical hash, issue
grants or authorize a run. Payload comparison is used only by artificial tests.
SOURCE links must be assessed separately; matching payloads are not those links.
"""
REQUIRED_LINKS = (
    'hamiltonian', 'split', 'trajectory', 'event_components', 'angles', 'wires',
    'schedule', 'scalar_phases', 'controlled_phase', 'whole_body_control',
    'wrapper', 'runtime_compiler',
)


def differences(left, right, path=()):
    """Keep every field, including type, names, phase and recursive definitions."""
    if type(left) is not type(right):
        return [(path, left, right)]
    if isinstance(left, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            if key not in left or key not in right:
                result.append((path + (key,), left.get(key), right.get(key)))
            else:
                result.extend(differences(left[key], right[key], path + (key,)))
        return result
    if isinstance(left, list):
        if len(left) != len(right):
            return [(path + ('length',), len(left), len(right))]
        return [d for index, pair in enumerate(zip(left, right))
                for d in differences(*pair, path + (index,))]
    return [] if left == right else [(path, left, right)]


def _at(payload, path):
    for key in path:
        payload = payload[key]
    return payload


def classify_payload_difference(left, right, *, display_name_paths=()):
    """Explicit, source-justified generic-composite name paths only.

    No global removal of names. A primitive's name, missing definition, changed
    control, parameters, order, wires or phase fails closed. Even Class 2 does
    not establish compiler insensitivity or historical correspondence.
    """
    allowed = set(display_name_paths)
    result = []
    for path, old, new in differences(left, right):
        metadata = False
        if path in allowed and path and path[-1] == 'name':
            a, b = _at(left, path[:-1]), _at(right, path[:-1])
            metadata = all(
                p.get('type') in ('qiskit.circuit.gate.Gate',
                                  'qiskit.circuit.controlledgate.ControlledGate')
                and isinstance(p.get('definition'), dict) for p in (a, b))
        result.append(dict(path=list(path), old=old, new=new,
                           difference_class=2 if metadata else 1))
    return result


def assess(*, historical_match, source_links, classified_differences=(),
           unresolved_semantic_fields=()):
    """Research candidate only. No boolean here can enable scientific dispatch."""
    missing = [k for k in REQUIRED_LINKS if source_links.get(k) is not True]
    classes = [d['difference_class'] for d in classified_differences]
    if 1 in classes:
        status = 'SCIENTIFIC_DIFFERENCE_STOP'
    elif missing or unresolved_semantic_fields or 3 in classes:
        status = 'UNRESOLVED_CORRESPONDENCE'
    else:
        status = 'SOURCE_SEMANTIC_CORRESPONDENCE_CANDIDATE'
    return dict(
        historical_status=('LEGACY_FINGERPRINT_MATCH' if historical_match
                           else 'LEGACY_FINGERPRINT_MISMATCH'),
        source_semantic_status=status, missing_source_links=missing,
        unresolved_semantic_fields=list(unresolved_semantic_fields),
        research_GO_candidate=status == 'SOURCE_SEMANTIC_CORRESPONDENCE_CANDIDATE',
        criterion_adopted=False, execution_authorized=False, dispatch_enabled=False,
        historical_program_identity='NOT_ESTABLISHED_BY_THIS_ASSESSMENT',
        compiler_name_insensitivity='NOT_ASSUMED', certificate_accepted=False,
    )
