"""Future saved-enclosure comparison only. No new pi, root, signal or shot calculation.

Exact integer/rational operations compare previously saved conditional enclosures.
Their physical/numerical coverage is inherited, never upgraded to certification.
"""
from fractions import Fraction as F
import re
from .control import require_permit
from .spec import contrast_keys, EVIDENCE, INPUT


class SavedRationals:
    def __init__(self, permit, resources):
        require_permit(permit)
        self.permit = permit
        self.resources = resources
        self.fields = 0

    def bounded(self, value):
        require_permit(self.permit)
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > self.resources['max_rational_bits']:
            raise RuntimeError('saved rational/intermediate bit cap')
        return value

    def parse(self, text):
        require_permit(self.permit)
        self.fields += 1
        if self.fields > self.resources['max_fraction_fields']:
            raise RuntimeError('fraction field cap')
        if (type(text) is not str or len(text) > self.resources['max_rational_text_chars']
                or re.fullmatch(r'-?[0-9]+/[1-9][0-9]*', text) is None):
            raise ValueError('bounded integer/rational string required')
        return self.bounded(F(text))


def classify(permit, lower, upper, pi_lower, pi_upper, rational):
    require_permit(permit)
    if not (0 <= lower <= upper and 0 < pi_lower <= pi_upper):
        raise ValueError('invalid closed enclosures')
    # Strict target gamma_star < pi/3. No float epsilon or adaptive refinement.
    if rational.bounded(3 * upper) < pi_lower:
        return 'SATISFIED'
    if rational.bounded(3 * lower) >= pi_upper:
        return 'NOT_SATISFIED'
    return 'UNRESOLVED'


def evaluate(raw, contract, permit, result, checkpoint):
    require_permit(permit)
    if (raw.get('schema') != INPUT['schema'] or raw.get('status') != INPUT['status']
            or raw.get('failure') is not None or raw.get('evidence') != EVIDENCE):
        raise ValueError('saved V1 schema/status/conditional evidence')
    if any(raw.get('provenance', {}).get(k) != v for k, v in INPUT['provenance'].items()):
        raise ValueError('saved V1 source/contract/authority provenance')
    if raw.get('scientific_GO') is not None or raw.get('physical_shot_guarantee') is not False:
        raise ValueError('saved V1 interpretation changed')
    rows = raw.get('rows')
    if not isinstance(rows, list) or len(rows) != contract['grid']['saved_method_rows']:
        raise ValueError('saved V1 row inventory')
    records = raw.get('comparisons')
    if not isinstance(records, list) or len(records) != contract['resources']['max_comparisons']:
        raise ValueError('saved V1 comparison inventory')
    inventory = {}
    for item in records:
        key = tuple(item[k] for k in ['split', 'R', 'delta', 'T', 'comparator'])
        if key in inventory:
            raise ValueError('duplicate saved comparison')
        inventory[key] = item
    if set(inventory) != set(contrast_keys()):
        raise ValueError('complete saved G3/G6/G9 grid required')
    rational = SavedRationals(permit, contract['resources'])
    pi_pair = raw.get('arithmetic', {}).get('pi_interval')
    if not isinstance(pi_pair, list) or len(pi_pair) != 2:
        raise ValueError('saved pi enclosure absent')
    pi_lower, pi_upper = [rational.parse(s) for s in pi_pair]
    if not 0 < pi_lower <= pi_upper:
        raise ValueError('saved pi enclosure order')
    result['threshold']['saved_pi_interval_rad'] = pi_pair
    # Display only the outward inherited enclosure scaled by an exact positive 3.
    result['threshold']['threshold_interval_rad'] = [
        str(rational.bounded(x / 3).numerator) + '/' + str(rational.bounded(x / 3).denominator)
        for x in [pi_lower, pi_upper]]
    for index, key in enumerate(contrast_keys()):
        item = inventory[key]
        out = result['comparisons'][index]
        if key[0] == 'G9':
            if (item.get('saved_visibility_contrast_status') != 'UNRESOLVED_ABSOLUTE_CONTRAST'
                    or item.get('scientific_gain_claim') != 'ABSOLUTE_CONTRAST_UNRESOLVED_NO_RELATIVE_RESCUE'):
                raise ValueError('G9 saved absolute-contrast limitation changed')
            out.update(status='G9_ABSOLUTE_GAIN_UNRESOLVED_NOT_TESTED',
                       saved_status=item['status'],
                       scientific_gain_claim=item['scientific_gain_claim'],
                       threshold_test_performed=False)
            checkpoint(result)
            continue
        if (item.get('status') != 'UNIQUE_TRADEOFF_BOUNDARY'
                or item.get('continuous_relation_before') != 'SHARED_LOWER'
                or item.get('continuous_relation_after') != 'SHARED_HIGHER'
                or item.get('boundary_location_status') != 'ENCLOSED_INTERIOR'
                or item.get('strict_integer_saving') != 'NOT_INFERRED_FROM_CONTINUOUS_ORDER'):
            raise ValueError('saved G3/G6 boundary orientation/semantics not applicable')
        pair = item.get('boundary_interval_rad')
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError('saved gamma boundary enclosure absent')
        lower, upper = [rational.parse(s) for s in pair]
        domain_lower = rational.parse(item['common_domain']['lower_open_rad'])
        if not (0 <= domain_lower < lower <= upper and rational.bounded(2 * upper) < pi_lower):
            raise ValueError('saved boundary/common domain consistency')
        state = classify(permit, lower, upper, pi_lower, pi_upper, rational)
        out.update(status=state, boundary_interval_rad=pair, threshold_test_performed=True,
                   common_domain=item['common_domain'],
                   shared_favourable_side='ABOVE_SAVED_ROOT',
                   strict_integer_shot_saving='NOT_ESTABLISHED',
                   interpretation='CONDITIONAL_CONTINUOUS_DESIGN_INTERSECTION_ONLY',
                   rule={'SATISFIED':'3*U < p_lower',
                         'NOT_SATISFIED':'3*L >= p_upper',
                         'UNRESOLVED':'OVERLAPPING_OR_TOUCHING_ENCLOSURES'}[state])
        checkpoint(result)
    result['comparison_counts'] = {name: sum(x['status'] == name for x in result['comparisons'])
        for name in ['SATISFIED', 'NOT_SATISFIED', 'UNRESOLVED', 'G9_ABSOLUTE_GAIN_UNRESOLVED_NOT_TESTED']}
    result['status'] = 'PI3_COMPARISON_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT'
    checkpoint(result)
