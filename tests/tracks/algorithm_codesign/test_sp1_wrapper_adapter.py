"""Actual adapter tests on tiny controls, never the 12-wrapper science grid."""
from copy import deepcopy
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
import unittest

import mpmath as mp

from trottertracks.algorithm_codesign.synthesis_placement.wrapper_accounting import Gate
from trottertracks.algorithm_codesign.synthesis_placement.wrapper_sequence import (
    Angle, Native, lower_logical, canonical_fusion, selected, fusion_audit, registered_domain,
)
from trottertracks.algorithm_codesign.synthesis_placement.wrapper_adapter import (
    StoredLibrary, bounds, classify_ratio, spec_record, resource_row, numerical_bias,
    dagger, tensor, pauli, ideal_native, embedded_sequence, initial_density, signal, diagnostic_signal,
)

ROOT = Path(__file__).resolve().parents[3]
PREP = ROOT/"artifacts/track_b_sp1_wrapper_source/2026-10-06"


def native(generator, angle, role="D", position=0):
    return Native(generator, angle, ({"role": role, "logical_position": position, "native_factor": 0},))


def tiny_path(logical, probability=F(1), weight=F(1)):
    pre = lower_logical(logical)
    post, audit = canonical_fusion(pre)
    return {"probability": probability, "outer_weight": weight,
            "pre": pre, "post": post, "fusion": audit}


class SequenceTests(unittest.TestCase):
    def test_all_registered_paths_static_no_cross_role_opportunity(self):
        contract = json.loads((PREP/"contract_v1.json").read_text())
        audit = fusion_audit(contract)
        self.assertEqual((len(audit["paths"]), audit["cross_role_fusion_opportunities"],
                          audit["fusion_candidate_count"], audit["actual_fusion_count"]), (16, 0, 0, 0))
        self.assertEqual(sum(len(p["pre_fusion_native_sequence"]) for p in audit["paths"]), 960)
        for path in audit["paths"]:
            self.assertEqual(path["pre_fusion_native_sequence"], path["post_fusion_native_sequence"])

    def test_fusion_crosses_role_labels_before_any_mask(self):
        pre = [native("IZ", Angle(pi=F(1, 8)), "D"), native("IZ", Angle(pi=F(1, 8)), "R")]
        post, audit = canonical_fusion(pre)
        self.assertEqual(post[0].angle, Angle(pi=F(1, 4)))
        self.assertEqual(post[0].roles, {"D", "R"})
        self.assertEqual(audit["cross_role_fusion_opportunities"], 1)
        for mask in ("NONE", "D", "R", "DR"):
            with self.assertRaises(ValueError):
                selected(post[0], mask)  # No unreviewed mixed-role placement rule.

    def test_exact_mixed_unit_addition_and_cancellation(self):
        a = Angle(pi=F(1, 7), rad=F(-1, 3))
        post, audit = canonical_fusion([native("ZZ", a), native("ZZ", a.scaled(-1))])
        self.assertEqual(post, ())
        self.assertEqual(audit["zero_angle_deletions"], 1)
        self.assertEqual(len(audit["events"][0]["result"]["lineage"]), 2)

    def test_zero_deletion_reveals_an_adjacent_candidate(self):
        pre = [native("IZ", Angle(pi=F(1, 8))), native("ZZ", Angle(rad=F(1, 9))),
               native("ZZ", Angle(rad=F(-1, 9))), native("IZ", Angle(pi=F(1, 8)))]
        post, audit = canonical_fusion(pre)
        self.assertEqual(len(post), 1)
        self.assertEqual(post[0].angle, Angle(pi=F(1, 4)))
        self.assertEqual(audit["actual_fusion_count"], 2)

    def test_commuting_terms_are_not_reordered(self):
        pre = [native("IZ", Angle(pi=F(1, 9))), native("ZI", Angle(pi=F(1, 11))),
               native("IZ", Angle(pi=F(-1, 9)))]
        post, audit = canonical_fusion(pre)
        self.assertEqual(tuple(pre), post)
        self.assertEqual(audit["fusion_candidate_count"], 0)

    def test_signed_lowering_preserves_full_phase_record(self):
        pre = lower_logical([{"pauli": "I", "role": "D", "angle": Angle(pi=F(-1, 3))}])
        self.assertEqual([g.generator for g in pre], ["II", "ZI"])
        self.assertEqual([g.angle for g in pre], [Angle(pi=F(-1, 6)), Angle(pi=F(1, 6))])


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mp.mp.dps = 80
        cls.contract = json.loads((PREP/"contract_v1.json").read_text())
        ref = cls.contract["saved_input"]
        cls.raw = (ROOT/ref["path"]).read_bytes()
        cls.library = StoredLibrary(cls.raw, ref["sha256"])

    def test_saved_input_and_sequence_digest_are_bound(self):
        with self.assertRaises(PermissionError):
            StoredLibrary(self.raw+b" ", self.contract["saved_input"]["sha256"])

    def test_new_native_angle_does_not_trigger_new_synthesis(self):
        with self.assertRaises(ValueError):
            self.library.baseline_key(Angle(rad=F(7, 13)))

    def test_coefficients_and_probabilities_are_exactly_consistent(self):
        spec = self.library.spec(native("IZ", Angle(pi=F(1, 16))), "D")
        record = spec_record(spec)
        self.assertEqual(sum(spec["gate"].coefficients), 1)
        self.assertEqual(sum(F(p) for p in record["canonical_probabilities_exact"]), 1)
        self.assertGreater(spec["coefficient_error"], 0)
        self.assertLess(spec["coefficient_error"], F(1, 10**60))

    def test_catalogue_fast_path_is_shared_by_all_masks(self):
        gate = native("ZZ", Angle(pi=F(-1, 4)))
        records = [self.library.spec(gate, mask)["gate"] for mask in ("NONE", "D", "R", "DR")]
        self.assertTrue(all(r == records[0] for r in records))
        self.assertEqual(records[0].t_counts, (1,))
        self.assertEqual(records[0].diamond_errors, (0,))
        zero = self.library.spec(native("IZ", Angle(pi=F(1, 2))), "NONE")
        self.assertEqual(zero["gate"].t_counts, (0,))

    def test_saved_generic_delta_is_at_least_twice_saved_eta(self):
        row = self.library.sequences["rad:1/10"]
        self.assertGreaterEqual(self.library.delta("rad:1/10"), 2*F(row["error_guard"]["projective_operator_upper"]))

    def test_all_supported_joint_embeddings_match_exact_notch_channels(self):
        rho = mp.matrix([[mp.mpf(1)/4]*4 for _ in range(4)])
        for generator in ("IX", "IY", "IZ", "ZX", "ZY", "ZZ", "ZI"):
            u = embedded_sequence(generator, "T")
            v = ideal_native(native(generator, Angle(pi=F(1, 4))))
            self.assertLess(mp.norm(u*rho*dagger(u)-v*rho*dagger(v)), mp.mpf("1e-70"))

    def test_actual_joint_controlled_lowering_for_all_system_paulis(self):
        theta = mp.mpf("-0.37")
        for p in "IXYZ":
            logical = [{"pauli": p, "role": "D", "angle": Angle(rad=F(-37, 100))}]
            lowered = mp.eye(4)
            for g in lower_logical(logical):
                lowered = ideal_native(g)*lowered
            system = mp.cos(theta/2)*mp.eye(2)-mp.j*mp.sin(theta/2)*pauli(p)
            target = mp.eye(4)
            for i in range(2):
                for j in range(2):
                    target[i+2, j+2] = system[i, j]
            self.assertLess(mp.norm(lowered-target), mp.mpf("1e-70"))

    def test_controlled_system_minus_identity_flips_ancilla_signal(self):
        path = tiny_path([{"pauli": "I", "role": "D", "angle": Angle(pi=2)}])
        for mask in ("NONE", "D"):
            specs = [[self.library.spec(g, mask) for g in path["post"]]]
            d = diagnostic_signal([path], specs, self.library, {})
            self.assertLess(abs(d["finite"]+1), mp.mpf("1e-70"))
            self.assertLess(abs(d["ideal"]+1), mp.mpf("1e-70"))

    def test_saved_PAI_channel_mean_matches_signed_controlled_target(self):
        for sign in (1, -1):
            path = tiny_path([{"pauli": "X", "role": "D", "angle": Angle(rad=F(sign, 5))}])
            specs = [[self.library.spec(g, "D") for g in path["post"]]]
            d = diagnostic_signal([path], specs, self.library, {})
            self.assertLess(abs(d["finite"]-d["ideal"]), mp.mpf("1e-60"))

    def test_noncommuting_order_uses_actual_basis_and_inverse(self):
        path = tiny_path([{"pauli": "Z", "role": "D", "angle": Angle(pi=F(1, 8))},
                          {"pauli": "X", "role": "D", "angle": Angle(rad=F(1, 5))}])
        specs = [[self.library.spec(g, "D") for g in path["post"]]]
        d = diagnostic_signal([path], specs, self.library, {})
        self.assertLess(abs(d["finite"]-d["ideal"]), mp.mpf("1e-60"))
        wrong_u = mp.eye(4)
        for g in reversed(path["post"]):
            wrong_u = ideal_native(g)*wrong_u
        # A second state checks operator order even when |0> signal happens to coincide.
        proper_u = mp.eye(4)
        for g in path["post"]:
            proper_u = ideal_native(g)*proper_u
        self.assertGreater(mp.norm(wrong_u-proper_u), mp.mpf("0.01"))

    def test_conditional_channel_composition_matches_explicit_small_branch_sum(self):
        path = tiny_path([{"pauli": "Z", "role": "D", "angle": Angle(pi=F(1, 8))},
                          {"pauli": "X", "role": "D", "angle": Angle(rad=F(1, 5))}])
        specs = [self.library.spec(g, "D") for g in path["post"]]
        matrices = [[embedded_sequence(g.generator, self.library.sequences[k]["sequence"])
                     for k in s["keys"]] for g, s in zip(path["post"], specs)]
        rho = mp.zeros(4)
        for indices in product(*(range(len(s["keys"])) for s in specs)):
            w, u = mp.mpf(1), mp.eye(4)
            for i, spec, choices in zip(indices, specs, matrices):
                f = spec["gate"].coefficients[i]
                w *= mp.mpf(f.numerator)/f.denominator
                u = choices[i]*u
            rho += w*u*initial_density()*dagger(u)
        d = diagnostic_signal([path], [specs], self.library, {})
        self.assertLess(abs(signal(rho)-d["finite"]), mp.mpf("1e-65"))

    def test_tiny_outer_coin_mean_is_enumerated_not_sampled(self):
        paths = [tiny_path([{"pauli": "Z", "role": "D", "angle": Angle(pi=F(1, 16))},
                           {"pauli": "X", "role": "R", "angle": Angle(pi=F(sign, 8))}], F(1, 2))
                 for sign in (1, -1)]
        specs = [[self.library.spec(g, "R") for g in p["post"]] for p in paths]
        joint = diagnostic_signal(paths, specs, self.library, {})
        separate = []
        for p, s in zip(paths, specs):
            copy = dict(p, probability=F(1))
            separate.append(diagnostic_signal([copy], [s], self.library, {}))
        self.assertLess(abs(joint["finite"]-sum(d["finite"] for d in separate)/2), mp.mpf("1e-65"))

    def test_coefficient_bias_bound_encloses_independent_product_difference(self):
        true = [(F(5, 4), F(-1, 4)), (F(9, 8), F(-1, 8))]
        approximate = [(F(5, 4)+F(1, 100), F(-1, 4)-F(1, 100)),
                       (F(9, 8)-F(1, 200), F(-1, 8)+F(1, 200))]
        specs = [{"coefficient_error": sum(abs(a-b) for a, b in zip(t, v)),
                  "gamma_upper": max(sum(abs(x) for x in t), sum(abs(x) for x in v))}
                 for t, v in zip(true, approximate)]
        independent_l1 = sum(abs(true[0][i]*true[1][j]-approximate[0][i]*approximate[1][j])
                             for i, j in product(range(2), repeat=2))
        bound = numerical_bias([{"probability": F(1), "outer_weight": F(3)}], [specs])
        self.assertGreaterEqual(bound, 3*independent_l1)

    def test_resource_adapter_on_artificial_records_only(self):
        path = {"probability": F(1), "outer_weight": F(2)}
        spec = {"gate": Gate((F(5, 4), F(-1, 4)), (0, 1), (0, F(1, 1000))),
                "coefficient_error": F(1, 10**20), "gamma_upper": F(3, 2)}
        profile, u, record = resource_row([path], [[spec]], self.contract)
        self.assertEqual(profile["second_moment"], 9)
        self.assertGreater(u, 0)
        self.assertEqual(record["status"], "ELIGIBLE")
        invalid = deepcopy(self.contract)
        invalid["metric"]["alpha_axis"] = "1/2"
        with self.assertRaises(ValueError):
            resource_row([path], [[spec]], invalid)

    def test_large_rational_serialization_is_outward_and_finite(self):
        x = F(10**6000, 3)
        b = bounds(x)
        self.assertLessEqual(F(b["lo"]), x)
        self.assertGreaterEqual(F(b["hi"]), x)

    def test_materiality_boundaries_and_overlap(self):
        for b, label in (({"lo": "0.9", "hi": "0.95"}, "MATERIAL_GAIN"),
                         ({"lo": "1.05", "hi": "1.1"}, "MATERIAL_LOSS"),
                         ({"lo": "0.96", "hi": "1.04"}, "NO_MATERIAL_SEPARATION"),
                         ({"lo": "0.94", "hi": "0.96"}, "NUMERIC_INCONCLUSIVE")):
            self.assertEqual(classify_ratio(b), label)


if __name__ == "__main__":
    unittest.main()
