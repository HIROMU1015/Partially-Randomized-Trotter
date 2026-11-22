from copy import deepcopy
import hashlib
import json

import pytest

from trottertracks.pr_df_price_completion import complete, exact
from trottertracks.rpe_hchain_prices import hydrate


def configuration(n_atoms, outer):
    system = f"H{n_atoms}"
    binding = dict(system=system, rank=10 if n_atoms == 4 else 19,
                   hamiltonian=system + "-h", snapshot=system + "-snapshot", state=system + "-state")
    spec = dict(binding=binding, method="PR", split={}, profile={"compiler": "compiler", "control": "control"},
                formula="2nd" if n_atoms == 4 else "4th(m5_best)", L_D=7,
                law={"model": "finite"}, delta_pf=1, h=1, M=4)
    rounds = [
        dict(m=m, actual_delta_pf=1, q_PF=2**m, R_outer=r)
        for m, r in enumerate(outer)
    ]
    return dict(id=system + "-case", rounds=rounds, spec=spec)


def measured(config, rounds, ids):
    def compiled(row, axis):
        q, outer = row["q_PF"], row["R_outer"]
        return 20 + 3 * q + 2 * q * outer + (axis == "Y")

    return [dict(m=row["m"], axis=axis, physical_id=ids[row["m"]], unit_price_RZ=compiled(row, axis),
                 kind="DIRECT_FINITE_COHORT", price_evidence={"preparation_hash": "prepared", "paths": [{}]})
            for row in config["rounds"] if row["m"] in rounds for axis in ("X", "Y")]


@pytest.mark.parametrize(("atoms", "outer", "calibration_max", "features"), [
    (4, [1, 2, 2, 3, 3], 2, ["one", "q", "qR"]),
    (6, [16] * 5, 1, ["one", "q"]),
])
def test_complete_all_axes_with_common_model(atoms, outer, calibration_max, features):
    config = configuration(atoms, outer)
    ids = {m: str(m) for m in range(5)}
    direct = measured(config, {0, 1, 2, 3}, ids)
    prices, models = complete(config, direct, ids, compiler="compiler", calibration_max_m=calibration_max)
    assert prices[:8] == direct
    expected = measured(config, {4}, ids)
    assert [exact(row["unit_price_RZ"]) for row in prices[-2:]] == [row["unit_price_RZ"] for row in expected]
    assert all(record["fit"]["features"] == features and record["descriptive_holdouts"] for record in models)
    assert all(record["route"]["system"] == f"H{atoms}" for record in models)
    assert all(not row["qualified_E_C"] for row in prices[-2:])

    other_step = deepcopy(config)
    other_step["rounds"][-1]["actual_delta_pf"] = 2
    with pytest.raises(ValueError, match="insufficient measured calibration"):
        complete(other_step, direct, ids, compiler="compiler", calibration_max_m=calibration_max)


def test_hchain_binding_preserves_system_and_rejects_cross_system_prices(tmp_path):
    config = configuration(6, [16] * 5)
    spec, system = config["spec"], "H6"
    binding = spec["binding"]
    facts = dict(fragment_count=binding["rank"], hamiltonian_hash=binding["hamiltonian"],
                 snapshot_sha256=binding["snapshot"], state_hash=binding["state"])
    ids = {m: str(m) for m in range(5)}
    rows = measured(config, set(ids), ids)
    axes = [dict(row, status="ADMITTED") for row in rows]
    signals = dict(status="SIGNAL_PACKETS_READY", M=4, h=1, target=-3.0, compiler="compiler", methods={
        "PR-DF": [dict(m=row["m"], signal={"raw": {"real": 1.0, "imag": 0.0}}, candidate=dict(
            system=system, q=row["q_PF"], R_outer=row["R_outer"], formula=spec["formula"],
            prefix=spec["L_D"], id=ids[row["m"]],
        )) for row in config["rounds"]],
    })

    def save(name, data):
        payload = json.dumps(data).encode()
        path = tmp_path / name
        path.write_bytes(payload)
        return str(path), hashlib.sha256(payload).hexdigest()

    signal_path, signal_sha = save("signals.json", signals)
    config_path, config_sha = save("configuration.json", config)
    price_path, price_sha = save("prices.json", dict(scope="MODEL", compiler="compiler",
        signal_packet=signal_path, signal_packet_sha256=signal_sha, methods={
            "PR-DF": dict(configuration_id=config["id"], configuration=config_path,
                          configuration_sha256=config_sha, axes=axes),
        }))
    problem = dict(problem={"system": system}, rpe={"M": 4}, target={"energy_ha": -3.0},
                   methods={"PR_DF": dict(caps=[59] * 5,
                                           L_D_range=dict(min=1, max_offset_from_fragment_count=-1))})
    source = dict(path=price_path, sha256=price_sha, route={"kind": "test"})
    candidates = []
    expected, routes = hydrate(problem, facts, candidates, source)
    assert candidates[0]["key"]["n_atoms"] == 6
    assert expected[config["id"]]["M"] == 4
    assert routes["PR_DF"][0]["configuration_ids"] == [config["id"]]
    problem["problem"]["system"] = "H4"
    with pytest.raises(ValueError, match="selected molecule"):
        hydrate(problem, facts, [], source)
