# 技術検証記録

2026-11-05 JST / LOCAL。専用branch/worktree、baseは
`1789a8ca731a1de631e80a34d2b8c90657d24458`。
PROJECT_MAPと正式共通運用指示2026-10-25版§8.1/§8.4/§10を実際に確認した。
科学数値評価、独立科学レビュー、旧停止stageの再開は行っていない。

## 実施した検証

新規人工30件と既存controlled/connection/sharpness人工118件、**計148件PASS**。
Python3.11.0rc1、NumPy1.26.4、python-flint0.9.0／FLINT3.6.0、Qiskit1.3.0。
ローカルpytest所要7.45秒。immutable CI・外部再現の結果ではない。

```bash
"$PY" -m pytest -q \
 docs/research/rpe_controlled_iid_success_energy_preparation_20261105/test_controlled_success.py \
 docs/research/rpe_support_bound_sharpness_20261105/test_sharpness.py \
 docs/research/rpe_categorical_circuit_connection_20261104/test_connection_adapter.py \
 docs/research/rpe_controlled_connection_runner_20261104/test_connection_runner.py \
 docs/research/rpe_controlled_hadamard_runner_v2_20261104/test_local_probes.py
```

新規数値テストは全て人工入力。人工36軸のend-to-end経路、exact rationalのU_cat+Δ_mean、
tのzero/負値STOP、Arb160のexp上端と独立Decimal100桁比較、exact和の1/20等号と超過、
tail欠測・重複・途中停止・既存出力再使用拒否を検証した。
SOURCE/sign/policy/hash、中心/N/ε/U_cat/Δ_mean、Dのbranch/alias/C referenceの欠測・不一致検出も確認した。
旧alpha/common/Dの復元byteが固定commitと同一であることを確認した。

defaultを実際に実行し、全36軸のliteral照合、既存D/C接続、
`new_H4_scientific_calls=0`、`controlled_success_probability=null`、
`controlled_probability_decision=NOT_EXECUTED_UNDETERMINED`を確認した。
defaultテストでは数値入力NPZの`numpy.load`と新規確率関数の呼出しを禁止して通過した。
元U_catとΔ_meanの加算、t、exp、確率総和は実H4入力で評価していない。

## 保全と限界

旧A/C・Dのhelper3個は別系列commitから同じbyteを復元しただけで、旧SOURCEを変更していない。
実行結果/raw、旧計画、grant/ledger/marker/STOPは変更しない。
本commitの変更は新SOURCE・人工テスト・接続文書・必要な入口リンクに限定する。
新しい科学結果がないため、既存の研究概要・validation manifestの科学認定は更新しない。

新しい36軸確率の科学呼出し0、新しい幾何/energy/eigenvalue/H/state評価0、
trajectory/sampling/circuit build/compile0。共有環境設定や他jobへの操作0。
正式全体状態false/UNDETERMINED/null/nullを維持。
公開後はremote SHAと公開blobをこのcommitに照合し、mandatory STOPする。
