# PR-2 new series：V0–V3結果packet

日付：2026-09-28

## 1. 結論

新系列`pr2-rebaseline-de7a5492-v1`は、結果前に固定したV1–V3を通過し、統合statusは
`S0_PRIME_PASS_V4_REVIEW_REQUIRED`となった。これは、保存済みdevelopment snapshotの完全性、
固定sectorでのmodel/state整合、rank 3/6/9のpartial分割とrandom tail再構成、記録guardが、
事前閾値内で動作したことを意味する。

V4/S1′は実行していない。`V4_authorized=false`、`S1_prime_authorized=false`、
`automatic_next_stage=null`で停止している。resource winner、PR-2の優位性、held-out transfer、
S2/S3、最終総costについての科学的結論はない。

## 2. Provenance

- 旧系列レビュー基点：`bf5d3b2405b2bec4f88dd0589b75dd737b13e052`
- 新系列specification commit：`30ea857ed4960d3e189bc0877282f11a9846635b`
- source commit：`ef86868`
- series ID：`pr2-rebaseline-de7a5492-v1`
- amendment SHA-256：`5522cc2b45d617c9be91cc5715828b1cc63baf5c73d8ea2efd51bd36cfb22b9b`
- authorization manifest SHA-256：`8040a734487dba44c68f763e2d6d9577232f16ec8ad5ba9cc8d768e1f255bc86`
- result artifact：`artifacts/pr2_new_series_validation/2026-09-28/pr2_v1_v3_result_v1.json`
- result artifact SHA-256：`0eb22c813eb838169eb455334146140467ebbc5636db78bd923b1e6bdaed46d8`
- result fingerprint：`b210b394e9cd5a8eded947b0fd12cefe19ce9f27eb6b8140b3e863df73ea7961`
- dedicated test log：`artifacts/pr2_new_series_validation/2026-09-28/pr2_v1_v3_tests_ef86868.xml`
- test log SHA-256：`c606e53812bcbf78bfabfa18ee10ddb9c747a29b0350d4d4252978c859b91da9`
- dedicated tests：7 passed、failures 0、errors 0、skipped 0
- evidence class：local validation。外部独立再現またはimmutable CIではない

実行時worktreeには別方針の未commit変更があり、artifactも`worktree_dirty=true`を明示する。ただしrunnerは、
V1–V3の実行source、専用test、直接依存sourceがsource commitと一致することを実行前に確認した。

## 3. V0：旧入力回収監査

Git history/object、stash、既知artifact/cache/NPZ/HDF5、OpenFermion MolecularData保存先、
`/home/abe`と`/tmp`の合理的な名称検索を一巡したが、旧pilotのconstant、one-body、ordered DF factors、
sector、stateを含む完全入力は回収できなかった。判定は
`OLD_INPUT_UNRECOVERABLE_USE_SEPARATE_NEW_SERIES`である。

従って旧系列の`STOP_INPUT_REPRODUCTION_MISMATCH`と`S1_authorized=false`は維持し、旧pilot数値を
新系列のsame-input baselineへ転記しない。詳細は
[`pr2_v0_input_recovery_audit_bf5d3b2_20260928.md`](pr2_v0_input_recovery_audit_bf5d3b2_20260928.md)
を参照する。

## 4. V1：snapshot integrity・model validation

development入力は、candidate性能で選別される前に最初に完全保存されたH4 linear 1.0 Å、STO-3G、
DF rank 12 snapshotである。

- raw file SHA-256：`3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`
- Hamiltonian hash：`de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`
- 二回loadの全layer digest：一致
- one-body Hermiticity relative Frobenius residual：0
- max DF-fragment Hermiticity relative Frobenius residual：0
- full/sector state norm error：ともに0
- sector外max amplitude：0
- full/sector state max差：0
- Rayleigh energy：`-2.16638744863476` Ha
- Rayleigh residual：`9.423976090351552e-16`
- sector claim：N=4、N_alpha=2、N_beta=2、S_z=0
- N_alpha=N_betaだけによるsinglet証明：なし

held-out H4 1.30 Åはraw file SHA-256
`ad7e3e7165c55dbaa395eef7a1dd74db89e1f7ab29a69ac64333f4aebf8b3e37`だけを照合した。
NPZ内部はloadせず、signal/cost/rankingは未評価である。

## 5. V2：rank 3/6/9 partial構造

全rank・B2-G/B2-Wでpartition exact cover、sampling coefficient sign、確率和、repeat preparation、
`H_D+H_R=H`再構成が通過した。

| rank | ordered prefix | exact RTE lambda_R | component数 | probability sum error | reconstruction relative spectral error |
|---:|---|---:|---:|---:|---:|
| 3 | `[0,1,2]` | 0.5832704246339983 | 324 | 0 | 4.609380101091762e-13 |
| 6 | `[0,1,2,3,4,5]` | 0.022078920369742017 | 216 | 0 | 4.60906248277871e-13 |
| 9 | `[0,1,2,3,4,5,6,7,8]` | 1.5605698408672919e-7 | 108 | 0 | 1.1272895856825816e-16 |

B2-GとB2-Wは全rankでordered tuple、unordered set、H_D、H_Rが一致した。従って新系列では
`collapse_B2_G_and_B2_W=true`とする。これは新手法の発見ではなく、同じ候補の重複を除く構造判定である。

## 6. 実行counter

| 操作 | 件数 |
|---|---:|
| input file read | 4 |
| development snapshot load | 2 |
| held-out raw hash check | 1 |
| held-out NPZ load | 0 |
| molecular calculation | 0 |
| operator reconstruction/application | 14 |
| signal evaluation | 0 |
| wrapper-probe trajectory | 0 |
| candidate trajectory | 0 |
| circuit compilation | 0 |
| quantum shot | 0 |

## 7. 解釈と次の停止点

今回確立したのは「新しい固定入力上で、本来のPR-2 correctness検査へ進むための入力・分割・記録基盤が
整った」ことまでである。lambda_Rの減少やG/W一致だけからtrade-offまたはresource優位性を推定しない。

次の候補は、外部レビュー後の限定V4/S1′である。実行する場合も別の明示的authorizationを先に固定し、
T=0.1、q=1を中心とするcorrected/raw signal、target/control、Re/Im、既定rank/control、限定compile smoke
だけに留める。expected-cost Monte Carlo、resource winner、held-out、S2/S3へは自動的に進まない。
