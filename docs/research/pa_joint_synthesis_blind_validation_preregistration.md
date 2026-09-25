# P-A v1 blind transfer validation事前登録

最終更新：2026-09-25 JST

## statusと目的

status：`completed_both_strata_passed`

完了結果は[専用検証文書](../research_direction_joint_synthesis_blind_validation.md)を参照する。

この文書は、P-A v1が単一H4・単一compilerで得た改善を、結果を見て規則を変更せずに

1. 未使用physical snapshot
2. 未使用compiler context

へ移せるかを判定するための事前登録である。新しいcompletion familyまたはpairwise transition最適化を
試す検証ではない。P-A v2は本検証の対象外とし、v1結果を見た後にv2を設計する場合は別のblind setを使う。

この文書作成時点で、以下のH5 P-A回路compileおよびH4 opt2 P-A比較は実行していない。

2026-09-25のcompile前dry-runで48 holdout taskと6 operator probeを生成し、expected-task bundleを
保存した。bundle content fingerprintは
`40490bd435b1286b59213559bd0296649a79e190ffc99025450089080e335e05`、file SHA-256は
`109bd7047f9e4832d1fa0c870af1a68c363829d9a61ff7f387630ac8fd6814c7`である。H4 holdout streamは
元pilotのdigest `22f753022216bbb9ce31e36cd4cbbb910f23c74eb962acbc57e13209d8aebfdd`と一致した。
このdry-runはevent生成とhash固定だけで、回路compile、operator比較またはgate集計を行っていない。

## 固定するv1実装

- 基点commit：`3336f030c56f4eb500b1bdf8f541619f9c445038`
- `src/trotterlib/research_direction_joint_synthesis_pilot.py` SHA-256：
  `6f239f0c1f11fda55f4949b730b2087eff3b3538bc003da035767f69dad0d92b`
- `src/trotterlib/df_rte_qiskit.py` SHA-256：
  `c710de23b7bfef525199ab28b897c21b2a78723a01b29da6067ff64354dcd436`
- 元P-A artifact file SHA-256：
  `b5165af6561ca7b1ba8dfe6920bbdd01a2c2c3699d8fd79ed721eba000b8b64f`
- 元P-A content fingerprint：
  `1a9840a4ee46daa6e3749272593acf3e8ae29be9fcecc1a05b0bd9ea817fbc37`

固定するcandidateは`interval_union_dp_basis_operation_count_v1`である。

- source-basis runごとの連続区間DP
- 区間候補はfull basisまたは決定論的preserved support-union completion
- 目的関数は`twice_basis_operation_count_then_segment_count_then_union_size`
- adjacent equal-basis cancellationを全policyで有効化
- 結果を見た後のthreshold、目的関数、completion規則、tie-break変更は禁止

## 共通baselineと報告指標

全条件で次の4 policyを同じevent列へ適用する。

1. `full_basis_shared`
2. `event_support_restricted`
3. `support_run_le_1`
4. `interval_union_dp`

primary metricはcompiled `rz_count`とする。次も全て報告する。

- `cx_count`
- `rz_depth`
- `cx_depth`
- `total_depth`
- `circuit_size`
- trajectory別差
- 4 policy内compiled oracle regret
- plan変更率
- multi-application interval使用数
- support-union interval使用数
- DP transition数、wall time、peak memoryを取得できる場合はその値
- operator最大残差とrelative ancilla phase一致

compiled oracleは4 policy内だけのoracleであり、global circuit optimumと呼ばない。

## Stratum A：physical transfer

### 固定snapshot

- molecule：H5 linear chain
- geometry：1.0 Å
- basis：STO-3G
- qubits：10
- charge：+1
- multiplicity：3
- DF rank：9
- `L_D=4`
- snapshot：
  `artifacts/rte_cost_system_size_h5/2026-08-25/snapshots/h5_sto3g_d100_rank9_df_snapshot_v1.npz`
- snapshot SHA-256：
  `52810e8e5fa00f7cd38f8c59dd9fa647268d0e2c022b76599228a43d9a3db81c`
- Hamiltonian hash：
  `3b7f161147cc72936ad74863933e14b58dc45d9fb32ab1be9b8ab4f70e6ff60e`
- partition hash：
  `94e7767b3a802aa84975f60db13daf59d16f8a97382a9204b00becf1777f834e`
- preparation hash：
  `4a0ceb1fccaa8ed3a5584d77e7e85b9c0bad8dabe1028da385470b7def88e0c0`

### RTE・event条件

- short-step time：0.025
- finite Taylor cutoff：2
- holdout lengths：3、5、8
- trajectory数：各8、合計24
- holdout master seed：`2026092601`
- operator probe：各長さ1件
- operator-probe master seed：`2026092602`
- holdout partition label：`h5_physical_transfer_blind`
- operator partition label：`h5_physical_transfer_equivalence`

既存H5 connected-cluster検証のtrajectoryまたはseedを再利用せず、P-A用にfresh event streamを作る。

### compiler

- Qiskit：1.3.0
- basis gates：`rz,sx,x,cx`
- optimization level：1
- transpiler seed：17
- coupling map：なし

このstratumはcompilerをH4 pilotと同じにし、physical snapshot移送だけを主に判定する。

## Stratum B：compiler transfer

### physical/event条件

- H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12、`L_D=3`
- `delta=0.02`、finite Taylor cutoff 2
- 元P-Aのholdout列長3、5、8、各8 trajectoryを同じseed・同じevent digestで再利用する
- 元P-A operator probe 3件も同じevent digestで再利用する
- holdout master seed／partition label：`2026092502`／`holdout`
- operator master seed／partition label：`2026092503`／`equivalence`

同一event streamを使うことで、optimization levelだけのpaired changeを評価する。

### compiler

- Qiskit：1.3.0
- basis gates：`rz,sx,x,cx`
- optimization level：2
- transpiler seed：17
- coupling map：なし

この条件はP-Aの候補選定に使ったopt1と異なり、P-A v1について未実行である。

## 事前固定gate

各stratumを別々に判定し、次の6 gateを全て満たすことを必須とする。

1. operator最大残差が$10^{-10}$以下で、relative ancilla phaseが全probeで一致する。
2. `interval_union_dp`のpooled RZが`support_run_le_1`より2%以上少ない。
3. trajectory別の最大RZ悪化が`support_run_le_1`比5%以下である。
4. 4 policy内compiled oracleに対するpooled RZ regretが、full-basis pooled RZの1%以下である。
5. holdout trajectoryの20%以上で現行policyとbasis planが異なる。
6. multi-application intervalとsupport-union intervalが少なくとも1 trajectoryで実際に選ばれる。

補助判定として、各列長の平均RZ変化、CX、depth、worst trajectoryを報告する。ただしprimary gateを
別metricの改善で置き換えない。

## theme decision

### 完了後の判断

2026-09-25に両stratumを完了し、各6 gateを全て通過した。H5 physical transferのpooled RZは現行比
-17.076%、H4 optimization level 2は-6.598%で、operator最大残差はそれぞれ
$2.998\times10^{-15}$、$3.126\times10^{-15}$だった。従ってP-A v1を正式主題候補へ進める。
artifact fingerprintは`78af3474898dbf989780ea5f2881cb5b61595609dd2698164b9846c1ce1c5919`である。

この判断は事前登録gateに基づく履歴として保持する。後続の
[形式化・機構監査](pa_joint_synthesis_v1_formalization.md)では、完成済み54 recordの全てが
一run一segmentで、run内区間分割は0件、holdout 256 eventは全てTaylor order 0と判明した。
従って形式化監査直後のP-Aは`conditional_candidate_pending_nondegenerate_mechanism_validation`であり、
blind gate通過だけからinterval subdivision固有の寄与を主張しない。
さらに後続の
[非退化mechanism validation](../research_direction_joint_synthesis_mechanism_validation.md)では、
forced-support order-2の全30 taskで一区間baselineとの差が0だった。現行判断は
`stop_pa_interval_dp_as_primary_and_return_to_pc`であり、P-Cを主研究候補とする。

### 事前固定していた分岐

- Stratum A/Bの両方が全gate通過：P-A v1を正式主題候補へ進め、v2の理論化・実装要否を判断する。
- いずれかが不通過：P-Aを一般化可能な主題として確定しない。失敗条件を限定的な成立範囲として記録し、
  P-Cへ戻る。
- 実装不具合、snapshot不整合、compiler failure：科学的な不通過と分けて`execution_invalid`とし、
  コード修正後も同じseedとgateを維持する。

不通過後にthresholdを緩めて同じ結果を採用しない。探索的follow-upを行う場合は、confirmatory resultと
明確に分離する。

## leakage防止

- H5 eventを生成した後、aggregateを見る前にexpected task manifestとevent digestを保存する。
- worker結果の一部を見てpolicy、completion、gateまたはsample数を変更しない。
- 全24 trajectoryと3 operator probeが揃うまで主比較を作らない。
- 追加精度が必要な場合は、最初のbatchを保持したままfresh seedのsupplementとして登録する。
- v2を設計するときはH5の本blind setをtrainingへ流用せず、別snapshotまたは別geometryを確保する。

## 本検証に含めないもの

- pairwise relative-basis transitionを直接最適化するP-A v2
- coupling map付きrouting
- backend/noise
- full partial-$S_2$ wrapperまたはHadamard wrapper
- RPE shot数、長$q$ proxy、最終総cost
- H12

これらはv1のtransfer gateを通過し、主題を確定する場合にだけ必要性を再判断する。
