# Full PF H_eff：主評価方法の採用とH4最小検証の準備

2026-11-13 JST。[未実行計画](SCIENCE_PLAN.md)／[PLAN.json](PLAN.json)／[B HANDOFF](HANDOFF.md)／[技術検証記録](TEST_RESULTS.json)。

ユーザーは、決定論PFのトロッターエネルギー誤差の主評価方法としてH_effの固有値シフトを採用した。**方法の採用を反映したが、実H4の適用確認は未実行。個別のδE、RPE bias、十分N、global RMSE boundは未資格。** 今回は既存SOURCEの調査・metadata転記・最小修正・人工テストだけで、実H/state配列のload、step生成、対角化、保存signal再解析は全て0。

## 初回の提案

| 項目 | 候補・条件（科学実行は別途承認待ち） |
|---|---|
| 保存系 | Linear H4/1.00 Å/STO-3G、tol-only 1e−8 DF10、weighted-Hermitian completion、sector(2,2)、dimension36 |
| Full PF | 全one-body＋DF10、prefix10、S2、Δ=0.1、signed weights=[1]、元snapshot順 |
| scalar | 元ham.constantをstep時間Δで一回。tail identityの抽出・追加なし、energy shiftなし |
| 二時点 | m0：T=.1、qPF=1／m12：T=409.6、qPF=4096 |
| 外側context | h=.1、M12、P=0、元H/state/snapshot・target・prior・decoder・compiler/profile。Full PFなのでκ/R/Kはnull |
| log分枝 | PRINCIPALを**診断上の初回規約として提案**。物理alias未資格。reference energyやdecoder priorから分枝を選ばない |
| 元Hの比較 | 保存solver proxyをそのまま保持し、同sector元full DF-Hの追加Hermitian対角化1回を承認scopeへ明記する |

Cの既存計画`c5466946ab0ef9b50511210ca1e58fb7147bb0f0`の`BASELINE_FINITE_MANIFEST.json:candidates[0].configuration`をそのまま参照する。canonical configuration IDは`f2d58f320ed0ebbeff8e6d4c0fbd5de14d6338b493c92015c966935201fd93ef`。m0/m12のphysical IDは`5669f142d693eee92583`／`4841a53bcaf02927b34a`。全13roundのcanonical candidate IDと物理descriptorを[PLAN](PLAN.json)へ転記し、再生成・再ラベルしていない。元configurationのscientific_definition_adopted=falseも変更しない。今回の方法採用とは別の資格である。

二時点は最短stepと固定horizon終端を確認でき、共通stepの行列累乗と全スペクトル再構成を最小scopeで照合する案である。単一幅・二時点で分枝の物理的正しさ、全RPE精度、最適幅を証明する案ではない。

## 再利用と最小修正

| SOURCE | 利用箇所 |
|---|---|
| `2f0de95304000af251a179fefdddf2a10b8fdc14` | `pf_effective_hamiltonian.evaluate/common_step_schedule`を維持。Schur・log branch・H_eff・再構成・spectral weight・UNKNOWNの既存処理を再利用 |
| `86f37d25ffd5eee3daf7f25ce001b090b1f2cf33` | `baseline_signal_numerics.load_shared_h4/Context.pf_step`の保存入力検査・fragment順・direct expm・cache。今回は呼び出していない |
| `dd24dd329949c255d5e37b90ff8993fa8df7159b`、実H4診断`b72ea6b30e48e6a2f5d805903b7bd750b965cf16` | `pf_power_stability.expm_step/power_diagnostics`を再利用する。既に確認したexpmの正当性検証を追加目的にしない |
| `c5466946ab0ef9b50511210ca1e58fb7147bb0f0` | 上記Full PF canonical scheduleと既存数値契約。Cの39round計画自体は実行しない |
| `bba150834ca30bdb186fbfd58c55a8d51177b855` | 元full DF-H/state/target・solver proxyのSOURCE。保存reference pairの再診断は今回・初回提案とも0 |

元評価器はeffective state weightsと元/effective固有basisの二乗overlapを返すが、**元Hに対する保存stateのweightは返さなかった**。overlapの絶対値二乗だけでは相対位相が失われるため、このweightを一般には復元できない。同じ既存`eigh`結果から、元固有vector・保存state weights/energy group、固有系residual/Gram/weight-sum、最低energy groupの射影trace overlapを追加した。追加の対角化は増やしていない。declared proxyとmatrix minimumを分けたシフトも返す。自動的な対応groupの選択はせず、`matched_effective_group=null`を保持する。

`scientific_method_status`はFull/Truncated PFで`ADOPTED_MAIN_DETERMINISTIC_PF_ERROR_ROUTE`、PR head/exact-tailでは`NOT_ADOPTED_FOR_TOTAL_PR_ERROR`。数値statusやalias/ground資格とは独立する。旧2f0のREPORT・人工packet・テスト記録は書き換えていない。新runner/controller・candidate ID生成器・承認管理機構は追加していない。

## 確認済み範囲と限界

新4人工テスト＋既存15人工回帰は全PASS。元H/stateのweightとproxyの非置換、縮退射影、effective minimumのstate weightが0でも自動ground対応しないこと、方法採用と資格の分離、元固有系不成立時のUNKNOWN/残差保存を確認した。実分子に関する性能の証拠ではない。

保存canonical metadata全13行について同一recipe/actual Δとexact T=qPFΔを照合し、`COMMON_STEP_DECLARED`。実operatorの全round同一性やphysical aliasを数値確認した意味ではない。実load後のSOURCE・sector・scalar検査は未実行。

δEは、指定分枝上のeffective eigenenergyと元full DF-Hの対応energyとの差として用いる。現時点のminimum差はその候補診断であり、対応する元ground/保存state/他spectral成分の証拠なしにRPE biasへ置かない。Truncatedのfull target、PR全体との区別、Bのraw直接riskとの二重会計防止を[HANDOFF](HANDOFF.md)に記した。

## 次の作業とSTOP

**ユーザー**が[有限案](SCIENCE_PLAN.md)の一候補・Δ・二時点・principal分枝規約・元H対角化1回を含むscopeを明示承認した場合に、**Codex A**が実行する。**B**は受け取ったspectrum/状態対応の資格を既存risk契約で扱う。今回の技術成果をcommit/push・remote照合後にmandatory STOPし、科学実行・幅探索・Truncated/PR拡張へ進まない。
