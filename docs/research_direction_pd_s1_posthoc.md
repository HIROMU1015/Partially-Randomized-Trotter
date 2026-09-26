# P-D S1固定artifactの事後再解析

日付: 2026-09-26
status: `posthoc_complete_stop_pd_without_s2`

## 結論

S1 v2の保存済み308候補だけを再集計した。一次結果の`Case B + undetermined_boundary`は変更しない。
その上で、内部workを数えるB1b、leading absolute-tail-time modelのB2、finite-RTE modelのB4は、
nested、native、combinedの各scopeで同一候補を選び、false acceptanceもB4 regretもなかった。

従って、今回の固定候補集合とB4参照に限れば、主baselineの選択は
`A_equivalent_on_frozen_candidate_set_under_B4`と事後解釈できる。これは事前登録Case Aへの再分類ではない。

B1aの未決定は内部workを目的値へ入れない定義に由来する。B1aが選んだ固定tail設定では、理想的に
signal radiusを1としてもfinite位相上界`1.8178593e-6 rad`だけで予算`8.0e-7 rad`を超える。
従って`m_D`だけを128より増やしても、採用中のB4上界では適格化できない。

finite補正固有の選択改善は得られていないため、P-DをS2へ進めない。次は新しい計算ではなく、
R3の最も近い先行研究との差分を監査し、採用する場合だけ別の最小研究契約を計算前に固定する。
R3の採用・新規性は本再解析では確定していない。

## 固定入力と方法

- source: `pd_s1_fair_comparison_v2.json`
- source fingerprint: `0ba7764da7b7d8b7e195a5c315d3dc0a65c2c79ce01c51cf021a2685977487d2`
- source file SHA-256: `6b8de6e255eb0796d93398c767017c2899837a63c7230e9956fb2d9beedbeaec`
- 主比較: B1b / B2 / B4
- 診断用ablation: B0 / B1a
- B4近傍: 各scopeの最良B4 objectiveから5%以内のB4-feasible候補
- 一次grid、誤差予算、feasibility、regret 5%、A--D分類は変更なし

再解析規則は[事後再解析計画](research/pd_s1_posthoc_reanalysis_plan.md)に記録した。これはS1の事前登録でも
独立holdoutでもない。

## 主baselineとB4近傍

| scope | B1b/B2/B4一致 | B4 regret | B4近傍候補数 | 近傍内B2 objective最大相対誤差 | 近傍内false acceptance |
|---|---:|---:|---:|---:|---:|
| nested | yes | 0 | 5 | 0.02316% | 0 |
| native | yes | 0 | 1 | 0.03132% | 0 |
| combined | yes | 0 | 1 | 0.03132% | 0 |

nestedの5%近傍は、最良の`delta=0.2,m_D=16,R=16`に加え、同じnew fourthの
`delta=0.1,m_D=8,R=16/32`と`delta=0.2,m_D=16,R=32/64`からなる。native/combinedの近傍は
最良候補1件だけだった。

一方、全候補ではB2 proxyが受理してB4では不適格となる候補がnested 31件、native 14件、combined
45件あった。比較可能候補におけるB2/B4 objective相対誤差の最大絶対値もnested 76.87%、native/
combined 77.15%である。

従って支持されるのは、**今回の選択近傍と固定gridではB2がB4と同じ選択をした**という限定結論である。
B2が全候補のfinite feasibilityを正しく判定する、または全域でB4の高精度近似になるとは言えない。

全scopeのB4選択は`R=16`という評価下端にある。固定grid内の選択は閉じられるが、未評価のより小さいRへ
一般化しない。

## B1aのm_D診断

固定したnested new fourth、`delta=0.4,R=16,K=2`では、B1a objectiveであるouter-stage数は
全`m_D`で22のままである。

| m_D | outer stage | deterministic actions | deterministic phase | finite phase | total phase | B4 feasible |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 22 | 672 | 4.4466e-6 | 1.8179e-6 | 6.2645e-6 | no |
| 16 | 22 | 1,344 | 1.6477e-6 | 1.8179e-6 | 3.4656e-6 | no |
| 32 | 22 | 2,688 | 9.4805e-7 | 1.8179e-6 | 2.7659e-6 | no |
| 64 | 22 | 5,376 | 7.7313e-7 | 1.8179e-6 | 2.5910e-6 | no |
| 128 | 22 | 10,752 | 7.2940e-7 | 1.8179e-6 | 2.5473e-6 | no |

これは物理的な真の誤差が必ず予算を超えるという結論ではなく、現在のB4解析上界による判定である。
B1aは削除せず、内部workを省略すると何を誤るかを見る診断として残す。主資源baselineはB1bとする。

## nested/nativeのwork内訳

B4は両構成でnew fourth、`delta=0.2,R=16,K=2`を選んだ。nestedだけ`m_D=16`を持つ。

| 量 | nested | native | nested / native |
|---|---:|---:|---:|
| deterministic component actions | 2,688 | 144 | 18.6667 |
| leading one-shot actions | 2,752 | 208 | 13.2308 |
| finite expected one-shot actions | 2,752.0184 | 208.0184 | 13.2297 |
| B2 shot factor | 1.0371807 | 1.0371807 | 1.0000 |
| B4 shot factor | 1.0374140 | 1.0374140 | 1.0000 |
| B4 objective | 2,854.9823 | 215.8012 | 13.2297 |
| deterministic phase bound | 2.7847e-7 | 1.3644e-7 | -- |
| finite phase bound | 2.2644e-7 | 2.2644e-7 | -- |

差の中心はrandom tailではなく、nested側の内部deterministic component actionsである。元Case D実装は
nested/nativeで選択formula名が変わるかだけを検査していたため、この構成差をCase Dとして検出しない。

ただし、これは解析的component-action proxyである。nested内部substep、作用単位、融合可能性、両構成の
deterministic error規則が異なるため、13.2297倍をcompiled circuitまたは物理的優位性とは解釈しない。

## 研究判断

1. **主baselineの未解決問題**: 固定gridの選択についてはなし。ただしR下端とB4モデル自体の限界は残る。
2. **構成差**: formula選択は同じでもwork proxy差は大きい。現在の内訳だけで回路優位性は判断しない。
3. **次段階**: P-D S2、H12、長RPE、追加gridへ進まない。R3は研究設計候補として先行研究との差分から監査する。

## 実行範囲と監査

- 新しいHamiltonian生成、対角化、RTE sampling、回路compile: 0
- 新規候補評価: 0
- H12、長RPE、最終総cost: 未実施
- 専用test: 4 passed
- 全suite: 615 passed、2 skipped、4 warnings、失敗0
- evidence status: dirty worktreeで生成したlocal evidence、外部独立再現ではない

## Artifact

- [S1 v2 source](../artifacts/research_direction_pd_fair_comparison/2026-09-26/pd_s1_fair_comparison_v2.json)
- [事後再解析artifact](../artifacts/research_direction_pd_fair_comparison/2026-09-26/pd_s1_posthoc_reanalysis_v1.json)
- artifact fingerprint: `976212ee45a472bf0091064e8baf3eb7a861f2c240cdcfda445e64b4c72e0245`

このartifactは元S1 artifactを変更せず、別schema・別fingerprintで保存している。
