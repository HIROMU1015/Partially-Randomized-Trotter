# P-D S1 公平再最適化の結果

日付: 2026-09-26  
status: `stop_s1_undetermined_boundary_no_go_decision`

## 結論

固定H4条件のS1では、one-shot workを含む `B1b`、leading absolute-tail-time modelの`B2`、
finite-RTE modelの`B4`が、nested/native/combinedの全比較で同じ設定を選んだ。B2のB4 regretは
全て0で、限定K=4感度でもformula、`delta`、`m_D`、`R`は変わらなかった。

一方、outer stageだけを見る`B1a`はnested/combinedでfinite-infeasibleな上限`m_D=128`を選び、
一段境界延長後も上限依存が残った。このため事前登録規則による一次分類はCase Bだが、正式statusは
`undetermined_boundary`である。Case Bを生じさせたのはB1aだけであり、B1bとB2/B4の差ではない。

したがって、S1から「finite-RTE補正がPF選択を変える」というCase Cの証拠は得られなかった。
S2へ自動的に進まず、ここで計算を停止する。H12、長RPE、compiled total costは実施していない。

## 固定条件

- H4 linear chain、1.0 Å、STO-3G
- 8 qubits、4-electron sector、DF rank 12
- `L_D=3`
- PF: second、standard/new fourth、Yoshida/Morales eighth
- 構成: nested / native
- 共通物理時間 `T=0.8 a.u.`
- `delta={0.1,0.2,0.4}`、`q={8,4,2}`
- nested `m_D={8,16,32,64}`、境界診断 `m_D=128`
- `R={16,32,64,128}` per outer step、必要時の境界診断 `R=256`
- primary `K=2`、事前登録triggerに限るK=4感度
- 総位相誤差予算 `8.0e-7 rad`
- decision-relevant regret `5%`

研究契約は[P-D主研究契約](research/pd_primary_research_contract.md)、既知baselineとの区別は
[prior-art/baseline文書](research/pd_prior_art_and_baselines.md)、計算前の規則は
[S1事前登録](research/pd_s1_fair_comparison_preregistration.md)を参照する。

## 主結果

### 選択

| scope | B1b / B2 / B4の共通選択 | B4 objective | B4 feasible数 |
|---|---|---:|---:|
| nested | new fourth、`delta=0.2`、`m_D=16`、`R=16`、K2 | 2854.982343 | 69 / 248 |
| native | new fourth、`delta=0.2`、`R=16`、K2 | 215.801160 | 18 / 60 |
| combined | native new fourth、`delta=0.2`、`R=16`、K2 | 215.801160 | 87 / 308 |

`B1b`、B2、B4のB4 regretはいずれも0だった。combined/nativeのB2 objectiveは215.733579、
B4 objectiveは215.801160で差は約0.0313%。nestedでは2854.321195対2854.982343で約0.0232%で
ある。この差は選択を変えなかった。

nativeとnestedのobjective差は解析的component-action proxy上の差である。nativeを
implementation-optimal、compiled circuitで有利、または物理的に採用可能と結論していない。

### B0 / B1a

- B0はcombined/nativeでMorales eighth、`delta=0.1,R=16`を選んだが、tail occurrence数に対して
  short stepが足りずinteger allocation不能で、B4 false acceptanceとなった。
- nested B0はnew fourth、`delta=0.1,m_D=64,R=16`を選び、B4 feasibleではあるがregretは
  671.739%だった。
- native B1aはB1b/B2/B4と一致した。
- nested/combined B1aはnew fourth、`delta=0.4,m_D=128,R=16`を選び、位相予算を満たさず
  false acceptanceとなった。

B1aは内部workを数えないouter-stage proxyなので、同じouter stageなら位相誤差の小さい最大`m_D`を
選び続ける。この上限依存を一段延長で解消できず、事前登録どおり`undetermined_boundary`を付けた。

### K=4限定感度

triggerされたdeterministic settingはnested/nativeで各1件、候補は合計10件だった。

- nested: new fourth、`delta=0.2,m_D=16,R=16`のまま
- native: new fourth、`delta=0.2,R=16`のまま
- finite phase boundはK2の約`2.2644e-7 rad`からK4の約`2.2896e-12 rad`へ下がった
- expected component actionsを含めるとobjective差は約`4.5e-7` relativeで、5%基準を大きく下回った

これは全grid K4比較ではなく、事前登録triggerに限る感度解析である。

## 事前登録Case判定

- Case C条件: 不成立。B2 false acceptanceなし、B2 regret 0、K4のdecision-relevant差なし。
- Case D条件: 不成立。nested/nativeでB4が選ぶformulaはともにnew fourth。
- Case B条件: 形式上成立。B1aがdecision-relevantに失敗し、B2/B4は一致した。
- 境界条件: 未解消。B1aが一段延長後の`m_D=128`を選択した。

よって`primary_case=B`と`undetermined_boundary=true`を併記し、GO判定は出さない。

## 実行監査

最初のv1 expected artifactは300 taskを固定したが、高段PFでtail occurrence数が`R=16`を上回る点を
infeasibleとして保存せず、result生成前に`ValueError`で停止した。grid、閾値、分類規則は変えず、
配分不能点を`allocation_feasible=false`として残すよう修正し、v1 expectedを保持したままv2 source
hashを非上書きで再固定した。

- v1 expected fingerprint: `c1e5631e547c68670c74244109817c3640216d1a55955d293d0f58bdbd172ff9`
- v2 expected fingerprint: `e3eacbb9d8928f781df7709208f048c59aaab7052304e4eae4adc346bbf6d0d5`
- result fingerprint: `0ba7764da7b7d8b7e195a5c315d3dc0a65c2c79ce01c51cf021a2685977487d2`
- result file SHA-256: `6b8de6e255eb0796d93398c767017c2899837a63c7230e9956fb2d9beedbeaec`
- elapsed: 1.369 s
- 専用test: 5 passed
- 全suite: 611 passed、2 skipped、4 warnings、失敗0
- expected/result fingerprint validator: pass

これはdirty worktreeで生成したlocal evidenceであり、immutable CIまたは外部独立再現ではない。

## Artifact

- [v1 expected（技術的失敗履歴）](../artifacts/research_direction_pd_fair_comparison/2026-09-26/pd_s1_expected_tasks_v1.json)
- [v2 expected](../artifacts/research_direction_pd_fair_comparison/2026-09-26/pd_s1_expected_tasks_v2.json)
- [v2 result](../artifacts/research_direction_pd_fair_comparison/2026-09-26/pd_s1_fair_comparison_v2.json)

## 次の判断

この結果だけからP-Dをfinite-RTE-aware selection研究としてS2へ進めない。まず、

1. B1aを研究baselineとして残す意味があるか、内部workを含むB1bで十分か
2. Case Bに既知モデルを超える新しい適用条件が本当にあるか
3. `undetermined_boundary`を解く追加計算が研究判断を変えるのか、それともB1aの定義上の問題か

を研究方針として再検討する。追加計算を行う場合は新しい事前登録を要求する。
