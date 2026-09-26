# P-D S1事後再解析計画

日付: 2026-09-26
種別: `post-hoc interpretation plan`（S1の事前登録または独立holdoutではない）

## 目的

P-D S1の一次結果を変更せず、次の三点を既存artifactだけから分離して確認する。

1. B1a由来の形式的な未決定と、内部workを含む主baselineの選択不確かさは同じか。
2. nested/nativeのformula一致が、構成による資源差の消失を意味するか。
3. P-D追加計算へ進む根拠が残るか、それとも新しい研究設計へ移るべきか。

## 固定入力

- 入力artifact: `artifacts/research_direction_pd_fair_comparison/2026-09-26/pd_s1_fair_comparison_v2.json`
- content fingerprint: `0ba7764da7b7d8b7e195a5c315d3dc0a65c2c79ce01c51cf021a2685977487d2`
- file SHA-256: `6b8de6e255eb0796d93398c767017c2899837a63c7230e9956fb2d9beedbeaec`
- 一次分類: `Case B`
- 一次status: `stop_s1_undetermined_boundary_no_go_decision`
- decision-relevant regret閾値: 5%

入力artifact、候補grid、誤差予算、feasibility規則、一次A--D分類は変更しない。

## 再解析する量

### 主baseline

- 主比較: B1b、B2、B4
- 診断用ablation: B0、B1a
- scope: nested、native、combined
- 各scopeについて、選択candidate、B4 regret、false acceptance、B2/B4 objective差、shot-factor差、探索境界hitを保存する。

### B4近傍集合

各scopeのB4最良値から5%以内にあるB4-feasible候補を固定集合とし、次を保存する。

- candidateと最良値からの相対差
- B2 objectiveとshot factorのB4に対する相対誤差
- formula、delta、`m_D`、R、K、construction

全候補の誤分類と、意思決定に近い候補内の誤差を区別する。

### B1aのm_D依存

B1aが選んだnested new fourth、`delta=0.4`、`R=16`、K2を固定し、利用可能な全`m_D`について、

- outer-stage値
- deterministic component actions
- deterministic / finite / total phase bound
- B4 feasibility

を並べる。同じtail設定で`m_D`だけを増やした場合に、採用中のB4上界を満たせるかを確認する。

### nested/native内訳

各構成のB4選択点について、

- deterministic component actions
- leading/finite one-shot work
- leading/finite shot factor
- deterministic/finite phase bound
- B2/B4 objective

を分離する。目的値比をcompiled circuit、実測shot、または物理的優位性とは解釈しない。

## 解釈規則

- 一次の`Case B + undetermined_boundary`を保存し、再分類しない。
- B1b/B2/B4が全scopeで同一候補、false acceptanceなし、regret 5%以下の場合だけ、
  `A-equivalent on the frozen candidate set under B4`という事後解釈を付ける。
- B1aの問題だけで未決定になっている場合、B1aを削除せず診断用ablationと明記する。
- nested/nativeでformulaが同じでも、目的値またはwork内訳の差が小さいとは結論しない。
- B4自体を真の回路・測定costとは扱わない。

## 範囲外

新しいHamiltonian生成、対角化、RTE sampling、回路compile、K4全grid、H12、長RPE、backend/noise、
最終総cost、R3 pilot、先行研究に対する新規性確定は行わない。

## 終了条件

次を機械可読artifactと結果文書へ保存して停止する。

1. 主baselineに未解決の選択問題があるか。
2. nested/native差から何が言え、何が言えないか。
3. P-Dの追加計算へ進むか、次テーマの研究設計へ移るか。

R3を検討する場合も、この再解析から自動的に採用しない。最も近い先行研究との差分と最小検証契約を
別途、計算前に固定する。
