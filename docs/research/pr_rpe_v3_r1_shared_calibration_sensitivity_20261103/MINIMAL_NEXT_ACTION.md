# R1後の最小次アクション

2026-11-03 JST / LOCAL。R1の固定保存価格再解析は完了。**現時点ではR2/R3を費用順位の決定目的で実行しないことを推奨する。** R2/R3のcompile・sampling・旧STOP変更・実行契約は今回作成／実施していない。

数値・仮定は[DEPENDENCE_AND_SENSITIVITY.json](DEPENDENCE_AND_SENSITIVITY.json)、定義は[REPORT](REPORT.md)、出典は[VALIDATION.json](VALIDATION.json)のfixed_inputsにある。以下はlinear H4/1Å/STO-3G/legacy DF12、δ=.02、固定18round、Partial L_D3／Full L_D12、保存opt2 compiler、状態準備なしcompiled RZ比較に限る。

## 1. 追加情報が変えられる判断

保存点費用は完全再現し、Partialの約4.858%点選好は変わらなかった。元SOURCEのsampling扱いだけ、または明示した独立性を置く参考scenarioでは1.96SE診断幅が0をまたがない。しかし共分散不明を許すPSD plug-in幅では0をまたぐ。さらに短q残差＋sampling、5%、25%の全scenarioでも未確定で、真の長q systematic幅はUNKNOWNである。

そのため、価格標本を増やすことだけでは現在の結論を変えられない。判別に必要なのは、主に次の二種類の情報である。

1. **実際の共分散に関する既存証拠**：保存済みで対応が明確なq1/q2・cos/sinのpaired価格vectorがあれば、未知の依存を任意相関で埋めずにsampling部分を限定できる。既知のLOCAL initial/fresh32保存先では個別raw directoryが未発見で、固定aggregateにも必要なvectorはない。今後既存vectorが見つかる場合に限る候補で、再生成・新sampling・他SERVER探索を自動実行しない。共分散が分かっても長q systematicは別に残る。
2. **主要な長q価格の誤差を限定する証拠**：後半3roundはPartial費用の約91.4%、Fullの約83.0%。この領域の差を限定できる価格model／直接証拠が必要であり、中間qを一つ測ることと、最終qまでの一様な誤差制御は区別する。

元SOURCE sampling診断幅を固定し、**後半3roundだけ**に共通の対称adverse比率を置き、他roundのsystematicを0とする参考条件なら、残る許容比率は各候補 `.0026475667887473717`、約**0.2648%未満**。全roundを対象にすると約**0.2306%未満**である。これは条件付きのsystematic帯に対する必要な余裕であり、直接価格のSEや95%CIの取得精度ではない。共分散を未知のまま許すPSD診断ではsampling幅だけで0をまたぐため、この形式の正の許容systematic幅は得られない（JSON null、0と補わない）。

これらの小さい閾値を達成したという証拠はない。A0 selected q32の保存予測較正相対SEは約2.31%、点残差は約4.911%であり、短qの5% gate PASSを上記0.2648%の一様長q保証へ昇格できない。仮定を狭めて得た費用差分離を無条件の優位としない。

## 2. R2：長q構造の限定holdout

| 項目 | R1後の判断 |
|---|---|
| 対象回路族 | Partial L_D3、δ.02、r32/K2、selected policy、cos/sinと、Full L_D12、δ.02、r0/K0、cos/sin。保存Qiskit1.3.0/opt2/basis/seed/profileを維持する候補 |
| 対応する最終tuple | q32768,65536,131072の後半3round。すべて直接q32域の外 |
| 既存直接証拠 | Partial最新q1/2 n32＋q8 n32、A0で照合したlegacy q16/32 n8。Full q1/2/16/32 n1 |
| 新価格で直接判別できること | 選んだ新qにおける平均価格・sampling SE・固定affine予測との差。大きなずれがあればその点で移送仮定を反証する情報価値がある |
| 残ること | q64/128等で適合しても最終q32768–131072の一様誤差幅、q1/2較正依存、候補間systematic相関、準備・backend/noiseは未解決 |
| 順位判別への情報価値 | 中間qから最終域へ誤差を限定する既存の検証済み則がなく、現保存情報では判別力を保証できない。数値上、点反転は後半両候補adverse各約2.858%だが、samplingも加えた分離にはさらに小さい条件付き幅が必要 |
| 今の推奨 | **費用順位を決定する次実験としては実施しない。** 単一点holdoutによるaffineモデルの反証を独立した研究目的にする場合だけ、別途提案可能 |
| 再利用候補 | `compute_full_opt2_randomized_cell`、`_compile_paired_requests`、`_compile_deterministic_point`、`fit_affine_holdouts`、A0固定holdout照合。旧SOURCEを変更せず必要な新版だけ用意する候補 |
| 必要な新規範囲・承認 | 新しいnamed-q価格取得、必要なtrajectory sampling、holdout評価。新q/n/seed/回数は未固定。研究目的と期待する判別力をユーザーが選び、局所精密化STOP／条件付きlong-q holdoutの変更範囲を明示承認した後だけCodexが実施する候補 |

q64/128のr32回路は後半r32族のprobeであり、最終scheduleのq64/128はr1なので、その選択roundの直接価格にはならない。最大q131072の直接compile、51task再compile、機械的な標本数増加を代替の既定路線にしない。

## 3. R3：最終r1/q16,32の補完

| 項目 | R1後の判断 |
|---|---|
| 対象 | Partial L_D3/δ.02/r1/K2、m4/q16とm5/q32、selected policy、両軸 |
| 既存直接証拠 | r1のq1/2/8は各n8。A0 r32 q16/32は条件不一致 |
| 直接補完の価値 | 未測定の選択tupleで固定affine価格を検査できる。r1の局所domainを明確にする価値はある |
| 費用比較への寄与 | 二つのround合計185863281 RZ、Partial総費用の約0.0147124%。5%上振れでもDへの影響は9293164.05 RZで、点差64508153033.0625 RZよりはるかに小さい |
| r1族全体へ波及させる参考評価 | r1全round合計937506875.5 RZ。局所2点だけでなくr1族全体へ5%変化を置いても、後半r32／Full長q問題を解決しない。新しいfitは実施していない |
| 反転条件 | 他価格を固定した局所2roundだけでは、現在価格への追加比率347.0731426131582（約347倍の上乗せ）が点反転閾値。未知誤差を無限大まで許す場合の絶対保証ではなく、固定subsetの感度指標 |
| 今の推奨 | **現在の総費用順位を判別するためには実施しない。** 5%・25%程度の補完だけでは主要な未知要因へ届かない |
| 将来の再利用・承認 | 必要なら既存paired builder／固定affine holdout照合。新compile・sampling・評価と既存局所精密化STOPの変更について、目的を変えた別途明示承認が必要 |

## 4. 維持する研究結論と担当

現時点の結果は `POINT_PREFERENCE: Partial`、総合 `UNDETERMINED`。追加compileなしでこの結論と未保証部分を記録してR1を完了できる。価格の比較をさらに追究するためだけに、自動的な計算継続は不要である。

次は**ユーザー**が、現在の条件付き結果で研究記録を閉じるか、価格modelの反証等の別目的を明示して追究するかを判断する。**Codex**は新しい指示で承認された範囲だけを担当する。V1/V2の入力状態・半径・alias/IID条件、状態準備の未測定、実backend/noise、別分子・rank、一般最適性は今回解決していない。

**既存M06-F STOPを維持。R1の公開・remote照合後mandatory STOP。R2/R3、SOURCE変更、モデル再較正、科学詳細レビュー、中心claim採択へ自動進行しない。**
