# 次の有限compiled価格取得：選択されたrequestだけ

2026-11-13 JST。**未承認・未実行。価格取得round/path数・精度/confidenceは未採択。** 全52軸coverageは実行menuではなく、後から対象を選べる索引である。Aのsignal実行範囲/承認をDのcompiled取得の承認へ転用しない。

## 1. A/Bの結果を受けて固定する内容

- Bのcanonical configuration、A physical ID、選ぶcandidate/round/axesの有限集合J。
- 目的が移送/負荷pilotか、finite-cohort診断か、同lawの平均価格/区間取得か。必要な精度・confidence・law/rare/数値誤差条件。
- path index集合I_j、normal cohort名と元seed規則、使用するinput/tail/実装/出力SOURCE、許可する集計と最大compile数。
- 保存LD1 pilotやE[S]を受け入れる場合のSOURCE付きbridgeとcohort role。replayをfresh nへ足さず、旧値を新LD2へ貼らない。

BのN又は根拠付き許容集合から比較を左右する価格cellを選ぶ。未資格nominalを正式G重みにせず、価格mean/区間が足りない点はUNKNOWNに保つ。長q/小角の未検証modelを、順位差が大きいという理由で採用しない。

## 2. 取得予算の有限形

選んだrequestごとの軸集合A_jと有限path index I_jを凍結すれば、最大compile数は

\[
\sum_{j\in J}|A_j|\,|I_j|.
\]

cache hitで実compile数が減っても、その枠で別round・pathを追加しない。X/Yは同pathの個別応答。過去のX=Yから片軸を省略せず、必要なaxisはユーザーのscopeに明示する。

現在はJ/A_j/I_jも精度も未定で、数値予算をここで決めない。全13roundを直接測ることや、以前の最大16/24 compile案を前段要件にしない。m7→m8→m9の逐次実行もしない。

## 3. 承認範囲での処理

1. 採択された新科学SOURCEでL.request_forを呼び、元A config/physical/lawと新request keyを保存する。技術coverageは変更しない。
2. 元loaderでH/state/snapshot/inputを照合。既存D.NativeRuntimeで候補contextを構築し、rank/fragment順、preparation/tail/basis/component、λ_R・expected scalarを確認する。新SCF/DF/state/ground生成なし。
3. 元normal sampler/seed規則で指定pathのみ取得する。又は承認された保存eventを同ctx.prepで復元する。forced rareや新seed retryを追加しない。
4. 元Accountingとcompiled_price_fast/新ZZ/full measured wrapper/固定compiler/hash/cacheでS/C/Dを対応取得。selected axesだけを個別compileし、scalar一回・q/R grouping・event順を保持する。
5. path metrics/digest/fingerprint/seed/sourceと、全pipeline wall・段階時間/peak RSSを保存する。各mean/分散/区間を算出する場合も、許可された有限集計だけを既存adapterで行う。
6. 不十分な精度・未資格law/rare/長qはそのまま返す。N/G比較、追加fit/window/modelの校正、次候補へ自動進行しない。commit/push・remote照合後mandatory STOP。

## 4. 想定される障害と判定

metadata上は全52軸に対応するが、実input/preparationの整合・負荷・compiler成功は未確認。m12の論理event数はLD1約3342万、LD2約1048万で、巨大word・hash・compileのwall/RSSが障害になり得る。既存m6の履歴倍率やevent数の比例を保証時間/RAMにしない。

構造平均Sはcompiled価格の厳密上下界ではない。outer-window法の長q tail上界も未証明。実測不能又は平均/区間の証拠が不足する場合はUNKNOWNで保留する。小roundの値で無断補完しない。

SOURCE/input/λ/scalar/fragment/candidate/law/profile不一致、未対応条件、実行不能/保存不能は部分結果を保全してSTOP。精度未達は標本数の自動増量で解消せず次の有限案にする。共有負荷だけで正常workerを再起動せず、実際の異常/継続不能には保護を維持する。

並列度は利用可能資源と実際のpath/保存依存から選び、CPU/RAM/workerの恣意的上限を設けない。他ユーザーや共有環境の設定・process・fileは操作しない。DはA/C signalやB decoder/N処理を再実装しない。

**次の担当**：Aが固定signal成果、Bが同target/claimの予備分析/shot評価・価格要求を供給し、ユーザーが上記有限scopeを明示承認する。Dはその対象だけを実行する。承認証明書や新grant発行は不要。今回は技術準備の公開照合後mandatory STOP。
