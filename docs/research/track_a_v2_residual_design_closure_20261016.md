# Track A V2：現行残差fit設計のNO_GO確定・終了

**利用者の今回の指示により、現行設計を`CLOSED_NO_GO_BY_USER_DECISION`として終了する。** 前回の[H6探索PF最終照合](track_a_v2_h6_pf_final_reconciliation_20261016.md)のNO_GO候補を採択した記録である。失敗した科学fitの結果ではなく、事前固定した比較可能性とtrain/holdout条件を満たさず実fitに入らなかった設計の終了である。

対象はSOURCE`5703dbda58ebb8b7a9d18dce18ba33ed45f3c11c`の、uniform A/B別・軸別・来歴stratum別に保存RZ単価をPR/PF共通affine fitし、固定holdoutで残差を比較する契約。探索/確認・profile・basisをpoolしない、train/holdoutそれぞれPR/PF各2件以上、固定hash分割、rank条件と.05 STOPを維持する。旧SOURCE、runner、未承認config、旧台帳・grant/ledgerを書き換えたり、有効化したりしない。現在の利用者指示はその実行を承認していない。

保存済み判定は比較可能PF0・fit可能stratum0。条件付きmetadata結合25件はPRのみ（H6探索17・確認8）。H6探索PRはtrain11/holdout6だがPFがない。H6確認とH8のPRは固定分割のholdout0。旧profile確認済みPF1件をbasis同値へ昇格してもよいとはしておらず、仮の件数上限でもtrain1/holdout0だった。前回の262件台帳や141 receiptの照合を再実行せず、保存された判定を参照した。

広域来歴回収、分割seedの探索、holdoutの作り直し、最小件数・profile/basis条件の緩和、V1新実測の事後的追加で同じfitを成立させる作業は終了する。新しい解析の提案は別の科学的問い・契約として利用者が判断する。

## 保全・再利用する既存成果

正本は[commit 7e416705](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7e41670565d9284da9aded94063be591fdcaa3dd/docs/research/track_a_native_df_theory_evaluation_20261016.md)。旧H6/H8各104候補×A/B×cosine/sine＝832理論wrapper行、1,212既存primitive receipts、144 finite-law receipts、10,480条件・指標行の保存compiled接続を保全する。追加V1/H8細格子の費用はこの832行に含まれない。

- A：Gaussian再利用なしの一様な理論モデル。B：同一basisの隣接境界の再利用を許す一様な理論モデル。
- 実native：決定論prefixはA相当、RTE tailの閉じたoccurrence内はB相当の混合。832行のuniform A/Bのtotalだけから、この混合のliteral期待費用を作ったとは言えない。
- 理論raw RZ/CXとcompiled full-wrapper費用の定義は異なる。一般identity phaseはoccurrenceごとのboundを保持し、測定・state preparation・routing・FT synthesisの扱いを混ぜない。PをRZと同一視しない。
- Fraction化したlawは保存binary64 lawを扱う。母平均保証・全数値誤差certificate・旧/新compiler bit同一性を今回認定しない。

## 代替解析の技術的な成立可能性だけ

次表は新結果ではない。差・比・残差・順位・図の数値作成は一切行っていない。

|別の問い|既存832行でできる範囲|不足・限界と承認条件|
|---|---|---|
|同一候補の理論A/Bにおける構造差を記述する|候補・系・軸・law・bound endpointを固定し、保存理論表だけを使う記述解析は技術的に成立しやすい。新primitive/DP/compileは不要|差・比・図の作成も新しい数値再解析なので別承認。compiled/native混合の効果、PR母平均優位、holdoutによる予測性能を主張できない。832行を新gridへ外挿しない|
|来歴が結合できた保存点に限った、fitを伴わない理論とcompiled値の対照表|候補/軸/precision反復をdedupし、定義・profile・basis・boundが結合できた点だけを別contractで記述することは可能|現状25件の条件付きPR metadataだけでPF対照や共通較正は成立しない。比・偏差の数値化には別承認が必要。失敗したfitの代替検証・holdout性能・因果効果とは呼ばない。既存接続表の再生成は不要|
|literal native混合の期待費用を組み立てる|保存primitive/law/preparation記述は将来のcomponent別処理の入力候補になる|uniform A/B totalだけでは不足。prefix/tail境界、閉じたoccurrence、scalar、boundary policyを分けるcounterと期待値処理・人工検証が必要。新しい費用評価/DPとなるため今回の再利用だけでは完了せず、別設計・実行承認が必要|

最小の再利用案は理論A/Bの記述解析である。ただし、今それを科学的に採択したわけではない。既存の対照強化が優先であり、現V2残差設計を救済するための数値処理・追加証拠回収は行わない。採用する場合は対象候補、算出量、boundの扱い、主張の限界とSTOPを新たに事前固定する。

## 終了記録

[H8準備bundle内の終了JSON](../../artifacts/resource_applicability/track_a_h8_strengthened_readiness/2026-10-16/v2_design_closure_v1.json)と全体manifestに現在の決定を追加する。過去のNO_GO候補・未承認案・科学実行SOURCEは履歴として保持する。新fit/残差/差/比/順位、law/primitive/DP、compile、grant・runtime configは0。公開後remote SHA/treeを確認してmandatory STOPし、代替解析へ自動移行しない。
