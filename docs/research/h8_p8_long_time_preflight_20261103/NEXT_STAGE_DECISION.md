# 次段階の判断

2026-11-03 JST。技術原因調査と未承認の実験案。正式科学採択・実行許可ではない。

|問い|今回の回答|
|---|---|
|p8は解決したか|false flagの原因は解決。分類EXPLAINED_NO_SCIENTIFIC_IMPACT（flag自体の意味・作用の範囲）。faithful:p8キー欠落で比較未実施を表し、比較不一致ではない|
|既存H8結果をそのまま使えるか|このflagを理由に修正・棄却・再計算する根拠はない。保存input/cohort/価格を限定結果として維持できる。独立p8 export比較済み、全結果完全保証とは書かない|
|追加検証の価値|同じH8入力で、時間と高次化によるq削減がsigned piece/B/N/Cの負担を補うかを初めて判別する価値がある。H4/H6の勝敗を前提にしない|
|最小の次実験|T=1.6・ε=.0005の一stratum。PR prefix7/8のS2対m5、同Tの強いFull/B0/B0I（S2/Y4/m5）対照。まずsignal/shot、必要な候補だけ直接価格。確認nは将来の分散計画後に固定|
|再利用SOURCE/pipeline|H46 carrier155529e2…のsignal/native/core/scheduleとprovenance、H8封印loaderとsparse/spin/direct-tailを優先。詳細はSOURCE_REUSE.csv。H46専用restore/dense contextをH8へ無変更適用しない|
|現在の実行状態|EXECUTION_HOLD_NOT_AUTHORIZED_AND_ADAPTER_NOT_VALIDATED。p8 flagを理由とする共通入力HOLDではない。新しい真の入力/作用不一致が出れば対象scopeをHOLD|
|必要な許可|ユーザーが条件案を判断。採用後、CodexがH8 adapter・人工検査・新SOURCEとcall/n設計を準備。新referenceを含む科学実行は固定SOURCE/input/範囲への明示許可後のみ|

第一比較が成立すればT=2.0・同εを次候補とする。m5のsigned-piece負担が支配的なら同TのYoshida4 PR、不適格域しか得られなければT=1.2・同εを候補にする。いずれも現時点で自動採択・実行しない。

成果物は[P8原因調査](P8_ROOT_CAUSE_REPORT.md)、[H8長時間計画](H8_LONG_TIME_VALIDATION_PLAN.md)、本判断、[SOURCE再利用表](SOURCE_REUSE.csv)、[7人工検査](test_p8_flag_semantics.py)、[検査記録](VALIDATION.json)。元false/結果/SOURCE/STOPと[希少RTE正式終了](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/862d732fba0a5f353c522a9219dec78ae2f21a4b/docs/research/h8_rare_rte_closure_policy_20261021.md)は変更していない。

**次の担当：ユーザー**が第一条件と比較域を採るか、技術準備を依頼するか判断する。**Codex**は今回の資料をcommit/push・本文照合した後mandatory STOPし、新科学計算・新pipeline実装へ自動移行しない。
