# A82 PR側だけのone-shot契約案（実行未承認）

新SOURCE：`a8961c45d7fd0310c1e34c50ffd004dd58898247`。固定数式SOURCE：`b6716396acc5760fde7cf7cc6f2481b183a08f10`。
既存mechanism_analysis.py、mechanism_explanation.py、run_mechanism_explanation.pyはb671のbyteと一致する。
追加はmechanism_a82.pyとrun_mechanism_a82.py。旧32件一括launcherは起動しない。

対象はA82_OLD_PR.PR、H8/1 Å/STO-3G/tol-only DF28/T=.8、通常signed/factor1、ε=.05、
B2_p7_q2_2nd_r4_K2、旧confirmation union n32。回収原signalの`/signals/32`とSOURCE・SHA・候補・DF/T/cohortを再確認した。
前回回収の2 fieldを転記した別payloadを固定し、旧入力・旧契約・旧結果を書き換えていない。

## 将来許可する最小実行

1. 新grantのapproved/approved_by_user/runtime_authorizationを検査し、明示承認receiptのSHAを照合する。
2. SOURCE closure5 file、入力bindingと全入力、Python/runtime、元core32・元説明32のSHAを照合する。
3. 未使用outputへ新one-shot marker/ledgerを排他的に作り、workerに独立claimを要求する。
4. **既存explain_sideをA82.PRへ1回だけ呼ぶ。cosine/sineの2軸だけを評価する。**
5. PF側は既存完了JSONをexact copy。他63 side、core、31比較因子、SEは再計算しない。
6. 入力・SOURCE/runtime・grant/receipt・ledger/marker・元結果SHAを再照合し、親がnative exit0/wait/reapと出力SHA/schemaを受け入れる。

normalization段階、empirical log margin、signed bias、headroom、alpha、保存log bound、保存整数ceilを
元のbinary64順序とabs1e-12で検査する。保存shotの置換・再最適化は0。
入力欠測／科学的不一致／破損・I/O/native異常ではfail-closed STOPし、自動retry/refundしない。
無関係worktreeのHEAD/statusや一時的PSIだけを停止条件にしない。資源の恣意的な固定上限は追加していない。

出力schemaは[A82_output_schema.json](A82_output_schema.json)。本契約は**新PR側の検証deltaのみ**。
A82のPR/PF因子比の追加計算もこの最小scopeには含めない。必要なら次の別契約で判断する。
既存EXPLANATORY_INPUT_MISSINGは履歴として保持し、成功時も別結果で新PR statusを記録する。

## 現在の状態と次の承認

契約・input binding・runtime profileはlocal専用監査領域に固定した。
grant案はapproved=false/approved_by_user=false/runtime_authorization=false、明示承認receipt=null。
無効grantによるCLI拒否は実際に確認し、契約・科学入力読込前にNUMERICAL_AUTHORIZATION_REQUIREDで終了した。
科学output root未作成、有効grant発行0、実データ解析0。

次担当はユーザー：このSOURCEとlocal入力契約に結合されたA82.PR2軸のone-shot1回を別途承認する。
承認後のCodexは新grantと承認receiptを用い、fresh SHA gate通過時だけ1回実行し、結果保存後mandatory STOPする。
core算術／PF評価／他side評価／因子比／新統計／sampling／compile／ranking／212比較／公開push／最終freeze／次stageは0。
