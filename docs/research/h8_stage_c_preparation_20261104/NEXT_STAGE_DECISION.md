# Stage Cの次段判断

推奨は、独立cohortの9標本で探索時と同じ報告SEを目指す**標本比較・記述的精度の確認**。母集団保証を目的とする別設計へ自動拡張しない。現時点でStage Cは未実行。

固定条件はLinear H8、1.00 Å、STO-3G、保存tol-only1e-8 DF28、元state・fragment順序、16 qubit/sector4900、T=1.6、ε=.0005、P=0、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17/backend-topologyなし。

| 項目 | 推奨案 | 根拠・限界 | ユーザー承認の要否 |
|---|---|---|---|
| PR確認対象 | `7eb0a3d691b94089969b`、m5/prefix8/q1/R_outer32/K4 | Stage Bで選定済み。全domain最適の主張なし | 科学実行前に固定案を採択 |
| PF対照 | `1425ae6ed83fd9098b39`、B0I/m5/prefix15/q1 | 保存G=89291219516568。固定価格を再利用し、新PF計算なし | 同上 |
| 比較方向 | G_PR < G_PF | 確認後に方向・候補を変更しない | 同上 |
| 確認標本数n | **9**、全てfresh CONFIRMATION | 保存探索nを引き継ぐ独立再現設計。最適nやrare観測を保証するnではない | 同上 |
| 精度目標h_G | **121655623062.28098 RZ** | 保存探索SEそのものを引用。新たな分散解析・ceil計算は未実施。母集団精度保証なし | 同上 |
| 希少orderへの対応 | CFの観測次数・既存law診断・unknown条件付き費用を明示 | 探索9標本でorder2/4とも0。保存lawでの未観測確率は約85.31%/99.9984%。未観測費用を0にしない | 今回は記述報告を採択。保証・条件付き科学評価は別途 |
| Stage C実行 | 公開最終SOURCEと固定scopeで、PR9 trajectory/18軸＋代表7検査 | 別namespace、B seed衝突拒否、PF固定、pool/top-upなし | **別途明示的な科学実行許可が必要。現在未実行** |

SOURCEの親は全件公開結果 `7fb1d125e05238a7d932fc89b8dab6ac6a7f8c84`。実行SOURCEにはこの専用準備branchの公開最終commitを使う。元Signal/native/phase SOURCEと入力identityは[計画](PLAN.md)・[scope案](../../../configs/h8_long_time/stage_c_proposal_20261104.json)に固定した。

このn/h_Gで確認するのは、別cohortでの観測平均の向きと記述的SE目標の達否である。n=9でrare費用を十分観測できる、SEが必ず目標内になる、母集団で優位になる、という意味ではない。目標未達でも自動追加しない。

母集団優位を求める場合の不足は、未観測orderの条件付き費用・一様なcompiled RZ上界・rare寄与/分散の評価・独立性等の仮定・事前の保証基準。未compile構造上界は調査したが、post-compile価格上界としての認定はしていない。全domain最適性、全RPE総費用、他分子への一般性も未保証。

次の担当：ユーザーが上表のn・h_G・比較・SOURCEと科学call範囲を採択し、実行する場合は別途明示許可する。Codexは許可後にだけStage Cを実行する。今回は技術準備完了でmandatory STOP。詳細科学レビューも自動開始しない。
