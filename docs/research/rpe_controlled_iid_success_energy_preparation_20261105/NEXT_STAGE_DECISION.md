# 判断：確率接続SOURCEは準備完了、Controlled成功確率は未判定

2026-11-05 JST / LOCAL。技術準備のみ。実H4の新しい確率評価は0回。
対象：linear H4／1.00 Å／STO-3G／保存legacy DF12／Partial L_D3／δ=.02／K2、
保存physical normalized ray・N4 sector70、全18round・保存条件。

| 対象 | 判定 |
|---|---|
| Controlled平均→Hoeffdingの数学的接続 | **成立**。U_cat+Δ_mean、同じGとN/εで接続 |
| Controlled測定のIID条件 | 各軸内fresh独立uniform53 trajectory＋同じrayのideal ±1測定を仮定。軸間・round間独立は不要。固定PCG64は対象外 |
| G→数学的decoder branch | **既存証明を再利用可能**。保存中心/ε/q/reference同一、17隣接strict条件成立済み |
| 数学的decoder→energy | **既存証明を再利用可能**。同じalias条件と保存DF sector ground target。Cのreference込み上端を引用 |
| 限定科学SOURCE | **準備完了**。旧alpha/aggregate無変更、Arb160・36セル＋1集約 |
| 新しいcontrolled成功確率 | **未実行・未判定**。成功下界/失敗総和/95%判定はnull/UNDETERMINED |

接続の数式と証拠は[CONNECTION_PROOF.md](CONNECTION_PROOF.md)、
固定入力hashと1 passの計画は[SCIENTIFIC_EXECUTION_PLAN.md](SCIENTIFIC_EXECUTION_PLAN.md)、
人工検証は[TECHNICAL_VALIDATION.md](TECHNICAL_VALIDATION.md)。

今回不足する証拠は、**新Controlledモデルの36 tail上界とexact総和そのもの**。
最新36/36の局所strict結果や旧Aの0.950307030063252182だけでは代用しない。
G上のenergy上端を変える必要はなく、新しい幾何・固有値計算を追加する必要もない。
実行の技術的阻害条件は見つからなかったが、数値目標の達成を予告しない。

次の担当は**ユーザー**：公開されたSOURCE commitと本計画の限定科学実行を別途明示許可するか判断する。
**Codex**はその許可後だけ、固定SOURCEで36セル＋1集約を一度実行し、
旧energy証拠へ接続した条件付き結果と原rawを公開・照合してSTOPする。
追加証拠取得、Float64 decoder、PCG64/compiled/noisy保証、費用評価、独立科学レビューはこの提案に含まない。

旧SOURCE/raw/grant/ledger/marker/STOPを変更せず、他worktree、他ユーザー、SERVER2 H8を操作しない。
`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、
`U_virtual_IID_upper=null`、`U_exec_upper=null`を維持する。
**今回の公開・remote SHA/成果物照合後mandatory STOP。**
