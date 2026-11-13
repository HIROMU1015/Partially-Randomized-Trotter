# 2026-11-14：全関数RPE decoder Z1＋T1とQ3直接モデルMSE評価器（技術実装）

利用者の技術実装指示に従い、W0（`104dacdc6`）の提案を実装した。

**推定量**
- 新しいopt-in推定量`consecutive_dyadic_nearest_branch_principal_energy_total_v1`を追加した。
- zeroのとき：m=0は+0.0、m≥1は前の推定値を保持する。tieのとき：計算上のargminを採る（同値はlowest ID）。
- 旧decoderと同じbinary64演算・遷移kernelを使うので、旧decoderが解決する履歴ではbit一致する。

**risk評価器**
- Bの`evaluate`/`_bound`は変更せずに再利用した。
- 全関数版にだけ、次を加えた：更新幅補題上界、zero/tieを遷移として扱う状態DP（状態依存の残余上界とE[Ê]・E[Ê²]）、有効上界のmin。
- 主値guard・f≤1のL_D・binary64の遷移allowance（宣言値）を明示した。
- δEは加算しない。

**検証**
- 人工8法則・11,748履歴の全数列挙で、DPがMSE・bias・Var・旧未解決確率・ηとexactに一致した。
- 部分追跡64通り、guard除去の負の対照、family上端<L_Dの除外も確認した。
- 回帰は本branchで33 file／367件PASS。B本線と、guard branchへのpatch適用でもPASSした。
- 既存fileは無変更で、新規3 fileだけを追加した。

**未実施**：実H4の評価、保存DPの再解析、契約・canonical IDの変更。

[REPORT](../rpe_total_decoder_q3_risk_20261114/REPORT.md)／[HANDOFF](../rpe_total_decoder_q3_risk_20261114/HANDOFF.md)。

次の作業：ユーザーがD1・Q3契約・LD1/LD2の2評価の許可を判断する。それまではmandatory STOP。
