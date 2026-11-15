# 粗幅p7／標準Yoshida4：固定配分のQ3復旧完了

ユーザーの復旧1件承認により、旧UNKNOWNを保全した新namespaceで、保存H4/1Å/STO-3G/tol-only DF10・B0I p7/標準Y4・PF要求幅32h/5・h_RPE=.1の保存13round信号と元Q3入力を再利用した。元X/Y配分 `[59,55,51,47,43,39,35,31,27,23,19,7,2]` の変更なし。

専用Python3.11.0rc1で本番PF workerの元AST hash/guardと人工exact DP・保存を確認し、実科学APIを1件実行した。数値閉包・保存は正常。粗幅MSE区間は表示で `[3.409058131550481301103526e−6,3.409058131550481301103539e−6] Ha²`、下界>元exact τとなり、この配分でのモデル不適合が確定した。同じ配分の現幅16h/3は保存結果b4296b110の適合（上界約2.418372e−6）を引用する。旧環境不一致UNKNOWNは履歴のまま維持する。

[REPORT](../d_p7_y4_coarse_q3_recovery_20261116/REPORT.md)／[HANDOFF](../d_p7_y4_coarse_q3_recovery_20261116/HANDOFF.md)／[C結合](../d_p7_y4_coarse_q3_recovery_20261116/C_RESULT_JOIN.json)。実行SOURCE5e2246d12、raw537df2376、C結合8a2f5c46d。

この結果は粗幅全体の不可能性やshot増の効果を判定しない。新信号/reference/build/compile/価格/fit/別配分0、粗幅費用null。統合担当へUNKNOWN解決と適合点追加なしを伝え、ロードマップ本体は変更せず公開後mandatory STOP。
