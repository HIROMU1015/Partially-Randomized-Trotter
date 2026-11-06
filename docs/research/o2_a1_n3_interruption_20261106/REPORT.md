# A1 N3中断原因調査：完了・原因UNKNOWN・mandatory STOP

2026-11-06 JST、SERVER1 GPUA100-gleap-01。
**直接原因、終了signal/送信元、exit codeは特定できなかった。UNKNOWNを維持する。新しい科学計算は0。**

## 確認したこと

科学SOURCE03928、中断結果d97e、事前判定基準a53cを固定して調査した。
対象は保存linear H4/1.00 Å/STO-3G/legacy DF12/8qubit/全4電子sector/L_D3、m0/q1/δ=.02/r1/K2/coefficient_atol0、X/Y、N1 actual Q/selected policyとN2保証ground・認証済exact engine。

- 397bit/397価格と全79 journalを再照合し、旧会計・call countsと一致した。旧係数、SOURCE、STOPを変更せず、N3を再開していない。
- 未完了77はA1/Y/trainの層1=51、層2=26。全て最後のCALLはselected_full_buildだが、これは入口記録であり、build/その後のexact gate適用のどこで停止したかは不明。
- 主sessionの20:17 usage_limit_exceeded、管理process交代、20:20:50の別processのWebSocket resetを取得した。campaignの最終親追記は20:20:35。時刻だけで因果関係を確定しない。
- dmesg/kernel/auth/syslogは権限上確認不能。現在のSSH/cgroupは20:40以後の別scopeで、memory.events=0を旧processのOOM否定証明に使えない。
- parent/workerのwait status、signal、call return、行timestamp、旧cgroup記録が不足し、親finalizerも未到達。観測性を補う必要がある。

詳細：[N3_INTERRUPTION_DIAGNOSTIC](N3_INTERRUPTION_DIAGNOSTIC.md)。

## 改善と次回方法

[再発防止案](N3_RECURRENCE_PREVENTION.md)では、会話/SSHと寿命を分ける持続manager、Restart=no、親・worker終了記録、実childのcgroup/ancestor/pressure、call start/return、bit先行保存と未完了attempt分類を提案する。
現在Linger=noであり、tmux/nohup/setsidやuser serviceだけで全logout/killへ耐える保証は置かない。今回はmanager起動・共有設定変更を行っていない。

最小のjournal診断と未接続のdirect-child観測prototypeを追加した。既存reconcile・atomic writer・shared_cpuを再利用し、**人工19件PASS**。科学入力/engineは起動していない。
prototypeは旧77workerへ配備された修正ではなく、持続manager経路とworker終了記録の接続・人工的な配備検証は次の技術作業として残る。

[復旧2案](N3_RECOVERY_OPTIONS.md)では、技術上の既定推奨を**案B：新しい独立campaign**とした。旧stream位置・77不確定attemptを保証できず、experiment分離が明確なためで、科学的な優劣や方針採択ではない。
案Aなら397件を再取得せず使える可能性があるが、PCG64 prefix復元、idempotent ledger、77追加attempt、旧one-shot契約の限定変更を先に具体化する必要がある。

いずれも未実行。次は持続manager/観測の技術接続と人工確認、新SOURCE固定を行い、案AまたはBの有限科学scope・stream/seed方針・STOPへ**別途明示的な実行許可**が必要。
ε/α・control・予算等は今回変更していない。重要科学レビュー、期待費用評価、追加experimentも開始しない。

## 成果物と保全

証拠は[artifact directory](../../../artifacts/o2_a1_n3_interruption_diagnostic/2026-11-06/)、[SOURCE](../../../artifacts/o2_a1_n3_interruption_diagnostic/2026-11-06/SOURCE.json)、[evidence manifest](../../../artifacts/o2_a1_n3_interruption_diagnostic/2026-11-06/EVIDENCE_MANIFEST.json)、[人工結果](../../../artifacts/o2_a1_n3_interruption_diagnostic/2026-11-06/TESTS.json)、[保全確認](../../../artifacts/o2_a1_n3_interruption_diagnostic/2026-11-06/PRESERVATION.json)、[STOP](../../../artifacts/o2_a1_n3_interruption_diagnostic/2026-11-06/STOP.json)に保存する。
独立worktree・専用branch `codex/o2-a1-n3-interruption-diagnostic-20261106`を使用。
元main HEAD/status・3087既存ファイル、旧科学worktree、SOURCE542ファイル、旧結果/STOP/grant/ledger/marker、C1凍結は維持。
新しく追加したのは診断/人工prototypeと証拠・文書であり、旧科学SOURCEのbytesは変更していない。

専用branchへcommit/pushし、remote SHA・公開本文/blob/treeを照合してmandatory STOPする。最終SHAと公開照合記録は最終応答と独立した照合記録へ残し、commit自身のSHAを自己埋込みしない。
