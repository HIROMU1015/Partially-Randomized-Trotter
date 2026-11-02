# 2026-10-19 H6 m5：1qubit保持対照の一回実検証起動

利用者の「実際にH6で改善されるか試してみて」に基づき、SOURCE68ee03a7で同じcosine一軸を新compiler policy `M5_OPT1_PRESERVE_ONEQ_RUNS_CONTROL_V1`により再compile・作用比較する実検証を起動した。小角度だけの切替APIがないため、1qubit再合成全体を保持する保守的対照とし、他のopt1 taskを維持する。小角度だけへの原因認定・旧費用互換性は主張しない。

保存lowering QPY/vector/PASSをSHA結合し、新compile1/作用1、raw3検査は順序固定・各1e-9未満（最初はreuse）。人工12件・preflight22項目・改変拒否12種PASS、2011seal・2658保護file、19:37:58 JST起動。実行前空きRAM約469.1GiB、proxy約10.29GiBで1productive workerと別監視を使用した。

前回raw基準超過STOP・58軸・AD03保留を維持。sine/sampling/会計/追加候補/retry/push0。全PASS・native exit/reapは未確認。[起動報告](../track_a_h6_m5_preserve_oneq_run01_start_20261019.md)。次はCodexの終了監査と結果報告、その後mandatory STOP。
