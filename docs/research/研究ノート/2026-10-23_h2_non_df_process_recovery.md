# 2026-10-23 H2非DF：process分離と別one-shot取得

利用者は、前回のQiskit thread競合を技術修正し、未完了Pauli PR/DF PFのみを新契約で各1回取得することを明示承認した。
旧run01・SOURCE・消費済みgrant・失敗STOPは解除せず保全。成功済みPauli PF/DF PRは再実行しない。
人工の同じborrow例外・経路を再現し、caseごとのfresh interpreterで共有session状態を分離した。
旧科学kernel/条件/数値閾値/seed/64標本は不変。これを採用した理由は、人工で意味・費用の一致と4process同時実行を確認できたためである。
新SOURCEで未完了2caseを取得し4方式・5精度を結合、全実施gate PASS、親子exit0/全回収。
CNGは固定H2の経験的allowanceと有限MCに条件づける。未観測eventを0費用と扱わない。
詳細科学レビュー・新モデル・独立確認・H4以降は未実施。旧決定履歴を書き換えず、ローカルcommit後mandatory STOP。

[技術報告・SOURCE・表・数値限界](../../non_df_h2_process_recovery_validation.md)。
