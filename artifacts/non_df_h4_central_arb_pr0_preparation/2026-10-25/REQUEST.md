# H4 Pauli PR0 cosine Arb包含評価の準備：承認境界

2026-10-25。基点branch `codex/non-df-h4-central-arb-pf-cosine-20261025`、
commit `78d7f24c80b500fc63c908efc768e683406571b3`。

利用者はPR index0 cosineの個別C-targetとcompiled回路の包含接続について、
既存コード確認・技術実装・人工2〜3qubit検証・実行準備だけを承認した。
保存32eventとC SOURCEに従いretained/product/rotation/符号/位相/順序とexact operandを保持する。
law meanやR0の経験的点値を保証区間に代用しない。共通raw ray/ν、wire/control、B_saved一度だけを維持する。

> 今回は実H4のArb状態作用を実行しないでください。

新科学grant発行/消費、compile/sampling/Hamiltonian/state生成、N_G、PR1/sine/R2拡張、正式certificate受理は禁止。
次回pilot案はPR0 cosineのみ、64→96→128bit、各bit C-target1＋compiled1、最大6状態作用。
保存ρ予算約2.593611368819805e-6との関係が決まった時点で終了、128bitでも判定不能ならINCONCLUSIVE。
幅1e-8は参考目標。**契約は無効のまま保存し、数値実行には別途明示承認が必要。**

既存SOURCE/結果/grant/ledger/marker/STOPを保全、再利用優先、資源の恣意的上限なし、共有環境/他ユーザーへの変更禁止。
専用branchにcode/tests/無効契約/報告書をcommit/pushしremote SHAを確認する。
準備完了後mandatory STOP。技術的な修正・人工再試行は許可範囲内で実施できる。
