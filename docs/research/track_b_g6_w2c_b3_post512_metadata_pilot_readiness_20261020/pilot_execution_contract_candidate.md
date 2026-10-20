# 実行契約候補：未採択・実行閉鎖

機械可読候補はpilot_execution_contract_candidate.json。4方式×A/B×X/Y、S→P1。各条件correctness256＋baseline4096＋instrumented4096、計8448、二条件16896。n256は提案で科学的十分性を認定しない。旧512は参考証拠で、候補予算から差し引かずspent scopeを再利用しない。

科学入力/precision/A6は既存のP5_general_orderを参照。新seed namespaceは旧提案PROPOSED_B3_LIMITED_G7_PILOT_20261019_V1のまま未採択。mode/layer/passを科学domainへ追加せず、global trial IDには含める案。新bit列0、IID主張false。

費用はfixed portfolioの記述的CPU/wallで、方式の普遍的順位や母平均CIではない。S/P1順序交絡・同時負荷・cacheを記録する。baseline coarse、instrumented coarse/fineの各partitionを別々にouterへreconcileし二重加算しない。親ownCPU、回収済みchild hierarchy（Git/helper含む）、worker END ownCPU、残差/未計測processを区別し、窓・marker/finalization除外範囲を明示する。CPU/RAM/wallの恣意的上限は追加しない。

retention/64GiB result/256MiB logは旧候補で正式未採択。旧512のResult38,884,103bytes/Log1,731,946bytesと費用83.526286145CPU秒/52.418725915wall秒は観測値の引用だけで、16896の時間・容量・varianceへ線形外挿しない。metadata/archive後処理と科学実行windowを混同しない。

STOPは既存原則を維持する案：source/input/runtime/manifest破損、invariant/Codec/bit/angle不整合、worker異常、実継続不能な資源不足、storage/call/byte超過、UNKNOWNで新規ENTERを止め、in-flight drainとprefix/unknown credit/no-refundを保全。無関係worktreeや一時的PSIだけで正常workerを停止しない。retry0/resume false、どの終了でもmandatory STOP。coverage遷移は採択前に確定する。

実行前条件は科学設計採択→Codexが上記差分だけのversioned release/人工受入/実値SOURCE固定→正確なmanifest採択と別の実科学one-shot承認→fresh独立scope/状態root。現候補にSOURCE release/authorization/claim/rootは発行していない。
