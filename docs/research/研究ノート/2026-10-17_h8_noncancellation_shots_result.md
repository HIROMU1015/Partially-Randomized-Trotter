# 2026-10-17 H8 経験的非相殺bias・shot診断：run01完了

利用者の明示承認に基づき、SOURCE `cbfd7cacad8a5828c0dbfd284a99ef0871860e66`・plan v2 `002f9edb…` の `NONCANCELLATION_SHOTS` stageだけを新しい一回限りgrantで実行した。native exit0、試行1・修正0・再実行0。
保存通常会計1,900行を凍結D0 ruleで完全再現し、PR6候補の成分・端点・forward・referenceの84検査がPASS。1,900行・380軸を評価し、未解決0。
非相殺条件ではε=.05で16候補（旧32の3、追加152の13）が通常の適格から不適格に変わった。ε≤.01と全PR6の適格性は通常と同一。両条件で適格な行のshot比は中央値1、最大46.2。
経験的感度診断であり厳密bias保証ではない（null）。PR4次Taylor層は未観測のまま。費用比較・ランキング・PR削減率・B2再実行・新signal/compile/sampling/fitは0。
並行B2（CPU1〜4）と重ならないCPU8〜63で実行し、PSI増分0、B2 process不変。元worktreeと保護fileは不変。
[結果](../track_a_h8_noncancellation_shots_result_20261017.md)。次はユーザーが結果を確認し、費用比較stageへ進むかを判断する。mandatory STOP。
