# C1-5A handoff — 技術的STOP・3方式未完了

**`C1_5A_TECHNICAL_FAILURE_STOP`。Scientific attempt 1、retry 0。SOURCE修正・条件変更・再実行なし。**

1. OrdinaryとFullは各1024 path / 4096 microstepを完走した。P5は104,977群のsetup後、workerの突然終了により`BrokenProcessPool`で停止。3方式比較は未完了。[失敗・保存値報告](FAILURE_REPORT.md)
2. 完走2方式はaccepted1024、zero0。経験的weight二乗平均1.1141049662602474、最大観測weight表示1.0555117082535121は同じ。raw logical instructions/classical trialはOrdinary285.125、Full284.6494140625。固定PRNG・小標本・TAIL_ONLY_PLUS_SCALARに限定し、優位を認定しない。
3. worker batch CPUはOrdinary57.444453536秒、Full4510.117112964秒。Fullのlocal query/F3検査、既存未最適化selector、cache/並列条件を含む観測。P5 setup CPU283.087432224秒、onlineはUNKNOWN。P5欠測を0にせず総費用順位を作らない。
4. workerの実exit signal/exit codeが残っておらず、終了原因は未確定。失敗後cgroup観測のoom/oom_killは0。OOMや外部killを原因と断定しない。科学的なP5不成立でもない。[原stderr](result/closed_P5_full.stderr.log)
5. native価格、deterministic halves、state準備/readout、十分N、全support certificate、Partial/P3や強い指数task対照は今回未取得。3方式未完了のため、C1継続・縮小・終了を決める十分な比較材料とはしていない。
6. 最小後続候補はworker終了・fork/メモリ・観測設計の技術調査。不要なcertificate closureを増やす前に、ユーザーがその必要性/scopeを判断する。今回追加のstress test・新科学計算・H4再試行は行っていない。

## 次の担当と必要な承認

| 担当 | 次の作業 | 承認境界 |
|---|---|---|
| ユーザー・GPT | 未完了と完走2方式の限定データを踏まえ、原因調査や比較完成へ投資するか、縮小するかを判断 | 科学的継続判断は未実施。詳細科学レビューを開始する場合はユーザーの明示開始承認 |
| Codex | 承認されたscopeで、保存ログ/静的SOURCEから原因調査・観測改善候補を具体化。必要な人工stress検証は明示scopeに従う | 今回から自動継続しない。封印SOURCE・旧失敗を変更しない |
| ユーザー | 新しいH4実行の必要性を判断し、新SOURCE/契約/入力/計測・one-shot/STOPを明示承認するか決める | 今回のgrantは消費済み、retry 0。P5だけの再実行も新科学scope・別承認が必要 |

native synthesis、十分N、Full-N/追加Pilot/全round RPE、新証明への自動進行はない。今回の結果からresource winner、原著価値、投稿GOを認定しない。主agentだけで作業し、別reviewerによる独立受入はしていない。

## 公開・来歴

- branch：`track-b-c1-5a-h4-limited-resource-pilot-20261021`
- 基点：`af75af03c0b334c29b5a4311bd2b4578a999460c`
- SOURCE：`900a5278c604b4c58b794ca891720d9dea293919`
- 新authorization：`079ba428257444997d89ee2a975251563d8d16b0`
- 結果commit・公開URL・bytesは`publication_receipt.json`。最終receipt込みHEADは最終チャットと専用runtimeにも記録する。
- 原データ513 filesは[アーカイブindex](raw_archives/index.json)でmember単位まで照合・保全。[復元方法](STORAGE_AND_PUBLICATION.md)。raw stdout/stderrとfailure/terminalも公開する。
- [保全照合](preservation_verification.json)：SOURCE549/input45/runtime30一致。旧SOURCE/raw/grant/claim/ledger/marker/STOP、非公開原本2件、他worktreeは不変。消費済みmarkerと失敗分類を保持。

**公開・保全確認後mandatory STOP。今回の科学jobを再開しない。**
