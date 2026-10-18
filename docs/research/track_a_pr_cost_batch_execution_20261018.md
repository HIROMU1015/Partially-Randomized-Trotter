# Track A：A_COST・保存B/D会計の実行開始とH8 D結合修正

2026-10-18の利用者指示「その方針でよいので計算を開始して、中断された場合は修正してまた再計算してよい」により、準備済み9 scopeを選択して起動した。Bは元の限定診断のまま、完全な無条件費用比較へ拡張していない。新SCF/DF/state/signal/fit、他研究・他jobの変更は0。

## 起動とSOURCE

- run ID：`pr-cost-batch-20261018-run01`
- 初期SOURCE：`31ef62a12407010e9c1a07f9b37facedb8dff5fe`、起点REVIEW：`4307e86681281f14e00b8ad3fef38c40538776df`。origin branch SHA一致を確認して起動。
- supervisor PID：3223120、batch driver PID：3223122。
- 起動gate/driver log：`/home/AbeHiromu/projects/track-a-pr-cost-batch-seal/20261018/runs/run01/`
- 個別log：`/home/AbeHiromu/projects/track-a-pr-cost-batch-seal/20261018/private/sealed03/authorization/`
- A_COST：H4/H6/H8は6/6/10 worker。既知の高次PF計算と共有CPU使用のfresh観測を踏まえ、8/8/12案から調整した。既存jobのaffinity等は変更していない。
- memory admission見積りはparent込み321.5GiB。追加32GiBの余裕、空きdisk/inode、source/input/runtime/未使用output/lockを確認。RSS/AS/worker/wallの恒常上限を新設していない。
- quotaは未公開のまま、準備案を了承した利用者指示に従い、ENOSPC/EDQUOT等で当該scopeを停止する方針をgrantへ明記。quota無制限と認定していない。

## 観測時点の状態

A_COST 3件は稼働中。H4/H6 DとB3件は個別terminalで完了。H8 Dは下記の修正後に完了し、native exit0・同一親のwait/reapを確認した。Aを含む初期batch全体のnative完了receiptは、A終了後に確認する。これは全A完了・全9契約初回成功という報告ではない。

各Aはpreparation復元を済ませてown workerへ処理を割り当てている。先頭control/oracle群の成功後に残りへ進み、compile単位で予約を残す。現時点の進捗は変化するため、private outputの`task_result.json`・`measured_*.json`・STOPと最終terminalを正本とする。

## H8 Dの停止・修正・再実行

初回は`ORIGINAL_PR_MEAN_MUST_REPRODUCE`でSTOPした。原因はH8の保存通常mode `USUAL_SIGNED_SAVED`を正規modeへ対応づけず、通常PRのshot参照を非相殺側へ向けていたこと。さらに旧D0の倍率は文字列`1.0`/`10.0`、後続会計は数値であり、keyを元六軸adapterと同じ規約で正規化する必要があった。

mode対応を明示し、欠測時の他modeへのfallbackを禁止し、元の`row_key`を再利用した。通常/非相殺のsaved shot整数、元n32費用、科学条件、compiler、再現検査は変更しない。人工7限定検査がPASS。最初のmetadata修正snapshotはキー未結合で停止し、科学launch/消費0のまま保存した。

修正SOURCE：`8ff002dd9524b49284fd8fb8e11354ee1ec18a00`。新output/lock/grantは`/home/AbeHiromu/projects/track-a-pr-cost-batch-seal/20261018/runs/H8_D_repair02/`。保存3800行、40比較を完了し、適格PR 88行の元平均再現検査を通過した。native exit0・reap確認。旧STOP・予約・消費は保持し、過去のSTOPをPASSへ書き換えていない。

修正は別branch `track-a-pr-cost-batch-execution-20261018`へ固定し、実行中のAや完了済みB/H4・H6 Dは初期SOURCEの独立snapshotを保持する。再試行は利用者の修正再実行指示に基づく新attemptであり、同じgrantの再利用や無修正の自動retryではない。

## 保全と次の作業

[軽量起動・修正監査bundle](../../artifacts/resource_applicability/track_a_pr_cost_batch_execution/2026-10-18/)にsource、result hash、scope対応、検査結果、保存場所を記録した。有効grant・private config・raw NPZ/QPY・runtime本体を公開しない。

次の担当はCodex。A3系の完了/STOPとnative cleanupを確認し、中断時は今回の認可範囲で原因を修正、新attemptへ記録する。科学的不一致を検査の緩和で救済しない。完了済みscopeや他研究を無条件にやり直さず、次stage/確認samplingへ自動進行しない。
