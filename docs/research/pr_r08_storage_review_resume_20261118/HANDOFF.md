# R08の容量対策・レビュー改善版への継続

計算コードは `e433938dd39ac4c2cb6b7ab8865e41c0b996d232`。GPUA100-gleap02 の既存 `codex/rpe-prefix-cache-storage-20261118` worktreeで、R08の第14サイクルまでの保存状態を `run01` に引き継いだ。元入力の H/state、Q3契約、要求幅、κ探索範囲、方式別Q3予算、200サイクル上限を継承する。[照合と保留](PREFLIGHT.json) ・ [固定ROUND](ROUND.json)。

元runの正常終端Q3はPR-Pauli 64件、PR-DF 47件。全て元のrunから通常のidentity gateに渡す。書込中断の17件ではprocess終端・完全な結果を確認できなかったため、対象配分だけを保留する。前回の完了準備・価格証拠は元SOURCEのまま使い、準備途中2件の扱いは既存handoverの保留規則に従う。

RomuTailの直近公開snapshotの探索範囲と重なる分割は、その状態を直接確認できるまで後続投入を保留する。今回進めるのはPR-Pauli LD1・2・3・8・9とPR-DF LD1～6。保留側の記録も元runに残る。[RomuTailの公開scope・状態](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/1209b16bb771e03d8a3ef07513db0b028a39bf57/docs/research/romutail_controller_runs/search-be59f46fceee86257816)。

通常入口で自価格下限、未試行family優先、最安近傍の交換、同一Pauliモデルに限るκ下限、Q3残差並列化、PR-DF X/Y並列化を有効にした。Q3出力の圧縮と、新規ローカルprefix cacheの40 GiB上限も適用する。元prefix cacheは所在がなく、新版は空のcacheから蓄積を始める。

同じrunを再開する前に `tmux has-session -t '=h4-pr-r08-storage-review'` を確認する。起動・再開は両方とも `bash docs/research/pr_r08_storage_review_resume_20261118/START_OR_RESUME.sh`。元runと新版の所有lockを取得し、既存の停止判定とサイクル上限を維持する。進捗は `run01/DRIVER_PROGRESS.jsonl`、`run01/PROGRESS.jsonl`、`run01/CONTROLLER.log` に保存される。
