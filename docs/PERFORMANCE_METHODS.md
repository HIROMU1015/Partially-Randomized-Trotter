# 性能対策の再利用索引

既存の大規模計算の手法集は固定 `481c4b7abffbf070347f697980f2ec051e6a6538:docs/PERFORMANCE_METHODS.md` を参照する。本C branchへその科学runnerを取り込んだ意味ではない。

## 保存RPE結果の読取

`src/trottertracks/saved_rpe_cost_catalog.py:Reader.document` は既存Git blob readerを使用し、固定commit/pathをprocess内でcacheする。同じSUMMARY/INPUTS/価格表が複数HANDOFF・配分から参照されてもJSONの再読取を省く。seed/計算順/価格値は変更せず、適用対象は保存JSONのmetadata処理だけ。

既存B2の保存summarizer同様、Python process内の整数文字列桁上限を解除し、exact MSEの長い整数を損失なく読み書きする。共有環境設定や科学runtimeを変更しない。

[再利用・検査・実行範囲](research/h4_saved_rpe_cost_catalog_20261116/HANDOFF.md)と[検収](research/h4_saved_rpe_cost_catalog_20261116/TESTS.txt)を参照。今回の固定10入力の実処理時間はCATALOG.jsonに保存。人工＋公開値の読取で確認したが、速度倍率や将来の大入力性能は測定していない。

## RPE最適化器：並列実行・先読み・候補選択・前回roundの再利用（B2、2026-11-16）

- **構成別scopeの並列実行器と先読み**（`1e88cd85c`、`rpe_method_optimizer.Executor`）
  - Q3 jobを `max_concurrent_jobs` 個のslotで同時に走らせる。閾値に近いときだけ、規則の順序で最大 `lookahead` 件を先に投入する。
  - 答えは規則の順に1件ずつ使うので、探索の軌跡は逐次実行と同じになる（人工系の受入で確認）。
  - **実測**：κ round01（18 slot、9構成）では、8時間11分で4,229件、約540件/時、1件約105〜165秒。
  - **失敗例**：先読みのうち895件（21%）は採用されなかった。閾値付近では、slotが先読みで埋まって止まる時間がある。
- **候補選択V2**（`cbe2947ad`、`rpe_priority_search.py`）
  - 方式ごとに1本の優先列で、最安を下げる候補を先にする。不適合点（MSE/τ ≥ 1+1e−4）以下の配分は後回しにする。
  - **保存ログでの再生**：停止時最安に届くQ3は、4,148件（旧の実際）から218件になった。旧が作ったログに条件付けられた比較で、未評価は6件。並列での効率は未測定。
  - **適用条件**：MSE上界は、shotについて厳密には単調ではない（|MSE/τ−1|<1e−7で27組）。後回しにしても、候補は落とさない。
  - 根拠：[kappa_round01/REPORT](research/rpe_method_optimizer_h4_20261116/kappa_round01/REPORT.md) §3・§4。
- **前回roundの再利用**（`cbe2947ad`、`Executor.prior_run_dirs`／`prior_records`）
  - 同じjob key・同じ経路の完了jobを、identity照合の後に知識へ読み込み、再起動しない。実対象で4,229件を読み込んだ。
  - **途中終了したroundの記録**：`--interrupted` で、完了jobだけを使って規則を再生する。先読みとREQUEST-onlyの件数は分けて記録する。

## RPE最適化器：V2の並列先読み（B2、2026-11-16）

- **有限予算での先読み**（`5d9d94e12`）と、**連続不適合に応じた深さ**（`26d84d90a`）
  - 答えはV2の順に1件ずつ使うので、採用される軌跡は逐次実行と同じになる（人工系で確認）。
  - **失敗例1**：round02では、既定の上限40手でbatchが止まっていた。予算で終わるように直した。
  - **失敗例2**：適合が続く1構成だけを下降している間は、先読みの75%が無駄になった（起動166件、採用40件）。深さを適応させると、今度は同時実行が平均2.2件に下がった。
- **連鎖の先読み**（`a58a0f31f`、`rpe_priority_search.chains`）
  - 直前の評価が適合した構成について、次も適合すると仮定し、その次の最安点から出る候補を最大 `lookahead` 手先まで投入する。
  - **人工系**（2秒のworker、予算16）：軌跡は逐次と全長で一致した。実時間は42.1秒から25.8秒、job数は16件から55件。
  - **実runでは未使用**：round03は、実装の前に規則どおり停止した。
- **前回までの評価の再利用**（`prior_run_dirs`）
  - round03では、round01の4,229件とround02の185件を読み込んだ。
  - **失敗例**：経路の登録表key（`configurations_file`）があると、前回の経路と別物と判定されて読み込めなかった。照合でこのkeyを無視するよう直し、回帰検査を加えた。
- 根拠：[kappa_abs_round03/REPORT](research/rpe_method_optimizer_h4_20261116/kappa_abs_round03/REPORT.md) §6、実行器受入 21/21。

## PR候補準備の再利用と方式間並行（D2、2026-11-16）

- `pr_candidate_completion.py` は既存V2が選んだ不足候補を方式別に並行準備し、元の同一Executor・累積予算へ戻す。候補順/shot規則は再実装しない。
- 信号は既存の物理key、価格は既存PriceRules、κ宣言aliasは元B2登録を再利用する。DFの同一分割descriptor・保存cohort/個別axis receiptを共有し、欠測軸だけ取得する。完了Q3は元job keyで再利用する。
- 実対象でPauli LD4/DF LD6の準備→Q3→C同期と完了runの新規Q3 0件を確認。部分axis再開・後続候補継続は人工検査。速度倍率は測定していない。
- 失敗例：最初のdescriptor envelope不足は科学call前に停止。DF supervisorの終了receipt形式をPauli形式として読んだ誤りは、保存exitcode/identityをcollectして解決し、再Q3していない。
- SOURCE・適用条件・検収：[API](research/d_pr_candidates_auto_20261116/API.md)、[REPORT](research/d_pr_candidates_auto_20261116/REPORT.md)。曖昧な稼働jobや失敗leafをretryせず、既存科学STOPを保持する。
