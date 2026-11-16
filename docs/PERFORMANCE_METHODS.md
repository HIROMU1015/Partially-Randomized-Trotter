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
