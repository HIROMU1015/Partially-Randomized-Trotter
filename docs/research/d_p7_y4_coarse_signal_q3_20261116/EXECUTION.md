# 採択された粗幅Y4の有限実行

対象・上限は `d01c780d035eb096186ef1b8aa6d5e580bca2a50:docs/research/d_pf_first_coarser_width_plan_20261116/FIRST_FINITE_PLAN.json` と今回のユーザー指示に従う。新規6 complex mean、13round packet1件、Q3最大1件。それ以外のreference/step/回路build・compile・価格・fit・新配分・retryは0。

専用branch `codex/d-p7-y4-coarse-signal-q3-20261116` はDの計画SOURCEから分岐。元価格branch、旧SOURCE・raw・STOP・他担当worktreeは変更しない。今回のチャット指示が実行許可であり、新grantや承認証明書は作らない。

## 再利用とSOURCE

- signal：A `06a15ee45beb177c018f6260ea4a5ac8cf66cfe6` の `pf_shortlist_mean_completion.py` / `saved_pf_mean_staging.py` をbyte同一で再利用。元 `baseline_signal_pipeline.py` / `baseline_signal_numerics.py` は同一。既存Dの `rpe_optimization_pipeline.py` はPauli metadata拡張を含む版で、PFのconstructor/produceを使い、今回configuration/物理13点と元7recordのdomainを照合する。
- Q3：B `b4296b110b129d226dbef2743c14f725a27c3bf6` のcode-only snapshot（scriptsとsrc/trottertracks）をこのworktree内 `.q3_runtime/<SOURCE>` に保存。科学input/結果をsnapshotへ移さず、元 `run_pf6_cost_budget_q3_fixed` → 保存対応済みworker → canonical total binding → 並列DPを使う。元SOURCE照合関数を維持する。signalとQ3を別processにして異なる世代の依存moduleを混ぜない。snapshotは再取得可能な既存SOURCEなので成果物に二重収録せず、固定commitと既存SOURCE検査で識別する。
- new caller：`scripts/run_p7_y4_coarse_signal_q3.py`。新しい数値kernel、N/decoder、compile/価格pipelineは追加しない。

## 固定順序

1. metadata preflight、人工3 test、SIGNAL_INPUTSをcommitする。
2. `python -B scripts/run_p7_y4_coarse_signal_q3.py signal-execute` を一回。保存step・入力を一度読み、同じcontextで元C.executeの13round保存/STOP経路を使う。新規はm6〜m11だけ。36次元・6点の軽いmatrix powerは元ループを使い、process起動による複製を避ける。START/ENDを各meanで記録する。
3. 完成packetとsignal状態をcommitし、`q3-prepare` でその固定packetをcanonical total viewへbindする。元Y4 B_PRICE列・元Q3契約を使い、Q3_INPUTSをcommitする。ここでまだQ3は起動しない。
4. `q3-execute` を一回。元の全事象loss、exact τ、新signalの閉包gate、部分DP・原API・資源・子exit記録を保存する。異常時の科学retryなし。

Q3は保存benchmarkで8 workerと16 workerのwall差が小さかった既存8-worker backendを使う。固定のCPU/RAM上限は設けず、開始時に利用可能資源を観測する。このscientific scopeでの主な費用はQ3で、元monitor/協調round logとworker自身の保存処理を再利用する。

## §10.5差分

Dの計画・旧coverageとBの後続 `0b33b4afdd285a99f610e59d8f5b77960a59a0d0` を照合。後続Bは**現幅p7/S2・16h/7** の別配分の結果で、粗幅Y4のsignal/packet/同配分Q3には流用できない。新しい保存達成費用415,721,884 RZは補足として別記し、今回指定された計画時点の435,641,568 / 435,481,090 RZを置き換えない。今回coarseの総費用はnull。

関連branchの公表namespaceと同ユーザーの対象caller/physical IDを開始直前に差分照合する。新しい同等保存結果があれば再利用へ減らし、出所不明・進行中の同等jobは保留する。未参照branch/host全体の不存在は主張しない。今回使用しないbranchの通常更新を停止条件にしない。

成果はREPORT / HANDOFF_B / 原signal / source-bound packet / Q3 API・partials・数値gate・資源記録。公開・remote SHA/tree/成果物照合後mandatory STOP。適合でもcompile・別配分・費用算出へ進まない。

## 実際の完了・停止追補

信号6件はSOURCE3179680edで完了。Q3 intakeに単一SOURCEが必要なため、C.b_meanによるbinding-only view（元record/SOURCE保持）を別に固定した。Q3 code snapshotには元src/trotterlib/h46_m5_plan.pyも追加しbyte照合。Q3 SOURCE278fb5214はAcceleratorのPython3.12 AST schema guardでAPI開始前に停止した。retryなし。詳細と新しい結果資格はREPORT/HANDOFF_Bを参照し、前節の実行手順を再実行許可とは扱わない。
