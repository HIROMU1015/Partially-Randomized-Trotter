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

## PR-Pauli κ探索：schedule単位の候補・共通入口のbatch・連鎖の先読みの実測（B2、2026-11-16）

- **κの物理schedule区間**（`24b51fbc3`、`rpe_kappa_schedule.py`、`kappa_neighbourhood_plan(..., sched)`）
  - 同じ13 roundのR vectorになるκを1点として扱い、次に異なるscheduleへ厳密に移る。これで、同じscheduleの重複評価を避ける。
  - **実対象で確認**：LD5では、範囲[1/2, 8]の中のscheduleが2つしかない。2点の評価で、この範囲を尽くした（旧規則では、κ=4/3・2・3が同じscheduleだった）。
  - **適用条件**：mapperが R_m=max(J, ceil(ceil(κc_m)/q_m)) であること。生成構成で、毎回の再現を検査している。
- **信号・価格のround単位の再利用**
  - 7点の新規平均値は、R_mが変わった16 roundだけで、全91 roundのうち75 roundを再利用した。
  - 価格は、Cの共通入口（`02e7cc6a1`）のbatchで26軸を再利用し、較正は0件。入口は、同じ較正cellを1件にまとめる。
- **連鎖の先読みの実測**（実runでの初めての使用）
  - Q3は284件で、平均145秒。同時実行は平均10.1件。
  - 採択82件で、V2は72分（逐次なら約198分）。
  - **失敗例**：適合と不適合が交互に出る下降では、先読みの71%（202件）が採択されなかった。
- 根拠：[kappa_abs_round04/REPORT](research/rpe_method_optimizer_h4_20261116/kappa_abs_round04/REPORT.md) §2・§5、検査 `kappa_abs_round04/checks/KAPPA_SCHEDULE_CHECKS.json`（7/7）。

## PR候補準備の再利用と方式間並行（D2、2026-11-16）

- `pr_candidate_completion.py` は既存V2が選んだ不足候補を方式別に並行準備し、元の同一Executor・累積予算へ戻す。候補順/shot規則は再実装しない。
- 信号は既存の物理key、価格は既存PriceRules、κ宣言aliasは元B2登録を再利用する。DFの同一分割descriptor・保存cohort/個別axis receiptを共有し、欠測軸だけ取得する。完了Q3は元job keyで再利用する。
- 実対象でPauli LD4/DF LD6の準備→Q3→C同期と完了runの新規Q3 0件を確認。部分axis再開・後続候補継続は人工検査。速度倍率は測定していない。
- 失敗例：最初のdescriptor envelope不足は科学call前に停止。DF supervisorの終了receipt形式をPauli形式として読んだ誤りは、保存exitcode/identityをcollectして解決し、再Q3していない。
- SOURCE・適用条件・検収：[API](research/d_pr_candidates_auto_20261116/API.md)、[REPORT](research/d_pr_candidates_auto_20261116/REPORT.md)。曖昧な稼働jobや失敗leafをretryせず、既存科学STOPを保持する。

## PR-Pauli広域schedule探索：粗い探索の優先・HYBRID並列・価格下界による打ち切り（B2、2026-11-16）

- **価格下界による打ち切り**（`run_rpe_schedule_search.py` の `prune`、`rpe_schedule_search.plan_wide(price_floor=)`）
  - 候補scheduleのMODEL価格で、1 shotずつの配分の費用を計算する。これは、そのscheduleで最も安い配分の費用になる。
  - この値が照合済みの全体最安以上なら、信号・Q3の前に除外する。価格はRとともに増えるので、それより大きいκも同じ理由で除外する。
  - 実対象（round05の初回選択、メタデータのみ）では、37件中7件を除外した。
- **粗い探索の優先**（V2の `coarse_gain_fraction`）：1%未満の削減を後回しにして、別のscheduleの評価を先にする。人工系で確認した。
- **並列枠**（V2の `lookahead_mode`：INDEPENDENT／HYBRID／CHAIN）
  - 保存結果の再生（時刻モデル、32枠、同時に調べるschedule 7つ）の結果：

| 方式 | Q3起動 | 実時間 |
|---|---:|---:|
| 独立候補のみ | 355 | 704分 |
| 連鎖 | 1,898 | 383分 |
| HYBRID | 1,799 | 395分 |

  - **失敗例**：調べるscheduleが少ないと、独立候補だけでは枠が埋まらない。
  - 次roundではHYBRIDを使う。多数のscheduleを同時に評価する実runでの効果は未検証。
- **全完了Q3の照合と再利用**（`build_rpe_q3_ledger.py`）：4,783件を照合し、拒否は0件。2種類の最安を分けて出す。
- 根拠：[kappa_wide_round05/REPORT](research/rpe_method_optimizer_h4_20261116/kappa_wide_round05/REPORT.md) §1・§3。


## H4 RPE探索：D2共通queue・DP prefixをB2通常入口へ接続（2026-11-17）

D2公開 `1eda4ba3f` / `0742a750b` のhelperを再利用し、`run_rpe_schedule_search.py` → 通常round CLI → workerへ接続。実装版は [SOURCE](research/h4_rpe_search_throughput_20261116/SOURCE.json)。`executor.scheduler="concurrent"`、`dp_workers`、`prefix_cache_directory`、`prefix_checkpoint_rounds` を指定する。同じsignal/contract/runtime/scopeのprefixだけが対象で、元AST guard・MSE/残余会計を保持する。新配分はQ3予算1件。元のschedule集約・価格下界・HYBRIDは再実装しない。

保存4,783起動ではm11の完了済みdonor機会532件、m5+m11は3,761件。ただし完全runtime/frontierが旧記録になくHIT実測ではない。人工3配分列はm5+m11でwall3.927→2.522秒/CPU3.926→2.519秒、110,437bytes、DP全出力一致。m2単独は機会が多くても0.987倍。同時coldの重複計算を消すものではない。実H4での速度・容量は未測定。

`adaptive_hybrid`、round間shot振替、`overlap_ready_batch` はopt-in。適応先読みは模擬起動数を減らしてもlatencyが増える例があり、一律に有効化しない。通常・再開経路で発見したfamily引継ぎ漏れ、未実行planの再開、prior重複、live終了待ち、準備中失敗のSTOP保存、V2全体previewの未起動予約残りを小さい人工検査で修正した。観測counterは再開でresetするが完了結果/起動済み予算は維持する。

[REPORT](research/h4_rpe_search_throughput_20261116/REPORT.md)、[B2設定例・開始/再開・制約](research/h4_rpe_search_throughput_20261116/HANDOFF.md)、[検査](research/h4_rpe_search_throughput_20261116/checks/VALIDATION.json)。通常入口はplan-only開始/同run再開で科学0を確認し、4方式の共通経路は人工workerで検査した。保存運用/再生の効果はPR-Pauliの範囲。今回の実科学0件。
