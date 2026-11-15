# Bの `$rpe-cost-compare` Skill経由で行ったAの受入テスト

想定読者：B（Skill）、B2（最適化プログラム）、統合担当。
- 受入ケースの定義：[../ACCEPTANCE.md](../ACCEPTANCE.md)
- 結果の正本：[SKILL_ACCEPTANCE_RESULT.json](SKILL_ACCEPTANCE_RESULT.json)
- Skillの出力形式の表：[SKILL_OUTPUT_TABLE.md](SKILL_OUTPUT_TABLE.md)

## 冒頭の答え

| 項目 | 内容 |
|---|---|
| 結果 | **PASS 38、FAIL 0**。前回PENDINGだったSkill経由の受入は、3ケースとも合格 |
| ① p7 S2限定の範囲遵守 | 3 runすべてで、探索候補・各step・次候補・待ち行列がp7 S2の範囲内だった。保存の4 shot点・3 shot点から始まり、規則による停止と具体的な次候補を返した |
| ② 粗幅Yoshida4 372,035,410 RZ | 保存Q3とDの価格から、方式の最安点として **372,035,410 RZ（exact）** を返した（`dd8bfc24`）。新規call 0件、job起動 0件。範囲を32h/5に限っても同じ |
| ③ 人工workerでの予算終了と再開 | 予算2で `Q3_BUDGET_EXHAUSTED`。最安点・次候補・価格待ちの候補を別々の欄で返した。同じround fileの総上限を4に上げ、同じコマンドで再開した。完了済みjobの再起動は0件で、前回の次候補から続いた。もう一度実行しても起動は0件 |
| 科学計算 | 実Q3・実信号・価格・compileは0件。起動したjobは、B2の印付き人工worker（`artificial` 経路）の5件だけ |

## Skill経由の実行方法

**使ったSOURCE**

| 対象 | commit |
|---|---|
| Skill | `19252baf39698570bd80a98bf10142cc8f215517` の `.agents/skills/rpe-cost-compare/` |
| program（B2の固定版） | `a2e2ef50a108d6f34089a99bd00f11a7085dd9c9` |

- Skillの各fileのsha256は、結果JSONに記録した。
- programのcommitは、Skillの `program-entry.md` とBのHANDOFFが指定する実行SOURCEである。
- programは、自分で作った独立checkoutで動かした（detached worktree `a-skill-acceptance-b2-a2e2ef5`）。runner・optimizer・Q3 job adapter・problemは、B2公開版 `a4b4d0e1f` とbyte一致した。実行後もcheckoutに変更は無い。

**Skillの読み方**
- Claude Code（A）はCodexの `$skill` 登録機構を持たない。そのため、`SKILL.md` と `references/` をそのcommitから読み込み、書かれた手順どおりに実行した。
- Skillは手順書であり、計算codeを含まない。実際の入口は、Skillが指定するB2のCLIである。

**手順の各段階**（runner：[`scripts/acceptance_rpe_cost_compare_skill.py`](../../../../scripts/acceptance_rpe_cost_compare_skill.py)、SOURCE `142db3001`）

1. **入口の照合**：program checkoutのHEADが、Skillの実行SOURCEと一致することを確かめた。文書のコマンドとCLIの `--round`／`--run-dir` が存在することも確かめた。
2. **入力の変換**：利用者の指定を、Skillの表どおりB2のround file（`RPE_OPTIMIZER_SEARCH_ROUND_V1`）にした。元はSkill同梱の `h4-round-budget0.json` である。
   - `round_id`
   - `methods`：TRUNCATED_PF
   - `max_new_q3_per_method`：方式別の総上限
   - `initial`：開始時に既知とする保存点
   - `method_overrides`：探索範囲
   - LDやshot列は利用者に尋ねず、programに選ばせた。
3. **実行**：文書のコマンドを、program rootで実行した。新しいroundには、毎回未使用のrun directoryとround_idを使った。
   ```
   PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src <python3.12> -B scripts/run_rpe_method_optimizer.py \
     --problem configs/rpe_method_optimizer/h4_df10_problem.json --round <round.json> --run-dir <run>
   ```
4. **再開**：同じround file・round_id・run directoryで、同じコマンドを再実行した。予算の追加は、同じround fileの総上限に書いた。
5. **判定**：Skillが読むように指定するfileで判定した。
   - `ROUND_RESULT.json` の `methods`
   - `NEEDS.json`、`INVOCATIONS.jsonl`
   - `jobs/`：LAUNCH・ARTIFICIAL_RESULT・OUTCOME

programの起動は計9回で、各約6.5秒、全体で約1分だった。

## ケース別の結果

### ① p7 S2限定（formula 2nd、幅16/7）

| run | 開始点（initial） | 予算 | 経過 | 停止理由・次候補 |
|---|---|---|---|---|
| 1a | 4 shot点（435,481,090） | 0 | m12を−2：保存Q3は適合（395,962,678、改善）。さらに−1：保存Q3は不適合（376,203,472） | `NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET`。次は386,082,768（実経路 `pf_b_pf6`） |
| 1b | 4 shot点＋3 shot点（415,721,884） | 0 | −2：保存Q3は不適合（376,203,472） | 同上。次は395,962,064 |
| 1b | 同上 | 1（人工worker） | 上の要求を尋ねずに人工workerで実行。人工の答えは適合で、最安が395,962,064に更新 | `Q3_BUDGET_EXHAUSTED`。次は356,443,652 |

| check | 結果 |
|---|---|
| `search_space_follows_requested_range`（3 run） | PASS。NEEDSの候補は `a6e55777` の1件だけ。trace・次候補・待ち行列も範囲内 |
| `starts_from_given_saved_point_only`（3 run） | PASS。最初の最安点は、与えた保存点（1a：435,481,090、1b：415,721,884） |
| `saved_Q3_revealed_only_after_selection`（3 run） | PASS（出力から観測できる形）。programが知っている保存結果は、与えた開始点と、各stepで選ばれた配分だけだった |
| `next_action_decided_without_asking`（3 run） | PASS。規則による停止。次候補にはconfiguration・26軸・費用・経路がある |
| `continues_after_improvement`（1a） | PASS。改善の後にもう1 stepを取り、さらに安い要求を返した |
| `next_action_from_m12_3_state_without_budget`（1b、予算0） | PASS |
| `remaining_budget_used_then_rule_stop`（1b、予算1） | PASS。起動1件（`artificial`）、worker成果1件、その後予算の規則で停止 |
| `budget0_launches_nothing`（予算0の2 run） | PASS。job directoryは0件、INVOCATIONSのlaunchedも0 |

`saved_Q3_revealed_only_after_selection` の注記：
- Skillの入口（CLI）は、call単位の応答記録を出力しない。
- そのため、ここで確かめたのは、出力に現れた既知の結果の範囲である。
- 応答器を直接記録する検査は、Python APIでの受入（`311571224`）で行っている。B2がa2e2ef50aで再実行し、PASSしている。

### ② 粗幅Yoshida4の価格反映（予算0）

| check | 結果 |
|---|---|
| `price_input_is_D_saved_file` | PASS。固定problemのC catalog INPUTS（`417a73e73`）のsourcesに、Dの価格 `c7c8c1edd`（`PRICE_AND_COST.json`）が含まれる |
| `coarse_Y4_ready_with_own_26_axis_price` | PASS。`dd8bfc24` は準備完了で、不足は無い。`SAVED_UNIT_PRICES_REPRODUCE_ALL_SAVED_TOTALS` |
| `saved_Q3_reused_without_new_call`・`budget0_launches_nothing` | PASS。保存Q3の2点を再利用した：A_SHAPE（適合、372,035,410）と粗い点（不適合、365,218,506）。新規call 0、job 0 |
| `D_price_total_in_comparison_table` | PASS。比較の行と方式の最安点が、ともに **372,035,410 RZ（exact）** |
| `next_action_decided_without_asking` | PASS。次候補は同じ構成でm12を6→4にしたもの（329,694,342 RZ）。予算外の要求として返した |

- p7全体の範囲でも、32h/5だけに限った範囲でも、同じ結果だった。
- 価格を与える前の状態（catalog `08c6a694b`）は、今回のSkill経由では実行していない。固定problemはすでに価格入りのcatalogを指しており、problemを書き換えると固定版の入力を変えることになるためである。価格前の判定は、APIでの受入でPASSしている。

### ③ 人工workerでの予算終了と再開

**設定**
- 範囲：TRUNCATED_PF・2nd・幅1〜16/7。この範囲に次の3候補が入る。
  - p7 S2（準備完了）
  - 2nd・1h（`e6754564`）：価格待ち
  - 2nd・2h（`155a3a2e`）：信号と価格の待ち
- 開始点：4 shotの保存点。
- 人工worker：B2の印付き `artificial` 経路。parameterは、B2の実行器受入（`a4b4d0e1f`）と同じ値を使った。

| 段階 | 結果 |
|---|---|
| 1回目（予算2） | 保存Q3で2 step（m12=2 適合、m12=1 不適合）を進めた。その後、人工workerで2件を実行した（386,082,768 適合、366,323,562 不適合）。`Q3_BUDGET_EXHAUSTED` で停止。最安は386,082,768（人工）、次候補は376,202,858。価格待ちの `e6754564` は、`needs=['PRICE_26AXIS']` として別欄に出た |
| 再開（同じround fileの総上限を4にし、同じコマンド） | 完了済み2件のLAUNCH・成果・OUTCOMEはbyte不変。新しいjobは2件で、最初の1件は前回の次候補そのもの。そのときの最安点は前回の最安点（386,082,768）だった。最安は376,202,858に更新し、再び予算で停止 |
| 3回目（同じコマンド） | 新規起動0件。最安は同じ。INVOCATIONSの予算は2→4→4 |
| 範囲を変えて同じrun directoryで実行 | programが `Run directory belongs to a different approved round (method_overrides)` で拒否した。jobもINVOCATIONSも増えなかった |

| check | 結果 |
|---|---|
| `budget_end_returns_best_unevaluated_and_reason_separately` | PASS |
| `no_science_call_beyond_budget` | PASS。起動2件、worker成果2件、すべて `artificial` |
| `resume_input_recoverable_from_output` | PASS。`resume_records` に、人工の2件のconfiguration・26軸・判定・MSE上下限がある |
| `resume_does_not_repeat_completed_calls` | PASS。ROUND.jsonは初回の宣言（予算2）のまま不変。現在の上限はINVOCATIONSで読む（Skillの記載どおり） |
| `resume_starts_from_previous_state` | PASS |
| `idempotent_rerun_launches_nothing` | PASS |
| `changed_round_contract_refused_in_same_run_dir` | PASS |

## Skill・B2への所見（不合格ではない）

1. **`routes_override` がSkillの表に無い**：人工workerを使う検証は、B2のround schemaの `routes_override` で行った。Skillのround欄の表（`program-entry.md`）にはこの欄が無い。科学実行の経路を差し替える欄でもあるので、「検証用の人工経路に限る」と表に注記するとよい。
2. **応答の記録が出力に無い**：「保存Q3を選択の後にだけ返す」は、CLIの出力からは間接的にしか確かめられない。call単位の記録（どの配分を、いつ、どこから答えたか）を `ROUND_RESULT` かjob directoryに出せば、Skill経由でも直接確かめられる。
3. **価格前の状態**：固定problemのままでは価格前の状態を再現できない。価格待ちの判定は、③の `e6754564` で確認した。

## 範囲と保全

- 保存結果と人工workerだけを使った。実Q3・実信号・価格・compileは0件。
- B・B2・C・Dのcode、原結果、旧STOPは変更していない。program checkoutは、実行後もtracked変更なし・未追跡fileなし。
- 人工workerの結果（`ARTIFICIAL_JOB`）は科学結果ではない。①1bと③の最安点のうち、人工の答えによるものは、検証用の値である。
- 検査の性質：作成者（A）自身による受入テストで、独立レビューではない。判定関数が想定した失敗を検出することは、`tests/check_acceptance_rpe_cost_compare_skill.py`（3件PASS）で確かめた。
- mandatory STOP。
