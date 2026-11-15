# B2最適化プログラム（5b7c07679）への受入ケースの実行結果

想定読者：B2・B・統合担当。受入ケースの定義は [ACCEPTANCE.md](ACCEPTANCE.md)、結果の正本は [result_b2_5b7c07679/ACCEPTANCE_RESULT.json](result_b2_5b7c07679/ACCEPTANCE_RESULT.json)。

## 冒頭の答え

| 項目 | 内容 |
|---|---|
| 接続した入口 | B2が公開した `rpe_method_optimizer.prepare` と `optimize`（`5b7c07679`、`codex/b2-rpe-method-optimizer-20261116`）。本branchはそのcommitから分岐した |
| 結果 | **PASS 15、FAIL 4、Skillは未公開のためPENDING 1** |
| 人の指示なしで探索が進むか | **ケース1では進む。** 4 shotの点から始めると、プログラム自身が配分を選び、保存Q3はその選択の後にだけ返った。改善の後も止まらず、最後は予算の規則で具体的な次の要求（configuration・26軸・費用・Q3経路）を返した。3 shotの状態から、予算0なら要求を返し、予算1なら尋ねずに使い切った |
| 直すべき点 | 下の4件 |

**FAILした4件**

| # | check | 観測されたこと | 直し方の案 |
|---|---|---|---|
| 1 | 探索範囲の遵守（ケース1） | p7 S2（2nd、16/7）に範囲を限っても、catalogにあって索引に無いconfiguration（Y4 16h/3、M5 16h、粗幅Y4 32h/5）が、範囲の外なのに候補に入った。1aの最後の要求は、範囲外のM5の生成seedだった | `enumerate_candidates` の `outside_index` の候補にも、公式・幅の範囲を当てはめる |
| 2 | Dの価格を与えた後の比較（ケース2） | 価格入りのC catalog（`417a73e73`）を与えると、`prepare` が `Unsupported saved index schema: c7c8c1edd…PRICE_AND_COST.json` で止まった | Cの新しいcatalog reader（`be5326bdc`）を取り込む。単価の欄 `unit_price_RZ` も読めるようにする（下の補足） |
| 3 | 予算を超える呼出し（ケース3） | 予算2回に対して、応答器は3回答えさせられた。3回目の答えは捨てられた | `run_method` で、予算を確かめてから `backend.answer` を呼ぶ。実際のQ3実行器では、捨てられた3回目が未承認の科学callになる |
| 4 | 出力からの再開（ケース3） | 出力のtraceに `MSE_lower` が無く、出力だけからは再開入力（`ingest` の形）を作れない。evaluatedの行にもMSEの上下限が無い | traceとevaluatedに、`configuration_id`・`MSE_lower`・`MSE_upper` を出す。または再開用の結果fileを書く |

**補足：FAIL 2の原因の調べ**（作業用のcopyでの試行。受入結果には含めていない）
- Cの `saved_rpe_cost_catalog.py` を `417a73e73` の版に差し替えると、catalogは粗幅Y4のA_SHAPE点を「26軸価格あり・372,035,410」として返す。この版では、TRUNCATED_PFの最安値も372,035,410になる。
- ただし、その次の段階で、B2の `resolve_unit` が `No unit price in source row` で止まった。Dの価格の行は単価を `unit_price_RZ` という欄に持ち、`UNIT_KEYS`（`C_ref_exact`・`C_ref`・`C`）に含まれていないためである。
- 両方を直せば、ケース2の残りのcheckが評価できる見込みである。

## ケース別の結果

### ケース1：p7 S2のshot探索

| run | 内容 | 結果 |
|---|---|---|
| 1a（4 shotから、予算0） | step1：m12を−2（→2 shot、保存Q3は適合、395,962,678、改善）。step2：m12を−1（→1 shot、保存Q3は不適合）。その後、新しいQ3が必要になり、予算外として停止。次の要求は範囲外のM5の生成seed（380,353,796） | 範囲以外のcheckはPASS |
| 1b（3 shotから、予算0） | step1：m12を−2（→1 shot、不適合）。その後、m11を−2した配分（395,962,064）を要求して停止。保存されたm12＝2の結果は、選ばれていないので開示されなかった | PASS |
| 1b（予算1、人工の答え） | 上の要求を尋ねずに実行し（人工の答えは適合）、最安点を更新した。その後、予算の規則で止まり、次の要求（376,202,858）を返した | PASS |

- 4→3→2→1という順序は、合格条件にしていない（実際の順序は4→2→1）。
- 保存Q3は、すべて選択の後にだけ返った。応答器への問い合わせの列は、選んだ配分と最後の要求に一致した。

### ケース2：粗幅Y4 A_SHAPEの価格待ち

- 価格の前は、すべてPASS。
  - configuration `dd8bfc24…` がnot readyで、不足は `PRICE_26AXIS` だけ。
  - 保存Q3（適合）は保持され、出力の未評価候補に価格待ちとして載った。
- 価格を与えた後：入力にDの固定価格fileが含まれることは確認した（PASS）。ただし、`prepare` が上のFAIL 2で止まったため、Q3の再利用と372,035,410の反映は確かめられなかった（FAIL）。

### ケース3：予算の終了と再開（人工）

- 予算2回で止まったとき、停止理由（`Q3_BUDGET_EXHAUSTED`）、確認済みの最安点（適合、988）、次の要求（940）、価格待ちの候補Bとその不足が、別々の欄で返った（PASS）。
- 予算を超える呼出し：FAIL 3。
- 出力だけからの再開：FAIL 4。
- 完了した結果を渡して再開すると、完了済みの配分へのcallは0で、前回の要求から続いた（PASS 2件）。

## 既存のテストとの照合（追加したのは不足分だけ）

| 確かめたいこと | B2の既存テスト・CLIのcheck | 今回追加したcheck |
|---|---|---|
| 改善の後も続く | `test_loop_continues_after_improvement`（人工）、CLIのREPLAY `next_from_updated_state` | 保存したp7 S2の4 shot・3 shotの状態で、問い合わせの記録とともに確かめる（1a・1b） |
| 保存Q3を選択の後にだけ返す | なし | `saved_Q3_revealed_only_after_selection` |
| 残りの予算から次の行動を決める | `test_saved_answers_…`（予算0）、`test_loop_…`（予算2）。いずれも人工 | 保存した3 shotの状態で、予算0と1（1b） |
| 探索範囲の遵守 | なし | `search_space_follows_requested_range` |
| 価格の不足を特定する | `test_not_ready_candidates_are_never_searched`（人工）、NEEDS.json | 保存した粗幅Y4で、不足と出力を確かめる |
| 価格を与えた後のQ3再利用と比較表 | なし | `saved_Q3_reused_without_new_call`、`D_price_total_in_comparison_table` |
| 最安点・未評価・停止理由を分けて返す | 出力の欄はある（直接のcheckは無い） | `budget_end_returns_…_separately` |
| 予算を超えて呼ばない | なし | `no_science_call_beyond_budget` |
| 再開で完了済みcallを繰り返さない | INGESTのdemo（次の要求が変わるだけ） | `resume_does_not_repeat_completed_calls`、`resume_starts_from_previous_state`、`resume_input_recoverable_from_output` |

## Skill（B）について

- Bのskillは、確認した範囲（B・B2のbranchと、localの `codex/rpe-cost-compare-skill-20261116`）には、まだ公開されていない。
- 同じケースの入力をSkillから与え、同じ判定を出力に当てることを期待動作として定めた（`PENDING_NOT_PUBLISHED`）。
- 現在のCLI（`run_rpe_method_optimizer.py`）は、固定のdemo（FULL・INGEST・REPLAY・FIXTURE・OWN_PRICE）を実行するだけで、ケースの入力（開始時に既知の点、予算、応答器、再開用の結果）を受け取れない。Skillから受入ケースを流すには、少なくとも次をCLIまたはproblem fileで指定できる必要がある。
  - 開始時に既知の点
  - 新しいQ3の予算
  - 再開用の結果

## 実施・保全

**作業の場所**
- branch `codex/a-rpe-optimizer-acceptance-20261116`（B2の `5b7c07679` から分岐）。
- 新しいfile：`scripts/acceptance_rpe_method_optimizer.py`、`tests/check_acceptance_rpe_method_optimizer.py`（4件PASS）、本directory。
- B2の既存テスト11件も、この環境でPASSした。

**範囲**
- 保存結果と人工fixtureだけを使った。新しい信号・価格・compile・Q3は0件。
- B2・C・Dのcode・原結果・旧STOPは変更していない（B2の `5b7c07679` は無変更）。
- 実行時間：受入の実行は約20秒（Python 3.12.3）。

**検査の性質**：作成者自身による技術検査で、独立レビューではない。

**mandatory STOP。**
