# RPE方式別最適化プログラムの受入ケース（利用者の立場から）

想定読者：B2（最適化プログラムの実装）、B（Skillからの起動）、統合担当。

**目的**：人が途中で指示を足さなくても、探索が進むことを確かめる。具体的には、ケースごとに次の3つを定める。
- 利用者が与える入力
- 実行の途中で返ってくる評価結果
- 観測すべき動作（合格条件）

機械可読版は [CASES.json](CASES.json)、実行器は [`scripts/acceptance_rpe_method_optimizer.py`](../../../scripts/acceptance_rpe_method_optimizer.py)。

**範囲**
- 保存結果と人工fixtureだけを使う。新しい信号・価格・compile・Q3は0件である。
- 人工の答えには `ARTIFICIAL_FIXTURE` の印を付け、科学結果として扱わない。

## 実行方法

```
python scripts/acceptance_rpe_method_optimizer.py --optimizer-root <最適化プログラムのcheckout> --output <出力dir>
```

- 対象の入口は、`--optimizer-root` にある `src/trottertracks/rpe_method_optimizer.py` の `prepare` と `optimize` である。
- 入力の正本は、同じtreeの `configs/rpe_method_optimizer/h4_df10_problem.json` である。ケースごとに変える部分（探索範囲、開始時に既知の点、予算、catalogのcommit）は、実行器が明示的に上書きする。
- 結果は `ACCEPTANCE_RESULT.json` に、checkごとのPASS／FAILとその根拠として書き出す。
- 終了codeは、すべてPASS（Skillは未公開の扱い）なら0、それ以外は1。
- 判定の関数が、想定した失敗を本当に検出することは、`tests/check_acceptance_rpe_method_optimizer.py` で確かめている（合成した結果を使う）。

## ケース1：p7 S2のshot探索

**入力**
- 探索範囲：TRUNCATED_PFを、公式2nd・幅16/7（p7 S2）に限る。
- 開始時に既知の点
  - 1a：保存されたm12各4 shotの適合点 `B0I_P7_S2_16h_over_7__B_PRICE` だけ。
  - 1b：それに加えて、m12各3 shotの適合点 `P7_S2_LOCAL_R12_D1`（3 shotで改善した後の状態）。
- Q3の答え：保存catalogの結果を使う。ただし、プログラムがその配分を選んだ後にだけ返す（記録付きの応答器）。
- 新しいQ3の予算：0。1bでは1も試す。そのときは、保存されていない配分に人工の答えを返す。

**途中で返るもの**
- stepごとに：選んだ配分、費用、Q3の判定、改善の前後の最安点。
- 最後に：最安の適合点、停止理由、次に必要な具体的な要求（configuration、26軸のshot数、費用、Q3の実行経路）。

**観測すべき動作**

| check | 合格条件 |
|---|---|
| `search_space_follows_requested_range` | 探索した候補が、指定した範囲（p7 S2）だけである |
| `starts_from_given_saved_point_only` | 最初の最安点が、与えた保存点（1aは4 shot、1bは3 shot）である |
| `saved_Q3_revealed_only_after_selection` | 応答器に問われた配分が、各stepで選ばれた配分（と最後の要求）だけである。選ぶ前に保存結果を覗かない |
| `next_action_decided_without_asking` | 規則による停止理由で終わる。続きが必要なら、具体的な要求（configuration・26軸・費用・経路）を返す。利用者にroundやshot数を尋ねない |
| `continues_after_improvement`（1a） | 改善の後も、次のstepを取るか、更新後の最安点より安い要求を返す |
| `next_action_from_m12_3_state_without_budget`（1b、予算0） | 3 shotの状態から、予算が無いことを理由に止まり、具体的な要求を返す |
| `remaining_budget_used_then_rule_stop`（1b、予算1） | 残り1回を尋ねずに使い、その後は予算の規則で止まって要求を返す |

**合格条件にしないこと**：保存済みの4→3→2→1という順序。

## ケース2：p7 Yoshida4・粗幅32h/5・A_SHAPEの価格待ち

**入力**
- 価格の前：B2の入力と同じC catalog INPUTS（`08c6a694b`）。Q3は適合、26軸の価格は無い。
- 価格を与えた後：C catalog INPUTS `417a73e73`。そのsourcesに、Dの固定価格が含まれる：`c7c8c1edd0996ad9ef97629961f67ca508a182f0` の `docs/research/d_p7_y4_coarse_price_calibration_20261116/PRICE_AND_COST.json`。
- 新しいQ3の予算：0。

**観測すべき動作**

| check | 合格条件 |
|---|---|
| `price_missing_identified` | その候補（configuration `dd8bfc24…`）がnot readyで、不足が `PRICE_26AXIS` だけである。保存Q3（適合）は保持されている |
| `price_wait_reported_in_output` | 出力の未評価候補に、価格待ちとして載る |
| `price_input_is_D_saved_file` | 与えた入力に、Dの固定価格fileが含まれる（入力の確認） |
| `saved_Q3_reused_without_new_call` | 価格を与えた後、既存のQ3を再利用し、新しいcallは0 |
| `D_price_total_in_comparison_table` | 比較の出力（evaluatedの行）に **372,035,410 RZ** が入り、方式の最安値は372,035,410以下になる |

## ケース3：予算の終了と再開（人工fixture）

**入力**
- 人工の方式 `ART` に2つのconfigurationを置く。
  - A：価格あり。保存の適合点を1つ持つ。
  - B：価格待ち。
- 人工のQ3：A以外のすべての配分に答える。
- 予算：2回。再開時は、完了したcallの結果を渡し、新しい予算2回で続ける。

**観測すべき動作**

| check | 合格条件 |
|---|---|
| `budget_end_returns_best_unevaluated_and_reason_separately` | 予算の停止理由と、確認済みの最安点（適合、出所付き）と、未評価のもの（次の要求、価格待ちの候補とその不足）を、別々の欄で返す |
| `no_science_call_beyond_budget` | 応答器に実際に答えさせた回数が予算以下。予算を超える問い合わせをして、その答えを捨てることがない |
| `resume_input_recoverable_from_output` | 1回目の出力（JSON）だけから、再開に必要な結果（configuration、shot数、判定、MSEの上下限）を復元できる |
| `resume_does_not_repeat_completed_calls` | 再開時に、完了済みの配分へ科学callをしない |
| `resume_starts_from_previous_state` | 再開した最初の最安点が前回の最安点で、最初の新しいcallは前回の要求である |

## Skill（B）からの起動

- Skillは、上のケースの入力を最適化プログラムへ渡し、同じ欄を持つ出力を返すこと。
- 途中で利用者に質問しないこと。
- Skillが公開されたら、同じ判定をSkillの出力にも当てる。現状は `PENDING_NOT_PUBLISHED`。
