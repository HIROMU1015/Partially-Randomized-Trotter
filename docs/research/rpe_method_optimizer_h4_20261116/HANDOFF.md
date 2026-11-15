# 統合担当・A/B/C/D/D2へ：方式別RPEコスト最適化（PR構成の生成まで）

正本は[REPORT](REPORT.md)の§E〜H（第3段階）と§A〜D（第2段階）。branch `codex/b2-rpe-method-optimizer-20261116`、実装SOURCE：PR生成は `2fa2dbfac3646b801a3d5b8ebb02676b989bcb78`、D2価格適用判定の接続は `47f86eb20`（報告出力 `d993d5e5b`、公開 `3014ae142`）。新規の実信号・価格・compile・Q3は0件。

## 追加：D2の価格適用判定の接続（policy=MODEL）

- PR候補ごとにD2の `PriceRules.quote` を呼ぶ（SOURCE `c1f95e08`、HANDOFF `7cd8ea139`）。渡すのは、configuration本体・登録された候補固有の実装契約・`policy: "MODEL"`（明示）。問題入力の `price_rules` で固定し、MODEL以外は拒否する。
- **保存構成**：単価・総額をexactに再現した。L_D=9は従来どおり価格待ち。
- **生成した構成**
  - 26軸が揃えば、その単価で費用を計算する。例：Pauli L_D=2・3・5のκ×2、PR-DF L_D=2のκ×2。
  - 欠ける場合は、不足軸・理由・必要な較正を `price_needs`／`PRICE_AXES:n_of_26` に出す。例：Δ_PF=h/2は「実stepが変わった」、PR-DF L_D=1のκ×2は「E_I[S]」。
  - 価格の状態とQ3の状態（信号・経路）は別に保持する。価格が揃っても信号が不足なら探索に入らない。
- 出力：`run04/PRICE_RULES.json`、候補ごとの `price_quote`・`price_needs`。

## 第3段階で追加したこと

- **PR構成の生成**
  - PR-Pauli・PR-DFについて、L_D・公式・Δ_PF・κを変えた構成を、各方式の固定mapperで生成する。Δ_PFはhの何倍か、κは共通κの何倍かで指定する。
  - roundごとの数値を手入力する必要はない。q_m・R_m・piece配分はmapperが決める。
  - 保存PR構成7件は、現行のS2・Δ_PF=h・共通κで完全に再現した。
- **対応する公式**：`2nd` と `4th(m5_best)`。高次公式では、signed piece・piece別event配分・event数を照合する。
- **Δ_PFの制約**：PR-DFは全roundでT/Δが整数になること（Δ_PF=2hは理由付きで拒否される）。PR-Pauliは任意のΔを扱える。
- **重複の除去**：物理schedule（κはRを通してだけ効く）が同じ変種は保存構成の別名とし、保存評価を再利用する（二重評価しない）。新scheduleの構成は候補になり、不足を機械可読で返す。
  - 構成の公開
  - 不足する信号round
  - packetの組立て
  - 不足する価格軸の数
  - Q3経路の雛形
- **κの判断材料**：`parameter_assessment` が、(L_D, 公式, Δ_PF) ごとに各κの物理同一性・不足・最小適合費用を並べる。PR-DFのL_D=7・9（Δ_PF=h）は「全κで同じschedule」。他は入力不足で判定保留。
- **停止**
  - `"UNLIMITED"`：件数上限なし。`science_approved: true` が必須。
  - `NEW_SCIENCE_NOT_APPROVED`：未承認。
  - `NO_IMPROVING_PROPOSAL`：範囲を尽くした。
  - `NO_IMPROVING_PROPOSAL_INPUTS_MISSING`：範囲内に入力不足の候補が残る。
- **受入**：生成8/8、実行器10/10、Aの固定受入20 PASS（Skillは PENDING 1）。価格接続後（`47f86eb20`）は、生成11/11・実行器10/10・A 20 PASS。

## 生成範囲の指定（round file）

```json
"method_overrides": {
  "PR_DF":    {"generation": {"formulas": ["2nd", "4th(m5_best)"], "delta_over_h_values": ["1", "1/2"], "kappa_multipliers": ["1", "2", "4"]}},
  "PR_PAULI": {"generation": {"formulas": ["2nd"], "delta_over_h_values": ["1", "1/2", "2"], "kappa_multipliers": ["1", "2"]}}
},
"science_approved": false,
"max_new_q3_per_method": {"PR_DF": 0, "PR_PAULI": 0}
```

雛形は [configs/rpe_method_optimizer/rounds/example_h4_pr_generation_round_unapproved.json](../../../configs/rpe_method_optimizer/rounds/example_h4_pr_generation_round_unapproved.json)。未承認のまま実行して、PR-DF 48変種（同一8、新40）、PR-Pauli 18変種（同一3、新15）を確認した。起動は0件。件数上限なしで範囲を尽くす場合は、`"science_approved": true` とし、予算を `"UNLIMITED"` にする。

## 実行可能になった範囲

- **実行できる**
  - 保存構成（保存と同一scheduleのκ・Δ_PF変種を含む）での、構成とshot配分の探索。
  - 採択roundでの、既存Q3経路の実行・取込・再開。
- **生成・不足判定まで**：新scheduleの構成。次の取得が揃うまで探索には入らない（停止理由 `…INPUTS_MISSING`）。
  - 信号（A／D2のrunner）
  - 価格（A／D2）
  - 構成の公開
  - packetの組立て
  - Q3経路の雛形
- **再開コマンド**：第2段階と同じ。`--round <file> --run-dir <dir>` を再実行する。

---

# （第2段階）統合担当・A/B/C/D/D2へ：方式別RPEコスト最適化（Q3実行器の接続まで）

正本は[REPORT](REPORT.md)の§A〜D。branch `codex/b2-rpe-method-optimizer-20261116`、SOURCE `a2e2ef50a108d6f34089a99bd00f11a7085dd9c9`。新規の実Q3・科学計算・compileは0件。

## 更新後の方式別最安点と次候補（Cの `417a73e73`、新規Q3予算0）

| 方式 | 最安適合構成・X/Y各軸・費用 RZ | 次候補（探索規則が選択）・費用 RZ | 経路 |
|---|---|---|---|
| Full PF | S2・16h/7、[59,55,51,47,43,39,35,31,27,23,19,13,2]、559,457,386 | M5・16h：[30,28,26,24,22,20,17,15,13,11,9,7,3]、541,075,202 | `pf_b_pf6` |
| B0I p7 | Yoshida4・32h/5（A_SHAPE）、[37,34,32,29,27,24,22,19,17,14,11,9,6]、**372,035,410** | 同じ構成でm12を6→4：[37,…,11,9,4]、329,694,342 | `pf_b_pf6` |
| PR-Pauli | L_D=5、[48,45,42,39,36,33,29,26,23,19,16,8,2]、541,743,905.57 | L_D=3でm12を2→1：[48,…,16,8,1]、483,460,003.67 | `pauli_b_ld3` |
| PR-DF | L_D=7、[48,45,42,39,35,32,29,25,22,19,16,12,9]、1,235,340,731.91 | 同じ構成でm12を9→7：[48,…,16,12,7]、1,134,606,918.91 | `prdf_d2_split` |

- 4件の次候補と次点のconfiguration（計6件）は、実経路でdry-checkした。case組立て・metadata・count範囲・runtimeの確認で、科学callは0件。すべてPASS。
- p7の372,035,410は、保存Q3（`8855f420a`）とDの価格（`c7c8c1edd`）から集計した。新規Q3は0件。

## 次段階で研究者が探索1回分の対象・予算を指定する方法

round file（例：[configs/rpe_method_optimizer/rounds/example_h4_round_budget0.json](../../../configs/rpe_method_optimizer/rounds/example_h4_round_budget0.json)）をcopyし、次の欄を書く。

| 欄 | 意味 |
|---|---|
| `round_id` | この探索1回分の名前。run directoryと結びつく |
| `methods` | 対象方式（`FULL_PF`・`TRUNCATED_PF`・`PR_PAULI`・`PR_DF` から選ぶ） |
| `max_new_q3_per_method` | 方式ごとの新規実Q3の上限。0なら何も起動しない |
| `method_overrides` | 探索範囲（任意）。例：`{"TRUNCATED_PF": {"formulas": ["4th"], "delta_over_h_range": ["32/5","32/5"]}}`、`{"PR_DF": {"L_D_values": [7]}}`、`configuration_ids` の許可リスト |
| `initial` | 開始時に既知とする保存結果。`"all"`（既定）・`"none"`・`{方式: [保存ラベル]}` |
| `resume_from` | 前回の `ROUND_RESULT.json` / `RESULT.json`（`resume_records` をそのまま再利用） |
| `approved_by` | 採択の記録（任意の文字列） |

- 範囲の中では、プログラムが次を決める：候補、配分、起動、取込、次候補、停止。
- 予算を広げるときは、同じ `round_id` のまま予算だけを書き換えて、同じコマンドを再実行する。完了済みのjobは再利用され、起動済みの件数は予算に数えられる。

## 実行・再開コマンド（Bのskillからもこのまま呼べる）

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/b2-rpe-method-optimizer-20261116
PY=/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python
# 探索roundの実行。中断後や予算追加後も、同じコマンドで再開する
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $PY -B scripts/run_rpe_method_optimizer.py \
  --problem configs/rpe_method_optimizer/h4_df10_problem.json \
  --round <round.json> --run-dir <round directory>
# 報告モード（新規Q3 0件）。前回の出力から続ける場合は --resume-from、実経路の事前検査は --dry-check-routes
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $PY -B scripts/run_rpe_method_optimizer.py \
  --problem configs/rpe_method_optimizer/h4_df10_problem.json --output <出力dir> \
  [--resume-from <前回のROUND_RESULT.json>] [--dry-check-routes]
```

- round directoryの中身
  - `ROUND.json`：採択内容
  - `ROUND_RESULT.json`：方式別最安・trace・次の要求・resume_records
  - `PER_METHOD.csv`
  - `NEEDS.json`
  - `INVOCATIONS.jsonl`：起動ごとの予算・起動数・停止理由
  - `jobs/<方式>/<job_id>/`：REQUEST・LAUNCH・PROCESS_RETURN・OUTCOMEと、workerの原API・partials・資源
- 停止理由
  - `NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET`：予算0
  - `Q3_BUDGET_EXHAUSTED`：予算切れ
  - `NO_IMPROVING_PROPOSAL`：改善候補なし
  - `JOB_IN_PROGRESS`：実行中jobあり
  - `STALE_JOB_NEEDS_REVIEW`：死んだjob。自動retryしない
  - `NO_ROUTE_ADAPTER`：経路なし
  - `UNKNOWN_RESULT`：gate失敗。モデル不適合とは別扱い

## 受入結果

- Aの固定受入runner（`3115712244`）をこのcheckoutで実行した。結果は PASS 20、Skillは PENDING 1（旧実装はFAIL 4）。出力：`a_acceptance_a2e2ef50a/ACCEPTANCE_RESULT.json`。
- 実行器の受入（実際の入口、人工worker・保存結果）は10/10 PASS（`executor_acceptance_a2e2ef50a/`）。
  - 要求→実行→取込→次候補
  - 予算切れ→再開（再起動なし）
  - 保存結果の再利用
  - 予算0では起動しない
  - 実行中jobの二重起動なし
  - 死んだjobの報告とcollect
  - UNKNOWNとモデル不適合の区別

## 人手が残る箇所

1. 探索roundの採択（round fileの作成）。
2. 経路の無いconfiguration（例：PR-PauliのL_D=2）の手動実行。
3. 新しい幅・分割の信号・価格・configurationの取得と、Cへの登録。
4. route view（読取専用worktree）の用意。pathは問題入力の `execution_routes` にある。
5. Bのskill化。
6. 死んだjobの判断。

## ロードマップへの差分（本体は統合担当が更新）

- 方式別最適化の実行器がつながった。採択された探索roundの範囲では、要求から次候補まで自動で進み、再開もできる。
- p7の保存集合内最安は372,035,410 RZ（粗幅Y4 A_SHAPE）。実Q3はまだ0件。

Cの集計・各原結果・旧STOPは変更していない。公開・remote照合後に **mandatory STOP**。
