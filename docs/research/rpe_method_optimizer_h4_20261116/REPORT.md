# H4：方式別RPEコスト最適化（第2段階：Q3実行器の接続とAの受入テストの反映）

2026-11-16 JST、B2（Claude Code、主agentのみ。作成者自身の確認で、独立レビューではない）。SOURCEは `a2e2ef50a108d6f34089a99bd00f11a7085dd9c9`。**新規の実Q3・科学計算・compileは0件。** 確認は人工workerと保存結果だけで行った。第1段階の記録は後半に残した。

## A. 結論

- 次の流れを同じ実行入口につないだ。
  - 探索器が要求を出す。
  - その方式の既存Q3経路を起動する。
  - 結果を照合して取り込む。
  - 次候補を選ぶ。
- 開始時に採択された方式・予算（探索round）の範囲では、候補ごとの人手による起動・取込は要らない。
- 同じコマンドを再実行すると再開になる。保存結果・完了jobを先に照合して再利用し、実行中のjobは二重起動しない。
- 固定入力をCの `417a73e73` に更新した（Cのreader `be5326bdc` を取り込み、価格欄 `unit_price_RZ` に対応）。
  - p7の最安適合点372,035,410 RZ（粗幅Yoshida4・32h/5・A_SHAPE）を、保存Q3だけで再現した（新規Q3は0件）。
  - 4方式の次候補は、旧要求を使い回さず探索規則から作り直した。
- Aの固定受入テスト（`3115712244`）を新しいcheckoutで再実行した。
  - 結果は **PASS 20、Skillは未公開のためPENDING 1**。旧実装 `5b7c07679` では PASS 15、FAIL 4 だった。
  - 出力は別の場所 [a_acceptance_a2e2ef50a/](a_acceptance_a2e2ef50a/ACCEPTANCE_RESULT.json) に保存した。
- 実行器の受入（実際の入口、人工worker・保存結果）は10件すべてPASS（[executor_acceptance_a2e2ef50a/](executor_acceptance_a2e2ef50a/EXECUTOR_ACCEPTANCE.json)）。
- 予算0の探索roundでは、実経路で科学callが1件も起動しないことも確かめた。

### 更新後の方式別最安点と次候補（[run02](run02/RESULT.json)、新規Q3予算0）

| 方式 | 最安適合構成 | X/Y各軸（m0〜m12） | MSE上界 Ha² | 費用 RZ | 次候補（探索規則が選択） | 経路 |
|---|---|---|---:|---:|---|---|
| Full PF | S2・16h/7 | [59,55,51,47,43,39,35,31,27,23,19,13,2] | 2.40247×10⁻⁶ | 559,457,386 | M5・16hの適合点からm12を−2：[30,28,26,24,22,20,17,15,13,11,9,7,3]、541,075,202 | `pf_b_pf6` |
| B0I p7 | **Yoshida4・32h/5**（A_SHAPE） | [37,34,32,29,27,24,22,19,17,14,11,9,6] | 2.07438×10⁻⁶ | **372,035,410** | 同じ構成でm12を6→4：[37,34,32,29,27,24,22,19,17,14,11,9,4]、329,694,342 | `pf_b_pf6` |
| PR-Pauli | L_D=5 | [48,45,42,39,36,33,29,26,23,19,16,8,2] | 2.50026×10⁻⁶ | 541,743,905.57 | L_D=3の適合点からm12を2→1：[48,…,16,8,1]、483,460,003.67 | `pauli_b_ld3` |
| PR-DF | L_D=7 | [48,45,42,39,35,32,29,25,22,19,16,12,9] | 5.56518×10⁻⁷ | 1,235,340,731.91 | 同じ構成でm12を9→7：[48,…,16,12,7]、1,134,606,918.91 | `prdf_d2_split` |

- 4方式とも停止理由は `NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET`（予算0）。
- 費用は各候補自身の保存26軸単価による条件付き総compiled RZ。MSEは保存Q3のモデル判定。

## B. 指示ごとの対応

### B1. 固定入力の更新（Cの価格反映）

- Cの集計branchを取り込み、問題入力の集計INPUTSを `417a73e7329fd520381dc88cc0de00a11dd0b641` にした。
- 粗幅Y4（`dd8bfc24`）の26軸単価は、Dの価格原本 `c7c8c1edd` の `unit_price_RZ` から取った。保存Q3（`8855f420a`）の配分の総額372,035,410 RZとexactに一致する。
- 旧入力（`08c6a694b`）も引き続き読める。Aのケース2「価格前」がPASSしている。

### B2.「既知の不適合に支配される」の枝刈りを外した

- 旧実装の条件は次のとおりだった。同じconfigurationの既知のモデル不適合配分fが、候補vに対して26軸すべてで f≥v なら、vを飛ばす。これは、各軸のshot数についてMSEが単調だと仮定していることになる。
- この枝刈りを削除した。除外するのは、同じconfiguration・同じ配分の保存結果・完了結果（再利用）だけである。
- テスト `test_no_monotonicity_pruning…` で、不適合配分の下にある候補が残ることを確認した。

### B3. Q3の自動実行と再開

- **実行器**（`rpe_method_optimizer.Executor`）
  - 要求ごとにjob directory（`<run-dir>/jobs/<method>/<job_id>/`）を作る。
  - その方式の経路で、adapter `scripts/run_rpe_optimizer_q3_job.py` を経路自身のinterpreterで1回だけ起動する。
  - adapterは、既存callerのworker・supervisor・結果検査を、各callerが実際に動いたcommitの読取専用view（detached worktree）から無変更で読み込む。新しく作るのは1件のcaseだけで、同じconfigurationの保存Q3 caseに要求の配分を入れる。
  - 経路とview：
    - PF：B `cbc471c91`（PF6 worker）
    - Pauli LD3：B `a844ac25f`
    - Pauli LD5：B `b6d4dea49`
    - PR-DF：D2 `23160cc87`（`run_pr_df_ld_split_q3_resume`、固定view `27e70efe5`）
- **取込前の照合**：次のいずれかが一致しなければ `UNKNOWN`（`IDENTITY_MISMATCH`）として止める。
  - configuration、配分（26軸）
  - total configuration・binding（Cの保存点と照合）
  - 評価契約（statistical contract・options）
- **予算**：新規Q3を起動し得る応答器を呼ぶ前に予算を確かめる。保存結果・完了jobは予算を使わない。
- **再開**
  - OUTCOMEのあるjobは再利用する。
  - 実行中（pidが生きていて、cmdlineにjob pathを含む）のjobは `JOB_IN_PROGRESS` で止め、起動しない。
  - 死んだjobでOUTCOMEが無いものは、collect modeで保存artifactだけを検査する（科学callなし）。artifactも無ければ `STALE_JOB_NEEDS_REVIEW` とし、自動retryしない。
- **UNKNOWNとモデル不適合の区別**
  - 数値gate・保存・process・照合の失敗は `UNKNOWN` で、その方式を止める。
  - 下界がτを超える場合は `MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD` で、探索を続ける。
- **実経路の技術検査**：4方式の次候補、および次点に出たconfiguration（M5、S2、LD5）で、計6件のdry-checkを行った。内容は、実経路でのcase組立て・metadata・count範囲・runtime確認で、科学callは0件。すべて `DRY_CHECK_PASS`（[run02/ROUTE_DRY_CHECK.json](run02/ROUTE_DRY_CHECK.json)）。

### B4. Aの受入テストの反映（旧FAIL 4件）

| 旧FAIL | 対応 | 今回 |
|---|---|---|
| 探索範囲の遵守 | 公式・幅の範囲を、索引に無いcatalogのconfigurationにも適用した（幅は `delta_pf/h`）。任意の `configuration_ids`・`L_D_values` の許可リストも、全ての候補源に適用する | PASS（p7 S2限定で最終要求まで範囲内） |
| Dの価格を与えた後の比較 | Cのreaderを取り込み、`unit_price_RZ` に対応した | PASS（保存Q3を再利用し、372,035,410、新規0件） |
| 予算を超える呼出し | 予算確認を応答器の呼出しより前に移した | PASS（予算2で応答器の新規回答は2件だけ） |
| 出力からの再開 | traceとevaluatedに次を出すようにした：configuration_id、N_X/N_Y、MSE_lower/upper、判定、出所（point ID・job・result sources）。`resume_records` をそのまま再開入力（`--resume-from`）として使える | PASS |

### 実行器の受入（実際の入口 `run_rpe_method_optimizer.py --round … --run-dir …`）

| check | 内容 |
|---|---|
| budget0_real_routes_launch_nothing | 実経路・4方式・予算0：jobは0件、全方式が予算外の要求を返す |
| saved_results_reused_without_budget | p7 S2の4 shot seedから、保存Q3（m12=2適合、m12=1不適合）をcall 0件で使う |
| request_execute_ingest_next | 予算2：人工workerのjobを2件起動し、結果を照合・取込して、次の候補を更新後の状態から選ぶ |
| budget_exhausted_stop | `Q3_BUDGET_EXHAUSTED` と次の要求を返す |
| scope_kept_to_final_request | p7 S2限定の範囲を、最終要求まで守る |
| resume_after_budget_end_without_relaunch | 予算を4に上げて同じコマンドを再実行：完了2件は再利用（再起動なし、LAUNCHの時刻不変）、新規2件 |
| idempotent_rerun_launches_nothing | もう一度実行しても新規起動0件 |
| running_job_not_launched_twice | 実行中に同じroundを起動すると `JOB_IN_PROGRESS`。終了後の再実行で結果を取り込む |
| dead_job_reported_then_collected | 死んだjob（artifactなし）は `STALE_JOB_NEEDS_REVIEW`。artifactが現れた後はcollectで取り込む（再計算なし） |
| unknown_distinct_from_model_failure | 人工のgate失敗は `UNKNOWN_RESULT` で止まる。不適合の結果では探索が続く |

テスト：新16件と、再利用したCの集計テスト12件がPASS（[TESTS_EXECUTOR.txt](TESTS_EXECUTOR.txt)）。

## C. 実行・再開のコマンド

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/b2-rpe-method-optimizer-20261116
PY=/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python
# 報告モード（新規Q3 0件）：方式別最安点・次候補・不足、必要なら実経路のdry-check
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $PY -B scripts/run_rpe_method_optimizer.py \
  --problem configs/rpe_method_optimizer/h4_df10_problem.json --output <出力dir> [--dry-check-routes] [--resume-from <前回のROUND_RESULT.json>]
# 採択された探索round：実行と再開（同じコマンドを再実行すると再開）
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $PY -B scripts/run_rpe_method_optimizer.py \
  --problem configs/rpe_method_optimizer/h4_df10_problem.json --round <round.json> --run-dir <round dir>
```

round fileの書き方は [configs/rpe_method_optimizer/rounds/example_h4_round_budget0.json](../../../configs/rpe_method_optimizer/rounds/example_h4_round_budget0.json) と HANDOFFにある。

## D. 実行後もなお人手が必要な箇所

1. **探索1回分の採択**：対象方式・新規Q3予算・範囲をround fileに書く。予算を広げるときは、同じround_idで予算を書き換えて再実行する。
2. **経路が無いconfiguration**：例えばPR-PauliのL_D=2には経路の規則が無い。要求は `NO_ROUTE_ADAPTER` として出るので、手動のcallerが必要。PR-DFのL_D=1・2は経路の規則に含まれるが、dry-checkはしていない。
3. **信号・価格・configurationの取得**：新しい幅・分割の取得と、Cへの登録は自動化していない。
4. **route viewの用意**：route viewは問題入力のpathに置いた読取専用worktreeで、別の環境では作り直す必要がある。
5. **B Skill**：Bの手順書としては未公開。実行するコマンドはHANDOFFで渡した。
6. **死んだjobの扱い**：artifactの無い死んだjobは自動retryせず、研究者の判断を待つ。

---

# 第1段階：H4 方式別RPEコスト最適化の最初の実装（保存結果と人工fixtureで確認）

2026-11-16 JST、B2（Claude Code、主agentのみ。作成者自身の確認で、独立レビューではない）。棚卸し `c5e03cd6c` を受けた最初の実装。**新規の科学計算・compile・Q3は0件。**

## 0. 結論

- 一つのコマンドで、H4の問題入力から方式別の最安適合構成と、次に必要なQ3・信号・価格までを出せるようにした。
  - 方式別の最安適合構成：構成・round別X/Y配分・MSE上下界・費用・停止理由・探索範囲。
  - 次に必要なもの：機械可読で出力する。
- 保存結果だけで次を確認した。
  - Cの保存集合内の方式別最安点を4方式とも再現した。
  - p7 S2・Full PF S2は、m12各4 shotの適合点だけを初期知識にして、より安い適合点（m12=2）へ自分で進んだ。そのうえで、次に評価すべき配分を決めた。
  - PRでは、L_Dの異なる構成をそれぞれ自分の価格で比べた。
- **現状の4方式は、どれも「承認済み予算の外で新規Q3が必要」で止まる。** 次のQ3要求（構成・配分・費用・根拠）を出す。

| 方式 | 評価済み集合内の最安適合構成 | X/Y各軸（m0〜m12） | MSE上界 Ha² | 費用 RZ | 次に必要なQ3（探索規則が選んだもの） |
|---|---|---|---:|---:|---|
| Full PF | S2・16h/7 | [59,55,51,47,43,39,35,31,27,23,19,13,2] | 2.40247×10⁻⁶ | 559,457,386 | M5・16hの適合点（A_SHAPE）からm12を−2：[30,28,26,24,22,20,17,15,13,11,9,7,3]、541,075,202 |
| Truncated PF（B0I p7） | S2・16h/7 | [59,55,51,47,43,39,35,31,27,23,19,13,2] | 2.41568×10⁻⁶ | 395,962,678 | M5・16hの適合点（A_SHAPE）からm12を−2：[30,28,26,24,22,19,17,15,13,11,9,7,3]、382,916,152 |
| PR-Pauli | L_D=5 | [48,45,42,39,36,33,29,26,23,19,16,8,2] | 2.50026×10⁻⁶ | 541,743,905.57 | L_D=3の適合点からm12を−1：[48,45,42,39,36,33,29,26,23,19,16,8,1]、483,460,003.67 |
| PR-DF | L_D=7 | [48,45,42,39,35,32,29,25,22,19,16,12,9] | 5.56518×10⁻⁷ | 1,235,340,731.91 | 同じ構成からm12を−2：[48,45,42,39,35,32,29,25,22,19,16,12,7]、1,134,606,918.91 |

- 費用は各候補自身の保存26軸単価による条件付き総compiled RZ数である。PFは直接価格と閉包価格、PRはモデル参照価格で、E[C]・CIではない。MSEは保存Q3（モデル判定）の値。
- 停止理由は4方式とも `NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET`。今回の予算は「新規Q3 0件」である。
- 詳細の正本は [run01/RESULT.json](run01/RESULT.json)、[run01/PER_METHOD.csv](run01/PER_METHOD.csv)、[run01/NEEDS.json](run01/NEEDS.json)。

## 1. 再実行コマンド

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/b2-rpe-method-optimizer-20261116
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -B \
  scripts/run_rpe_method_optimizer.py \
  --problem configs/rpe_method_optimizer/h4_df10_problem.json \
  --output docs/research/rpe_method_optimizer_h4_20261116/run01
```

- 新しく得たQ3結果は `--ingest results.json` で取り込める。形式は `{configuration_id, N_X, N_Y, status, MSE_lower, MSE_upper, source}` の配列で、取り込み後の知識から次の要求を計算し直す。
- 全体のwallは約6.6秒で、大半はCの集計器の再構築（固定Git objectの読み取り）である。
- 実行SOURCEは `5b7c07679f16b1a3cf14c5236e0d75611a53485a`（`run01/CHECKS.json` の `execution_source`）。

## 2. 実装

| 部分 | 場所 | 内容 |
|---|---|---|
| 問題入力 | [configs/rpe_method_optimizer/h4_df10_problem.json](../../../configs/rpe_method_optimizer/h4_df10_problem.json) | H4固有の項目はすべてここにまとめ、codeには持たせない（下の表） |
| 最適化器 | [src/trottertracks/rpe_method_optimizer.py](../../../src/trottertracks/rpe_method_optimizer.py) | 入力確認、Cの集計の再構築、候補の列挙、価格の取付け、探索、要約 |
| 実行入口 | [scripts/run_rpe_method_optimizer.py](../../../scripts/run_rpe_method_optimizer.py) | FULL・INGEST・REPLAY・FIXTURE・OWN_PRICEと検査を一度に実行 |
| テスト | [tests/check_rpe_method_optimizer.py](../../../tests/check_rpe_method_optimizer.py) | 人工fixture 10件＋保存結果の統合1件 |

**問題入力に置いたH4固有の項目**

- Hデータ：snapshotとreceiptのpath・sha256、期待する次元（fragment 10、8 qubit、sector 36）。
- target、精度条件、RPEのM。
- 証拠の場所：Cの集計INPUTS（`08c6a694b`）、PF候補索引（`7f27e18f0` の `WIDTH_SIGNAL_COVERAGE.json`）。
- 方式ごとの定義と候補範囲：PFは公式と幅の範囲（h/2〜16h）、PRはL_Dの範囲の規則。
- 各軸のshot上限（N0・VP・V0）。
- 探索パラメータ、既存Q3経路の案内、確認用のseedラベル。

**処理の流れ**

1. **Hデータの確認**
   - receiptとsnapshotのsha256を照合する。
   - receiptの配列記録から次の値を求め、入力の期待値と照合する。
     - fragment数（`g_matrices` の先頭次元＝10）
     - qubit数（状態ベクトル長256から8）
     - sector次元（36）
2. **保存結果**
   - Cの `saved_rpe_cost_catalog.Catalog` を、Cの固定INPUTSからその場で再構築する。結果は49点。
   - Cの比較条件（system・snapshot/H/state hash・εE・decoder・loss・target・M）が問題入力と一致することを確かめてから、保存Q3を再利用する。
3. **候補の列挙**
   - PF：索引から、方式・公式・幅が入力範囲に入るconfigurationを列挙する。Full PF 20件、B0I p7 9件。
   - PR：L_Dを「1〜fragment数−1」としてcodeで列挙する（両方式とも1〜9）。各L_Dにconfigurationがあれば、それぞれ1候補とする。
   - 各候補に、configuration・信号・26軸価格を取り付ける。
4. **価格**
   - 各configurationの保存26軸単価を、Cの価格locatorから取り出す。
   - **そのconfigurationの全保存点で、Σ N×単価が保存総額とexactに一致する**ことを確認する。12 configurationとも一致した。
   - 一致しない、または価格が無い場合は、その候補を「価格が必要」とする。
5. **探索（`FINITE_LOCAL_DESCENT_V1`）**
   - 変数はround別のX/Y shot数（26整数）。
   - 各configurationで最も安い適合点から、1 roundの減算を候補にする。X/Yの単価が違うroundでは、軸ごとの減算も加える。
     - 減算幅は1または2。
     - その適合点のQ3上界がτの9/10以上なら、減算幅は1に限る。
   - 適合点の無いconfigurationには、方式の最安費用を予算とする既存の配分生成器（`e094b2d60` の `generate`）で候補を作る。どこにも適合点が無ければ、各軸上限の列も候補にする。
   - 方式の最安費用より厳密に安い候補だけを残し、費用の安い順に並べる。
   - 既知の不適合列が全成分で上回る候補は「既知不適合に支配される」として飛ばし、その旨を記録する。これは探索順の発見的規則で、不適合とはみなさない。
   - 選んだ候補をQ3の取得元に問い合わせる。
     - 取得元は、保存結果（Cの集計）か、テスト用の人工fixtureである。
     - 答えが無く、承認された新規Q3予算も残っていなければ、その候補を要求として出して止まる。
   - 答えを受け取るたびに、最安点・候補・順位を作り直す。最初の改善の後も、停止規則が働くまで続ける。
   - 停止規則は5種類。
     - 改善候補が無い
     - 予算外の新規Q3が必要
     - 新規Q3の予算を使い切った
     - UNKNOWN
     - 手数の上限
6. **不足の出力**
   - 信号・価格・configurationが無い候補は、不適合とは扱わない。`NEEDS.json` に、何が揃えば探索に入れるかを出す。
   - 探索可能な候補には、次のQ3要求を出す。要求には、配分、費用、どの適合点からどう変えたか、既存Q3経路の案内、次点の4候補を含める。

## 3. 動作確認（`run01/CHECKS.json`）

| 確認 | 結果 |
|---|---|
| Cの方式別最安点の再現（Full PF・p7・PR-Pauli・PR-DF） | 4方式とも一致 |
| 自分の単価が保存総額を再現 | 12 configurationとも一致 |
| L_Dの列挙がfragment数から決まる | PR-Pauli・PR-DFとも1〜9（fragment数10） |
| **REPLAY：p7 S2** | 初期知識はm12各4 shotの1点（435,481,090）だけ。①m12を−2（m12=2）→保存結果が適合で、最安を395,962,678へ更新。②m12を−1（m12=1）→保存結果が不適合。③次は別configuration（M5）の生成器候補380,353,796で、保存結果が無いため要求として停止。保存結果を使ったので新規callは0 |
| **REPLAY：Full PF S2** | 初期知識はm12各4 shotの1点（615,304,502）だけ。①m12=2→適合で559,457,386へ更新。②m12=1→不適合。③次はM5の生成器候補537,125,648 |
| 次の候補が更新後の状態から作られる | 適合で更新した直後の減算は、更新後の適合点を起点にしている |
| **INGEST** | FULLモードのp7の要求（382,916,152）に、人工の適合結果（0.95τ）を1件与えた。最安がその点に更新され、次の要求はその点からの減算（351,866,322）に変わった |
| **FIXTURE**（人工MSE、`ARTIFICIAL_FIXTURE_NOT_SCIENCE`） | 何も知らない状態から開始した。12手のうち改善10手、不適合を受けた後も別のroundへ進み、予算12件で `Q3_BUDGET_EXHAUSTED` で停止。最初の改善の後も続くことを確認した |
| **OWN_PRICE**（PR） | 同じ2つの配分（LD3のFULL_PF__B_PRICEとLD3のVP）を、PRの6 configurationの単価でそれぞれ数えた。保存点がある組の費用はすべて一致した（例：FULL_PF__B_PRICEはLD3 541,936,295.95、LD5 541,743,905.57） |
| 正解の埋め込みが無い | 最適化器のsourceに、既知の最安額や配分列は含まれていない |
| テスト | 新11件と、再利用したCの集計テスト11件がPASS（[TESTS.txt](TESTS.txt)） |

REPLAYは「4→3→2→1」の経路をたどらない。費用の安い順の規則により、m12=2を先に問い合わせ、m12=1の不適合を受けた。そのうえで次の問い合わせを、保存結果の無い候補に決めた。既知の最安列や経路はcodeにも入力にも書いていない。入力にあるのは、開始点（seed）のラベルだけである。

## 4. 未評価候補と、次に必要なもの（`run01/NEEDS.json`）

| 方式 | 探索に入れる候補 | 不足のある候補 |
|---|---:|---|
| Full PF | 3（S2 16h/7、Y4 16h/3、M5 16h） | 17件で信号と価格が必要（h/2〜8hなど、索引にある他の幅） |
| B0I p7 | 3 | 価格だけ必要：S2・h（`e6754564`、信号あり）、粗幅Y4・32h/5（`dd8bfc24`、Q3あり）。信号と価格が必要：4件 |
| PR-Pauli | 3（L_D=2・3・5） | L_D=1・4・6・7・8・9でconfigurationから必要 |
| PR-DF | 3（L_D=1・2・7） | 価格だけ必要：L_D=9（`ea087a7e`、Q3適合）。L_D=3・4・5・6・8でconfigurationから必要 |

探索に入れる候補については、要求（`pending_Q3`）と次点の4候補を出している。

## 5. 実行後もなお人手が必要な箇所

1. **新規Q3の実行**：このversionには実行器が無い。要求は出るが、実行するには既存の方式別caller（入力の `q3_routes` に案内）と、探索1回分の対象・予算の承認が必要。結果は `--ingest` で戻す。
2. **信号・価格の取得**：不足のある候補（§4）の信号・価格は、A・D・D2の各runnerで別途取得する必要がある。取得後、Cの集計INPUTSに登録する作業も人手である。
3. **Cの集計に無い証拠は見えない**：例として、PR-PauliのL_D=1の保存信号（`714247619`）はCの集計に無いため、「configurationから必要」と表示される。Cへの登録が必要。
4. **問題入力の作成**：H4の入力fileは手で書いた。H・target・精度・方式の定義・候補範囲・shot上限・探索パラメータ・Q3経路の案内が含まれる。
5. **探索規則のパラメータ**：減算幅、headroomのゲート（9/10）、生成器の規則、shot上限は研究者が決めた値で、入力にある。
6. **PFの新しい幅**：索引に保存configurationがある幅だけを列挙する。索引に無い幅のconfiguration生成は自動化していない。
7. **別のH**：入力とcodeはHに依存しない形にしたが、Cの集計（保存Q3・価格）はH4にしか無い。新しいHでは、まず保存結果の集計が必要になる。

## 6. 範囲・資格

- 新規の科学計算・compile・Q3・価格の再較正は0件。費用は各候補の既存価格規則を変えずに使った。価格式の統一も行っていない。
- 結論は「評価済み集合内の最安適合構成」と「有限の探索規則による次の要求」である。十分N・最小N・物理的global RMSE・正式G・winnerではない。
- 探索規則は発見的である（費用の安い順、headroomによるゲート、支配による飛ばし）。飛ばした候補は不適合とは扱わず、記録に残す。
- FIXTUREとINGESTの人工結果は、科学結果ではない（出力に `ARTIFICIAL_*` と明記した）。
- Cの集計の入力・code、各原結果、旧STOPは変更していない。
