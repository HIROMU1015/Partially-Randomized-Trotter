# H4：方式別RPEコスト最適化の最初の実装（保存結果と人工fixtureで確認）

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
